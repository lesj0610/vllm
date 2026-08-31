# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every supported QSA configuration has to reach FlashInfer, not Triton.

The Triton kernels are still present as a fallback. This pins down which
configurations are allowed to need them: the attention step replaces each
legacy entry point with one that raises, then runs the whole support matrix. A
configuration that still lands on Triton fails here rather than quietly
regressing to the slower path.
"""

import itertools

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

flashinfer = pytest.importorskip("flashinfer")

from vllm.models.qwen4_exp.nvidia.ops.qsa_flashinfer import (  # noqa: E402
    QSAFlashInferRunner,
    supports_qsa_flashinfer,
)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="QSA runs on CUDA"
)

PAGE_SIZE = 64
# What the model config accepts for its main KV cache.
SUPPORTED_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")
LEGACY_ENTRY_POINTS = (
    "qsa_sparse_paged_attention",
    "qsa_mqa_paged",
    "expand_qsa_block_indices_cuda",
)


class _ReachedTriton(RuntimeError):
    pass


def _raise(*_args, **_kwargs):
    raise _ReachedTriton("a Triton QSA kernel was reached")


def _build(dtype, rows, heads, head_dim, seq_len, num_requests, width, device):
    g = torch.Generator(device=device).manual_seed(rows + head_dim)
    pages_per_request = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pages = pages_per_request * num_requests
    block_table = (
        torch.randperm(pages, device=device, generator=g)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    per_request = max(1, rows // num_requests)
    token_to_req = (
        (torch.arange(rows, device=device, dtype=torch.int32) // per_request)
        .clamp(max=num_requests - 1)
        .contiguous()
    )
    query = torch.randn(
        rows, heads, head_dim, dtype=torch.bfloat16, device=device, generator=g
    )
    logical = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        count = min(seq_len, width)
        logical[row, :count] = torch.randperm(seq_len, device=device, generator=g)[
            :count
        ].to(torch.int32)

    if dtype == "nvfp4":
        full = nvfp4_kv_cache_full_dim(head_dim)
        kv = torch.randint(
            0,
            256,
            (pages, 2, PAGE_SIZE, full),
            device=device,
            dtype=torch.uint8,
            generator=g,
        )
        scales = (
            torch.rand(
                pages, 2, PAGE_SIZE, full - head_dim // 2, device=device, generator=g
            )
            + 0.5
        )
        kv[..., head_dim // 2 :] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
        return (
            query,
            kv[:, 0::2],
            kv[:, 1::2],
            block_table,
            token_to_req,
            logical,
            "HND",
        )

    keys = torch.randn(
        pages,
        PAGE_SIZE,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    )
    values = torch.randn_like(keys)
    if dtype.startswith("fp8"):
        keys = (keys.float() / 0.5).to(torch.float8_e4m3fn)
        values = (values.float() / 0.5).to(torch.float8_e4m3fn)
    return query, keys, values, block_table, token_to_req, logical, "NHD"


@requires_cuda
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_every_supported_dtype_is_served_by_flashinfer(dtype, head_dim):
    """The gate has to accept every cache dtype the model config allows."""
    assert supports_qsa_flashinfer(head_dim, dtype), (
        f"{dtype} at head_dim {head_dim} would fall back to Triton"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", ("auto", "fp8", "nvfp4"))
@pytest.mark.parametrize(
    "head_dim,rows", list(itertools.product((128, 256), (1, 8, 64)))
)
def test_attention_never_reaches_triton(monkeypatch, dtype, head_dim, rows):
    """With the Triton entry points poisoned, the step still has to complete."""
    device = torch.device("cuda")
    if not supports_qsa_flashinfer(head_dim, dtype):
        pytest.skip(f"{dtype} at head_dim {head_dim} is not served by FlashInfer")

    query, keys, values, block_table, token_to_req, logical, layout = _build(
        dtype, rows, 12, head_dim, 1024, 2, 512, device
    )
    out = torch.empty_like(query)
    runner = QSAFlashInferRunner(device)
    for name in LEGACY_ENTRY_POINTS:
        monkeypatch.setattr(qsa_ops, name, _raise)

    if dtype == "nvfp4":
        data = head_dim // 2
        runner.run(
            query,
            keys[..., :data],
            values[..., :data],
            keys[..., data:],
            values[..., data:],
            logical,
            block_table,
            token_to_req,
            out,
            PAGE_SIZE,
            layout,
            1.0,
            1.0,
        )
    else:
        scale = 0.5 if dtype.startswith("fp8") else 1.0
        runner.run(
            query,
            keys,
            values,
            None,
            None,
            logical,
            block_table,
            token_to_req,
            out,
            PAGE_SIZE,
            layout,
            scale,
            scale,
        )
    # Reading the tensor is enough of a barrier for the assertion below.
    assert torch.isfinite(out).all()
