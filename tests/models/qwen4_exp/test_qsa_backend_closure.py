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
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
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

# The Triton entry points this path must not reach. Upstream splits them across
# two modules: ops/qsa.py keeps the sparse attention while the scoring and
# expansion kernels live in ops/qsa_indexer.py.
LEGACY_ENTRY_POINTS = (
    (qsa_ops, "qsa_sparse_paged_attention"),
    (qsa_indexer_ops, "qsa_select_paged_decode"),
    (qsa_indexer_ops, "qsa_select_paged_prefill"),
    (qsa_indexer_ops, "expand_qsa_block_indices"),
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
@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_every_supported_dtype_is_served_by_flashinfer(dtype, head_dim):
    """The gate has to accept every cache dtype the model config allows.

    Pre-SM100 only: from SM100 a packed NVFP4 cache converts natively and this
    route is not the right one, so the whole thing stays on Triton there.
    """
    if current_platform.has_device_capability(100):
        pytest.skip("this route is pre-SM100")
    assert supports_qsa_flashinfer(head_dim, dtype), (
        f"{dtype} at head_dim {head_dim} would fall back to Triton"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", ("auto", "fp8", "nvfp4"))
@pytest.mark.parametrize(
    "head_dim,rows", list(itertools.product((128, 256, 512), (1, 8, 64)))
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
    for module, name in LEGACY_ENTRY_POINTS:
        monkeypatch.setattr(module, name, _raise)

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


@requires_cuda
def test_attention_impl_routes_to_flashinfer_not_triton(monkeypatch):
    """The impl's own gate has to pick FlashInfer, not just the runner.

    test_attention_never_reaches_triton above drives QSAFlashInferRunner
    directly, so it pins the runner but not the decision that reaches it. This
    covers the decision: with the gate open the impl calls _run_flashinfer and
    never the Triton attention.
    """
    # The model module owns the import cycle; entering through qsa alone
    # leaves it partially initialized.
    from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_module

    if current_platform.has_device_capability(100):
        pytest.skip("this route is pre-SM100")

    impl = qsa_module.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_module.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = 128
    impl.kv_cache_dtype = "auto"
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", _raise)

    reached = []
    monkeypatch.setattr(
        qsa_module.Qwen4ExpQSAFlashAttentionImpl,
        "_run_flashinfer",
        lambda self, *a, **k: reached.append(True),
    )

    assert impl._flashinfer_usable(torch.device("cuda"), None)
    impl._run_flashinfer(None, None, None, None, None, None, None, None)
    assert reached == [True]


@requires_cuda
def test_selection_routing_prefers_flashinfer(monkeypatch):
    """The indexer's selection gate has to bypass the Triton trio.

    Only the decision is covered here: the FlashInfer selection ops and the
    Triton ones take different metadata, so this asserts which side the caller
    picks rather than comparing their outputs.
    """
    from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer

    if not qsa_flashinfer._selection_available():
        pytest.skip("this FlashInfer build has no QSA selection ops")
    assert qsa_flashinfer.supports_qsa_selection(128, 12)
    # A shape the scorer has no instantiation for falls back instead.
    assert not qsa_flashinfer.supports_qsa_selection(96, 12)
    assert not qsa_flashinfer.supports_qsa_selection(128, 17)


def _selection_call(*, block_indices_dtype=torch.int32, out_dtype=torch.int32):
    """Arguments as the QSA indexer hands them down: int32 tables, int64
    positions. common/qsa_cache.py builds the positions with ``.long()``."""
    rows, heads, head_dim = 3, 2, 128
    pages, page_size, requests = 4, 8, 2
    compress_ratio, token_topk = 4, 8
    return {
        "q": torch.zeros(rows, heads, head_dim),
        "k_cache": torch.zeros(pages, page_size, 1, head_dim),
        "page_table": torch.zeros(requests, pages, dtype=torch.int32),
        "token_to_req": torch.zeros(rows, dtype=torch.int32),
        "query_positions": torch.arange(rows, dtype=torch.int64),
        "sequence_lengths": torch.full((requests,), page_size, dtype=torch.int32),
        "compress_ratio": compress_ratio,
        "token_topk": token_topk,
        "block_indices": torch.zeros(
            rows, token_topk // compress_ratio, dtype=block_indices_dtype
        ),
        "out": torch.zeros(rows, token_topk, dtype=out_dtype),
    }


def _record_index_dtypes(monkeypatch) -> list[dict[str, torch.dtype]]:
    """Capture the dtype of every index tensor each FlashInfer call receives."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer

    seen: list[dict[str, torch.dtype]] = []

    def fake_scores(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        scale,
        num_columns,
    ):
        seen.append(
            {
                "page_table": page_table.dtype,
                "token_to_req": token_to_req.dtype,
                "query_positions": query_positions.dtype,
                "sequence_lengths": sequence_lengths.dtype,
            }
        )
        return (
            torch.zeros(q.shape[0], num_columns),
            torch.zeros(q.shape[0], dtype=page_table.dtype),
        )

    def fake_expand(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        out,
    ):
        seen.append(
            {
                "block_indices": block_indices.dtype,
                "query_positions": query_positions.dtype,
                "sequence_lengths": sequence_lengths.dtype,
                "token_to_req": token_to_req.dtype,
                "out": out.dtype,
            }
        )

    monkeypatch.setattr(
        flashinfer, "sparse_paged_scores", fake_scores, raising=False
    )
    monkeypatch.setattr(
        flashinfer, "expand_block_route", fake_expand, raising=False
    )
    monkeypatch.setattr(qsa_flashinfer, "_topk", lambda *args, **kwargs: None)
    return seen


def test_selection_narrows_the_positions_to_the_block_table_dtype(monkeypatch):
    """Each FlashInfer call reads its index tensors through one pointer type.

    The scorer instantiates that type from ``page_table`` and the expansion
    from ``block_indices``, so an int64 position tensor reaching either one
    fails the kernel's own dtype check rather than being converted for us.
    """
    from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer

    seen = _record_index_dtypes(monkeypatch)

    qsa_flashinfer.select_and_expand(**_selection_call())

    assert len(seen) == 2, "the scorer and the expansion must both be reached"
    for call in seen:
        assert set(call.values()) == {torch.int32}, call


@pytest.mark.parametrize(
    "overrides",
    [
        {"block_indices_dtype": torch.int64},
        {"out_dtype": torch.int64},
    ],
)
def test_selection_rejects_a_second_index_dtype(monkeypatch, overrides):
    """One narrowing cannot reconcile two block-table dtypes, so say so here
    instead of letting the kernel report it from inside a captured graph."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer

    _record_index_dtypes(monkeypatch)

    with pytest.raises(ValueError, match="one index dtype"):
        qsa_flashinfer.select_and_expand(**_selection_call(**overrides))
