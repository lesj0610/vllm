# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nothing a QSA step runs may reach a Triton kernel.

The Triton kernels are still here as the fallback, and the reference path in
the indexer still uses some of them. This pins down which of them a served
step is allowed to need: every Triton launch on the step path is replaced with
one that raises, and then the step's ops are run. An op that still lands on
Triton fails here rather than quietly costing what the CUDA one was written to
save.
"""

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_pre_indexer as pre_ops
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the QSA step runs on CUDA"
)

HC = 4
HIDDEN = 1024
HEAD_DIM = 128
CR = 4
EPS = 1e-6
STATE_SIZE = 4
COMP_PAGE = 4
MROPE_SECTION = (11, 11, 10)

# Every Triton launch a served step could reach.
TRITON_LAUNCHES = (
    (hc_ops, "_grouped_gemma_rmsnorm_kernel"),
    (hc_ops, "_hc_silu_kernel"),
    (hc_ops, "_hc_gate_mix_kernel"),
    (hc_ops, "_hc_combine_kernel"),
    (hc_ops, "_hc_combine_norm_kernel"),
    (pre_ops, "_qsa_pre_indexer_kernel"),
    (qsa_cache, "_build_qsa_metadata_kernel"),
)


class _ReachedTriton(RuntimeError):
    pass


class _Poisoned:
    """Stands in for a Triton kernel: raises however it is launched."""

    def __getitem__(self, _grid):
        return self

    def __call__(self, *_args, **_kwargs):
        raise _ReachedTriton("a Triton kernel was reached")


@pytest.fixture
def no_triton(monkeypatch):
    for module, name in TRITON_LAUNCHES:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _Poisoned())
    return None


@requires_cuda
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("tokens", [1, 8, 2048])
def test_hc_ops_never_reach_triton(no_triton, tokens):
    """The five hyper-connection ops the layer runs."""
    device = torch.device("cuda")
    dim = HIDDEN * HC
    bf = dict(dtype=torch.bfloat16, device=device)
    x = torch.randn(tokens, dim, **bf)
    gate = torch.randn(tokens, dim, **bf)
    block = torch.randn(tokens, HIDDEN, **bf)
    inject = torch.randn(tokens, HC, **bf)
    group_weight = torch.randn(HIDDEN, **bf) * 0.2
    row_weight = torch.randn(dim, **bf) * 0.2

    assert torch.isfinite(hc_ops.grouped_gemma_rmsnorm(x, group_weight, EPS, HC)).all()
    assert torch.isfinite(hc_ops.hc_silu(x, HC)).all()
    assert torch.isfinite(hc_ops.hc_gate_mix(x, gate, HC)).all()
    assert torch.isfinite(hc_ops.hc_combine(x, block, inject, HC)).all()
    combined, normed = hc_ops.hc_combine_norm(x, block, inject, row_weight, EPS, HC)
    assert torch.isfinite(combined).all()
    assert torch.isfinite(normed).all()


@requires_cuda
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("tokens", [1, 37])
def test_pre_indexer_never_reaches_triton(no_triton, mrope, tokens):
    """The fused pre-indexer, which is what a served step uses."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.common.qsa_cache import (
        circular_qsa_slot_mapping,
        compressed_qsa_slot_mapping,
    )

    device = torch.device("cuda")
    rope_params = {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    if mrope:
        rope_params["mrope_interleaved"] = True
        rope_params["mrope_section"] = list(MROPE_SECTION)
    with torch.device(device):
        rope = get_rope(
            head_size=256,
            max_position=32768,
            rope_parameters=rope_params,
            dtype=torch.bfloat16,
        )

    seq = 256
    token_to_req = torch.zeros(tokens, dtype=torch.int32, device=device)
    positions_1d = torch.arange(seq - tokens, seq, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    positions = (
        torch.stack([positions_1d, positions_1d // 7, positions_1d // 13])
        if mrope
        else positions_1d
    )
    raw_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    comp_blocks = (seq // CR + COMP_PAGE - 1) // COMP_PAGE
    comp_table = torch.arange(comp_blocks, dtype=torch.int32, device=device).reshape(
        1, comp_blocks
    )
    raw_slots = circular_qsa_slot_mapping(
        raw_table, token_to_req, positions_1d, STATE_SIZE, query_start_loc
    )
    comp_slots = compressed_qsa_slot_mapping(
        comp_table, token_to_req, positions_1d, COMP_PAGE, CR
    )
    work = torch.full(
        (max(1, (tokens + CR - 1) // CR), 2), -1, dtype=torch.int32, device=device
    )
    work[:, 0] = 0
    work[:, 1] = torch.arange(work.shape[0], dtype=torch.int32, device=device)

    bf = dict(dtype=torch.bfloat16, device=device)
    heads = 4
    qk = torch.randn(tokens, (heads + 1) * HEAD_DIM, **bf)
    q_out = torch.empty(tokens, heads, HEAD_DIM, **bf)
    raw = torch.zeros(1, STATE_SIZE, 1, HEAD_DIM + 12, **bf)
    comp = torch.zeros(comp_blocks, COMP_PAGE, 1, HEAD_DIM, **bf)

    pre_ops.qsa_pre_indexer(
        qk[:, : heads * HEAD_DIM],
        qk[:, heads * HEAD_DIM :],
        positions,
        rope.cos_sin_cache,
        torch.randn(HEAD_DIM, **bf) * 0.2,
        torch.randn(HEAD_DIM, **bf) * 0.2,
        EPS,
        q_out,
        raw,
        raw_slots,
        raw_table,
        query_start_loc,
        positions_1d,
        comp,
        comp_slots,
        work,
        compress_ratio=CR,
        mrope_section=MROPE_SECTION if mrope else None,
        rope_pos_offset=HEAD_DIM,
    )
    assert torch.isfinite(q_out).all()


@requires_cuda
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("requests,tokens_each", [(1, 1), (8, 1), (2, 64)])
def test_metadata_never_reaches_triton(no_triton, requests, tokens_each):
    """The per-step metadata the side cache is addressed with."""
    from types import SimpleNamespace

    device = torch.device("cuda")
    num_tokens = requests * tokens_each
    qsl_cpu = torch.arange(0, num_tokens + 1, tokens_each, dtype=torch.int32)
    seq = 512
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=qsl_cpu.to(device),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.full((requests,), seq, dtype=torch.int32, device=device),
        block_table_tensor=torch.arange(
            requests * (seq // 64), dtype=torch.int32, device=device
        ).reshape(requests, seq // 64),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device),
    )
    work = torch.empty(
        max(1, num_tokens // CR + requests), 2, dtype=torch.int32, device=device
    )
    token_to_req, positions, slots = qsa_cache.build_qsa_metadata(
        common,
        torch.empty(num_tokens, dtype=torch.int32, device=device),
        torch.empty(num_tokens, dtype=torch.int64, device=device),
        torch.empty(num_tokens, dtype=torch.int64, device=device),
        storage_block_size=16,
        compress_ratio=CR,
        k_work_metadata_buffer=work,
        request_capacity=requests,
    )
    assert token_to_req.numel() == num_tokens
    assert positions.numel() == num_tokens
    assert slots.numel() == num_tokens
