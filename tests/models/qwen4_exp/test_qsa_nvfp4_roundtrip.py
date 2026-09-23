# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the NVFP4 writer stores has to be what the QSA reader reads.

The other NVFP4 tests build their own packed bytes and check the reader
against a decode of those same bytes, so the reader agrees with itself no
matter what the writer actually lays down. Nothing exercised the join: the
FlashInfer slot writer the serving path calls, the slot views it is handed,
and the Triton reader that walks them. These two do, on a non-contiguous slot
mapping that fills both pages.
"""

import pytest
import torch

# Import the model package first: it pulls the QSA owner in, which breaks the
# cycle a bare ``from ...qsa import`` would hit here.
from vllm.models.qwen4_exp.nvidia import (  # noqa: F401
    model as _qwen4_exp_model,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

NUM_PAGES = 2
NUM_KV_HEADS = 2
PAGE_SIZE = 8
HEAD_DIM = 256
NUM_QUERY_HEADS = 8
NUM_QUERIES = 4
SELECTION_WIDTH = 8

# Every slot of both pages, in an order that is neither sequential nor
# page-ordered: a writer that ignores the mapping, folds the page into the
# offset, or walks the slots in arrival order lands somewhere else.
SLOT_PERMUTATION = (11, 3, 15, 0, 7, 12, 1, 9, 4, 14, 2, 10, 6, 13, 5, 8)

# Distinct and non-unit: a reader that drops one, applies it twice, or swaps K
# for V lands somewhere else. Equal or unit values would hide all three.
GLOBAL_K_SCALE = 0.375
GLOBAL_V_SCALE = 1.75


def _packed_selection(width: int, num_rows: int) -> torch.Tensor:
    """Selection rows 0..width-1 in the packed layout the kernel reads."""
    packed = torch.empty((num_rows, width + 1), device="cuda", dtype=torch.int32)
    packed[:, :width] = torch.arange(width, device="cuda", dtype=torch.int32)
    packed[:, width] = width
    return packed


def _write_slots(
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: float = GLOBAL_K_SCALE,
    v_scale: float = GLOBAL_V_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize through the writer the serving path uses; return the slot views."""
    import flashinfer

    writer = flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping
    full_dim = nvfp4_kv_cache_full_dim(HEAD_DIM)
    data_dim = HEAD_DIM // 2
    kv_cache = torch.zeros(
        NUM_PAGES,
        2 * NUM_KV_HEADS,
        PAGE_SIZE,
        full_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    # The serving allocation interleaves K and V over 2 * num_kv_heads slots.
    k_slot, v_slot = kv_cache[:, 0::2], kv_cache[:, 1::2]
    writer(
        key,
        value,
        torch.tensor(SLOT_PERMUTATION, dtype=torch.int32, device="cuda"),
        (k_slot[..., :data_dim], v_slot[..., :data_dim]),
        (
            k_slot[..., data_dim:].view(torch.float8_e4m3fn),
            v_slot[..., data_dim:].view(torch.float8_e4m3fn),
        ),
        torch.tensor(k_scale, dtype=torch.float32, device="cuda"),
        torch.tensor(v_scale, dtype=torch.float32, device="cuda"),
        kv_layout="HND",
    )
    return k_slot, v_slot


def _decode_slots(slots: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Decode a slot view to the dense BF16 page the unquantized path takes.

    Same arithmetic as the reader's contract: FP4 E2M1 data, one E4M3 block
    scale per 16 dims, and the global scale the writer was given, applied once.
    """
    data_bytes = HEAD_DIM // 2
    data = slots[..., :data_bytes]
    scales = slots[..., data_bytes:].view(torch.float8_e4m3fn).float()
    low = (data & 0x0F).to(torch.int32)
    high = (data >> 4).to(torch.int32)
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    sign = torch.where(nibbles & 8 != 0, -1.0, 1.0)
    exponent = (nibbles >> 1) & 3
    mantissa = (nibbles & 1).float()
    magnitude = torch.where(
        exponent > 0,
        torch.exp2(exponent.float() - 2.0) * (2.0 + mantissa),
        mantissa * 0.5,
    )
    block_scale = scales.repeat_interleave(16, dim=-1)
    # Slot views are [pages, heads, tokens, width]; the BF16 path reads
    # [pages, tokens, heads, dim].
    return (
        (sign * magnitude * block_scale * global_scale)
        .to(torch.bfloat16)
        .permute(0, 2, 1, 3)
    )


def _gather_written_rows(decoded: torch.Tensor) -> torch.Tensor:
    """Read back the rows in the order the slot mapping wrote them."""
    slots = torch.tensor(SLOT_PERMUTATION, dtype=torch.long, device=decoded.device)
    return decoded[slots // PAGE_SIZE, slots % PAGE_SIZE]


@requires_qsa_kernels
def test_nvfp4_writer_places_every_slot_where_the_reader_looks() -> None:
    """The mapping the writer is given has to be the mapping the reader sees."""
    torch.manual_seed(7)
    rows = len(SLOT_PERMUTATION)
    key = torch.randn(rows, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    k_slot, v_slot = _write_slots(key, value)

    for slots, source, scale, side in (
        (k_slot, key, GLOBAL_K_SCALE, "K"),
        (v_slot, value, GLOBAL_V_SCALE, "V"),
    ):
        got = _gather_written_rows(_decode_slots(slots, scale)).float()
        want = source.float()
        # FlashInfer's own NVFP4 acceptance: mean relative error and cosine
        # similarity, not an elementwise bound -- a 4-bit mantissa cannot
        # carry one. The contract under test is placement and scaling, and
        # both fail these by orders of magnitude when they are wrong.
        rel = ((got - want).abs().sum() / want.abs().sum()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), want.flatten(), dim=0
        ).item()
        assert rel < 0.3, f"{side} mean relative error {rel:.3f}"
        assert cos > 0.9, f"{side} cosine similarity {cos:.3f}"


@requires_qsa_kernels
def test_qsa_reads_the_cache_the_nvfp4_writer_produced() -> None:
    """The packed reader and the BF16 path must agree on written slots.

    This is the whole join: ``qsa_kv_dispatch``, the slot-view strides, the
    FP4 and E4M3 decoders, and the global K/V scales the host folds in.
    """
    torch.manual_seed(11)
    rows = len(SLOT_PERMUTATION)
    key = torch.randn(rows, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_slot, v_slot = _write_slots(key, value)

    q = torch.randn(
        NUM_QUERIES, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    block_table = torch.arange(NUM_PAGES, device="cuda", dtype=torch.int32).reshape(
        1, -1
    )
    token_to_req = torch.zeros(NUM_QUERIES, device="cuda", dtype=torch.int32)
    logical_indices = _packed_selection(SELECTION_WIDTH, NUM_QUERIES)
    # One gate for both calls: it is applied to the output, so a fresh draw
    # would move the two apart for a reason that is not under test.
    output_gate = torch.randn_like(q)

    packed = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_slot,
        v_slot,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=GLOBAL_K_SCALE,
        v_scale=GLOBAL_V_SCALE,
        nvfp4=True,
        output_gate=output_gate,
    )
    reference = qsa_ops.qsa_sparse_paged_attention(
        q,
        _decode_slots(k_slot, GLOBAL_K_SCALE),
        _decode_slots(v_slot, GLOBAL_V_SCALE),
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        output_gate=output_gate,
    )
    torch.testing.assert_close(packed.float(), reference.float(), rtol=2e-2, atol=2e-2)
