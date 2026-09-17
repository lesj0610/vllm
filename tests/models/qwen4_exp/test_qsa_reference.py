# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)


def _packed_selection(width: int, num_rows: int, device: str = "cuda") -> torch.Tensor:
    """Selection rows 0..width-1 in the packed layout the kernel reads.

    The trailing column carries the row's valid-entry count, so a test that
    hands the kernel a bare index range would lose its last column to it.
    """
    packed = torch.empty((num_rows, width + 1), device=device, dtype=torch.int32)
    packed[:, :width] = torch.arange(width, device=device, dtype=torch.int32)
    packed[:, width] = width
    return packed


def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)
    raw_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        block_table=torch.empty(0),
        query_start_loc=torch.arange(3),
        logical_positions=torch.arange(2),
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        k_work_metadata=torch.empty(0),
    )
    updates = []
    selections = []
    indexer = SimpleNamespace(
        skip_topk=True,
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=1,
        indexer_dtype=torch.bfloat16,
        raw_key_cache=SimpleNamespace(
            kv_cache=torch.empty(0),
            rope_position_cache=None,
            rope_position_offset=0,
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=torch.empty(0)),
        use_fused_pre_indexer=True,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(0)),
        q_layernorm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.ones(1)),
        compress_ratio=2,
    )

    monkeypatch.setattr(
        indexer_qsa,
        "qsa_pre_indexer",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        qsa_indexer_ops,
        "qsa_select_paged_decode",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )
    monkeypatch.setattr(
        qsa_indexer_ops,
        "qsa_select_paged_prefill",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )

    actual = indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.zeros(2, 2),
        torch.tensor([7, 8]),
        rows,
    )

    assert actual is rows
    assert len(updates) == 1
    assert not selections


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_select_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        row_sequence_lengths // compress_ratio,
    ).to(torch.int32)
    logits = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
    )
    starts = torch.zeros_like(visible_blocks)
    return _qsa_relative_topk_reference(
        logits,
        starts,
        visible_blocks,
        token_topk // compress_ratio,
    )


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Dense reference for QSA sparse paged attention.

    Mirrors the kernel's dequant: fp8-e4m3 K/V caches are dequantized with the
    per-tensor k_scale/v_scale host floats; bf16 caches use unit scales.
    """
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = (k_cache[pages, offsets].float() * k_scale).repeat_interleave(
            repeats, dim=1
        )
        values = (v_cache[pages, offsets].float() * v_scale).repeat_interleave(
            repeats, dim=1
        )
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys)
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values).to(q.dtype)
    return output


@requires_qsa_kernels
def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.reorder_batch_threshold = 4
    builder.is_circular_buffer = False
    builder.storage_block_size = 64
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [0] * 4, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=4,
        max_query_len=4,
        max_seq_len=68,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4, device=device),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.logical_positions.tolist() == [
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        -1,
        -1,
        -1,
        -1,
    ]
    assert metadata.slot_mapping.tolist() == list(range(12)) + [-1] * 4
    assert metadata.visible_blocks.tolist() == [65, 66, 67, 68] * 3 + [0] * 4


@requires_qsa_kernels
def test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.reorder_batch_threshold = 1
    builder.is_circular_buffer = True
    builder.kv_cache_spec = SimpleNamespace(block_size=4)
    builder.storage_block_size = 4
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 7, 13, 13], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 7 + [1] * 6 + [0] * 3, device=device)
    block_table = torch.tensor([[1], [0], [2]], dtype=torch.int32, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=3,
        max_query_len=7,
        max_seq_len=11,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([9, 11, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.full((16,), -1, dtype=torch.int64, device=device),
        block_table_tensor=block_table,
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)
    expected = [
        -1,
        -1,
        -1,
        5,
        6,
        7,
        4,
        -1,
        -1,
        3,
        0,
        1,
        2,
        -1,
        -1,
        -1,
    ]

    assert metadata.slot_mapping.tolist() == expected


@pytest.mark.parametrize("chunk_start", list(range(8)))
def test_qsa_circular_buffer_survives_one_speculative_step(chunk_start: int) -> None:
    """A speculative step must not overwrite the open group's committed keys.

    The step stores every row it computes, drafts included, before acceptance
    is known, while the next step still reads the earlier members of the group
    being compressed from the ring. A ring sized at the compression ratio makes
    those rows alias, so a rejected draft silently replaces a committed key.
    """
    compress_ratio = 4
    num_spec = 3
    capacity = compress_ratio * -(-(compress_ratio + num_spec) // compress_ratio)
    query_len = num_spec + 1

    slots = qsa_cache.circular_qsa_slot_mapping(
        torch.tensor([[0]], dtype=torch.int32),
        torch.zeros(query_len, dtype=torch.int32),
        torch.arange(chunk_start, chunk_start + query_len),
        capacity,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
    )

    committed = torch.arange(chunk_start - chunk_start % compress_ratio, chunk_start)
    assert set(slots.tolist()).isdisjoint((committed % capacity).tolist())


def _qsa_key_cache(block_size: int, compress_ratio: int) -> qsa_cache.QSAKeyStateCache:
    return qsa_cache.QSAKeyStateCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=block_size),
        prefix=f"raw.{block_size}.{compress_ratio}",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=compress_ratio,
    )


def test_qsa_state_caches_adapt_the_unified_logical_layout() -> None:
    raw_cache = _qsa_key_cache(block_size=32, compress_ratio=4)
    compressed_cache = qsa_cache.QSACompressedKeyCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=32),
        prefix="compressed.bind",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=4,
    )
    raw_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)
    compressed_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)

    raw_cache.bind_kv_cache(raw_view)
    compressed_cache.bind_kv_cache(compressed_view)

    assert raw_cache.kv_cache.shape == (2, 8, 1, 64)
    assert compressed_cache.kv_cache.shape == (2, 8, 1, 64)
    assert raw_cache.kv_cache.data_ptr() == raw_view.data_ptr()
    assert compressed_cache.kv_cache.data_ptr() == compressed_view.data_ptr()


@pytest.mark.parametrize(
    ("compress_ratio", "num_spec", "expected"),
    [(4, 0, 4), (4, 1, 8), (4, 3, 8), (4, 4, 8), (4, 5, 12), (2, 3, 6)],
)
def test_qsa_ring_capacity_covers_one_speculative_step(
    compress_ratio: int, num_spec: int, expected: int
) -> None:
    """Capacity spans the open group plus one speculative step, in whole groups."""
    spec = _qsa_key_cache(
        block_size=48, compress_ratio=compress_ratio
    ).get_kv_cache_spec(SimpleNamespace(num_speculative_tokens=num_spec))
    assert spec.block_size == expected


@requires_qsa_kernels
def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.reorder_batch_threshold = 1
    builder.is_circular_buffer = False
    builder.storage_block_size = 16
    builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(8, dtype=torch.int32, device=device)
    # Simulate max_num_seqs exceeding the three live requests below.
    builder.request_capacity = 8
    builder.k_work_metadata_buffer = torch.empty(4, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 3, 8], dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0, 0, 0, 2, 2, 2, 2, 2], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=8,
        num_reqs=3,
        max_query_len=5,
        max_seq_len=12,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([7, 0, 12], dtype=torch.int32, device=device),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.zeros((3, 1), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.slot_mapping.tolist() == [-1] * 8
    assert metadata.visible_blocks.tolist() == [1, 1, 1, 2, 2, 2, 2, 3]
    assert metadata.k_work_metadata.tolist() == [[0, 0], [2, 0], [2, 1], [-1, -1]]


@requires_qsa_kernels
@pytest.mark.usefixtures("default_vllm_config")
def test_qsa_unfused_cache_update_ignores_padded_qk() -> None:
    """Padded projected Q/K rows must not affect either side cache."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    device = torch.device("cuda")
    # Five tokens complete one compressed group and retain four keys in the ring.
    raw_metadata = SimpleNamespace(
        num_actual_tokens=5,
        slot_mapping=torch.tensor([-1, 1, 2, 3, 0], device=device),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        token_to_req=torch.zeros(5, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        logical_positions=torch.arange(5, device=device),
    )
    compressed_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([-1, -1, -1, 0, -1], device=device),
    )
    with torch.device(device):
        rope = get_rope(
            head_size=128,
            max_position=32,
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
            dtype=torch.bfloat16,
        )
    raw_cache = torch.zeros((1, 4, 1, 64), dtype=torch.bfloat16, device=device)
    compressed_cache = torch.zeros((1, 2, 1, 64), dtype=torch.bfloat16, device=device)
    norm = SimpleNamespace(
        weight=torch.zeros(64, dtype=torch.bfloat16, device=device),
        variance_epsilon=1e-6,
    )
    indexer = SimpleNamespace(
        _metadata=lambda: (raw_metadata, compressed_metadata),
        skip_topk=True,
        index_kv_heads=1,
        use_fused_pre_indexer=False,
        index_n_heads=1,
        index_head_dim=64,
        indexer_dtype=torch.bfloat16,
        q_layernorm=norm,
        k_layernorm=norm,
        rotary_emb=rope,
        compress_ratio=4,
        raw_key_cache=SimpleNamespace(
            kv_cache=raw_cache, key_cache=raw_cache, rope_position_cache=None
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=compressed_cache),
    )
    keys = torch.arange(1, 6, dtype=torch.bfloat16, device=device)[:, None].expand(
        5, 64
    )
    padded_keys = torch.full((8, 64), torch.nan, dtype=torch.bfloat16, device=device)
    padded_keys[:5].copy_(keys)
    indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.cat((torch.ones_like(padded_keys), padded_keys), dim=-1),
        torch.zeros(8, dtype=torch.long, device=device),
        torch.full((5, 5), -1, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(raw_cache[0, :, 0], keys[[4, 1, 2, 3]])
    expected_compressed = torch.zeros_like(compressed_cache)
    expected_compressed[0, 0] = 1
    torch.testing.assert_close(compressed_cache, expected_compressed)


@requires_qsa_kernels
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("num_reqs", [2, 3, 4, 7, 8, 9])
def test_qsa_triton_metadata_matches_pytorch(
    compress_ratio: int, num_reqs: int
) -> None:
    device = torch.device("cuda")
    num_tokens = 8
    query_start_loc = torch.tensor(
        [0, 3, *([3] * (num_reqs - 2)), 8], dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor(
        [0, 0, 0, *([num_reqs - 1] * 5)],
        dtype=torch.int32,
        device=device,
    )
    block_table_rows = torch.tensor(
        [
            [4, -1, 8, -1, 12, -1],
            [1, -1, 2, -1, 3, -1],
            [7, -1, 9, -1, 11, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    block_table_storage = block_table_rows[
        torch.arange(num_reqs, device=device) % block_table_rows.shape[0]
    ]
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    seq_lens[0] = 10
    seq_lens[-1] = 20
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        slot_mapping=torch.tensor(
            [0, 1, -1, 3, 4, -1, -1, -1], dtype=torch.int64, device=device
        ),
        block_table_tensor=block_table_storage[:, ::2],
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (
        (num_tokens + (compress_ratio - 1) * num_reqs) // compress_ratio
        if compress_ratio != 1
        else 0
    )
    actual_k_work = (
        torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
        if max_num_work
        else None
    )
    actual_buffers = make_buffers()
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *actual_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=actual_k_work,
        request_capacity=num_reqs,
    )

    expected_k_work = (
        torch.empty_like(actual_k_work) if actual_k_work is not None else None
    )
    expected_buffers = make_buffers()
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *expected_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=expected_k_work,
        request_capacity=num_reqs,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    if actual_k_work is not None:
        torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
def test_qsa_fused_metadata_matches_pytorch_for_large_padded_prefill() -> None:
    device = torch.device("cuda")
    num_mapped_tokens = 4096
    num_tokens = 4224
    query_start_loc = torch.tensor(
        [0, num_mapped_tokens], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor(
            [num_mapped_tokens + 32], dtype=torch.int32, device=device
        ),
        block_table_tensor=torch.arange(256, dtype=torch.int32, device=device)[None],
        slot_mapping=torch.tensor(
            [0] * num_mapped_tokens + [-1] * (num_tokens - num_mapped_tokens),
            dtype=torch.int64,
            device=device,
        ),
        token_to_req_indices=lambda buffer: buffer.zero_(),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (num_tokens + 3) // 4
    actual_k_work = torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
    expected_k_work = torch.empty_like(actual_k_work)
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=actual_k_work,
    )
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=expected_k_work,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("decode_query_len", "num_requests"),
    [
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (4, 33),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_decode_selection_correctness(
    decode_query_len: int, num_requests: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(1)
    heads, head_dim = 4, 128
    rows = num_requests * decode_query_len
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    page_size, pages_per_request, max_sequence_length = (
        (16, 40, 2560) if num_requests > 32 else (4, 20, 320)
    )
    num_pages = num_requests * pages_per_request
    cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        num_requests, pages_per_request
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(num_requests, device="cuda", dtype=torch.int32),
        decode_query_len,
    )
    sequence_lengths = max_sequence_length - 4 * (
        torch.arange(num_requests, device="cuda", dtype=torch.int32) % 8
    )
    query_positions = torch.cat(
        [
            torch.arange(
                length - decode_query_len,
                length,
                device="cuda",
                dtype=torch.int32,
            )
            for length in sequence_lengths.tolist()
        ]
    )
    visible_blocks = torch.minimum(
        (query_positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    )

    token_topk, compress_ratio = 2048, 4
    actual = torch.empty(
        (rows, token_topk // compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.qsa_select_paged_decode(
        q,
        cache,
        page_table,
        visible_blocks,
        token_topk,
        compress_ratio,
        decode_query_len,
        actual,
    )
    expected = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    if dtype == torch.float8_e4m3fn:
        # fp8 logits tie at the top-k boundary more often than bf16, so index
        # identity is not stable; compare the selected value multisets.
        # SM90 wgmma accumulates fp8 in reduced precision (~3e-4 abs
        # observed); SM100 tcgen05 is exact fp32.
        rtol = atol = 1e-3 if current_platform.is_device_capability(90) else None
        logits = _qsa_mqa_paged_reference(
            q, cache, page_table, token_to_req, visible_blocks
        )
        for row in range(rows):
            selected = actual[row][actual[row] >= 0]
            wanted = expected[row][expected[row] >= 0]
            assert selected.numel() == wanted.numel()
            torch.testing.assert_close(
                logits[row, selected.long()].sort().values,
                logits[row, wanted.long()].sort().values,
                rtol=rtol,
                atol=atol,
            )
        return

    torch.testing.assert_close(actual.sort().values, expected.sort().values)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("seq_len_slack", [0, 1792])
@pytest.mark.parametrize("force_chunk", [False, True])
def test_qsa_prefill_selection_correctness(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_slack: int,
    force_chunk: bool,
    dtype: torch.dtype,
) -> None:
    # page_size=24 (does not divide the 64-aligned clipped width) and an
    # oversized page table, so the clipped logits width comes from
    # max_seq_len, not page geometry. seq_len_slack > 0 simulates the
    # spec-decode case where the bound is an over-estimate. force_chunk
    # drives the logits budget to one row per chunk.
    if force_chunk:
        monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0")
    torch.manual_seed(2)
    query_lens = [3, 33]
    rows, heads, head_dim = sum(query_lens), 4, 128
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    cache = torch.randn(128, 24, 1, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    page_table = torch.randperm(128, device="cuda", dtype=torch.int32).reshape(2, 64)
    token_to_req = torch.repeat_interleave(
        torch.arange(2, device="cuda", dtype=torch.int32),
        torch.tensor(query_lens, device="cuda"),
    )
    query_start_loc = torch.tensor([0, 3, 36], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([5120, 4224], device="cuda", dtype=torch.int32)
    query_positions = torch.cat(
        [
            torch.arange(length - query_len, length, device="cuda", dtype=torch.int32)
            for query_len, length in zip(
                query_lens, sequence_lengths.tolist(), strict=True
            )
        ]
    )
    token_topk, compress_ratio = 2048, 4
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    )

    actual = torch.empty(
        (rows, token_topk // compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.qsa_select_paged_prefill(
        q,
        cache,
        page_table,
        query_start_loc,
        visible_blocks,
        token_topk,
        compress_ratio,
        max(query_lens),
        actual,
        max_seq_len=sequence_lengths.max().item() + seq_len_slack,
    )
    expected = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    if dtype == torch.float8_e4m3fn:
        # fp8 logits tie at the top-k boundary more often than bf16, so index
        # identity is not stable; compare the selected value multisets.
        # SM90 wgmma accumulates fp8 in reduced precision (~3e-4 abs
        # observed); SM100 tcgen05 is exact fp32.
        rtol = atol = 1e-3 if current_platform.is_device_capability(90) else None
        logits = _qsa_mqa_paged_reference(
            q, cache, page_table, token_to_req, visible_blocks
        )
        for row in range(rows):
            selected = actual[row][actual[row] >= 0]
            wanted = expected[row][expected[row] >= 0]
            assert selected.numel() == wanted.numel()
            torch.testing.assert_close(
                logits[row, selected.long()].sort().values,
                logits[row, wanted.long()].sort().values,
                rtol=rtol,
                atol=atol,
            )
        return

    torch.testing.assert_close(actual.sort().values, expected.sort().values)


@requires_qsa_kernels
def test_qsa_block_expansion_correctness() -> None:
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda")
    sequence_lengths = torch.tensor([6, 11], device="cuda")
    token_to_req = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    visible_blocks = torch.minimum(
        (query_positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    ).to(torch.int32)

    # Packed layout: one trailing column per row holds the valid-entry count
    # (never a token index). Row 0: 1 visible block + 2 tail; row 1: 2 blocks
    # + 3 tail.
    actual = torch.empty((2, 12), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.expand_qsa_block_indices(
        blocks,
        query_positions,
        visible_blocks,
        compress_ratio=4,
        token_topk=8,
        out=actual,
    )
    expected = _expand_qsa_indices_reference(
        blocks,
        query_positions,
        sequence_lengths,
        compress_ratio=4,
        token_topk=8,
    )

    torch.testing.assert_close(actual[:, :11], expected)
    assert actual[:, 11].tolist() == [6, 11]


@requires_qsa_kernels
@pytest.mark.parametrize(
    (
        "num_rows",
        "num_query_heads",
        "num_kv_heads",
        "page_size",
        "use_prefill_config",
        "num_requests",
        "fp8",
    ),
    [
        # Production page sizes from hybrid-cache block alignment: 784/800
        # at TP4 and 1568/1600 at TP1/TP2 (no-MTP / MTP num_spec=3). Head
        # splits are per-rank TP1/TP2/TP4; the largest batch runs both
        # use_prefill_config variants.
        pytest.param(1, 24, 2, 1600, True, 2, False, id="tp1_r1"),
        pytest.param(16, 12, 1, 1600, True, 3, False, id="tp2_r16"),
        pytest.param(32, 6, 1, 800, True, 5, False, id="tp4_r32"),
        pytest.param(128, 24, 2, 1568, True, 7, False, id="tp1_r128"),
        pytest.param(257, 6, 1, 800, True, 13, False, id="tp4_r257"),
        pytest.param(513, 6, 1, 784, True, 17, False, id="tp4_r513"),
        pytest.param(700, 6, 1, 800, True, 23, False, id="tp4_r700"),
        pytest.param(1024, 24, 2, 1600, True, 33, False, id="tp1_r1024"),
        pytest.param(2048, 24, 2, 1600, True, 63, False, id="tp1_r2048_prefill"),
        pytest.param(2048, 24, 2, 1600, False, 63, False, id="tp1_r2048_uniform"),
        # fp8_e4m3 K/V caches on the TP1 head split.
        pytest.param(1, 24, 2, 1600, True, 2, True, id="tp1_r1_fp8"),
        pytest.param(128, 24, 2, 1568, True, 7, True, id="tp1_r128_fp8"),
        pytest.param(2048, 24, 2, 1600, True, 63, True, id="tp1_r2048_prefill_fp8"),
        pytest.param(2048, 24, 2, 1600, False, 63, True, id="tp1_r2048_uniform_fp8"),
    ],
)
def test_qsa_sparse_paged_attention_correctness(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
    use_prefill_config: bool,
    num_requests: int,
    fp8: bool,
) -> None:
    """QSA sparse paged attention matches the dense reference.

    fp8 only changes the K/V cache dtype (e4m3 with a per-tensor scale pair) and
    the scales; the reference dequantizes the same cache with those scales, so
    both paths compare the production kernel against the reference on identical
    inputs. fp8=True additionally covers the host-side scale folding.
    """
    torch.manual_seed(2)
    # One QSA attention problem: bf16 Q and paged K/V, a packed selection with
    # the trailing count column, block table and row-to-request map.
    head_dim = 256
    num_selected_pages = 64
    # Keep the newest page outside the synthetic top-k as causal headroom.
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    output_gate = torch.randn_like(q)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    # Uniform row split; the last request takes the remainder (possibly 0).
    request_row_counts = torch.full(
        (num_requests,), rows_per_request, device="cuda", dtype=torch.int32
    )
    request_row_counts[-1] = num_rows - rows_per_request * (num_requests - 1)

    # Mix context lengths: every third request is short-context, attending
    # to only its first few pages; the rest fill their cache.
    context_lengths = torch.full(
        (num_requests,),
        num_pages_per_request * page_size - 1,
        device="cuda",
        dtype=torch.int32,
    )
    short_requests = torch.arange(num_requests, device="cuda") % 3 == 1
    context_lengths[short_requests] = request_row_counts[short_requests] + 8
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    # Eight blocks per page; adjacent rows overlap by six of those eight.
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_lengths[token_to_req.long()]
        - request_row_counts[token_to_req.long()]
        + rows_within_request
    ).to(torch.int64)
    sequence_lengths = context_lengths
    visible_blocks = torch.minimum(
        (query_positions + 1) // indexer_compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // indexer_compress_ratio,
    ).to(torch.int32)
    # +1: the packed trailing column holds each row's valid-entry count
    # (never a token index); the reference reads only the selection region.
    logical_indices = torch.empty(
        (num_rows, selection_width + 1), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        indexer_compress_ratio,
        indexer_budget,
        logical_indices,
    )

    scale = head_dim**-0.5

    if fp8:
        # A fixed non-unit pair (k != v) exercises the host-side scale folding
        # and catches a k/v swap; scales are host floats, as the layer exposes
        # them. Stored values are the scaled ones, as reshape_and_cache does.
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        # Below compute capability 8.9 the kernel reads e4m3 pages as raw
        # bytes; the torch reference below keeps the typed tensor either way.
        _fp8_pages_as_dispatched(k_cache) if fp8 else k_cache,
        _fp8_pages_as_dispatched(v_cache) if fp8 else v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=use_prefill_config,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices[:, :selection_width],
        block_table,
        token_to_req,
        scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = expected * torch.sigmoid(output_gate)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
@pytest.mark.parametrize("decode_query_len", [1, 2, 3, 4])
def test_qsa_split_selection_correctness(workspace_init, decode_query_len: int) -> None:
    query_lens = [decode_query_len, decode_query_len, 33]
    rows, heads, head_dim = sum(query_lens), 4, 128
    token_topk, compress_ratio = 2048, 4
    torch.manual_seed(13)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(120, 16, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.arange(120, device="cuda", dtype=torch.int32).view(3, 40)
    token_to_req = torch.repeat_interleave(
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.tensor(query_lens, device="cuda"),
    )
    query_start_loc = torch.tensor(
        [0, decode_query_len, 2 * decode_query_len, rows],
        device="cuda",
        dtype=torch.int32,
    )
    sequence_lengths = torch.full((3,), 2560, device="cuda", dtype=torch.int32)
    query_positions = torch.cat(
        [
            torch.arange(2560 - query_len, 2560, device="cuda")
            for query_len in query_lens
        ]
    )

    block_indices = torch.empty(
        rows,
        token_topk // compress_ratio,
        device="cuda",
        dtype=torch.int32,
    )
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    ).to(torch.int32)
    num_decode_tokens = 2 * decode_query_len
    decode_slice = slice(0, num_decode_tokens)
    qsa_indexer_ops.qsa_select_paged_decode(
        q[decode_slice],
        cache,
        page_table[:2],
        visible_blocks[decode_slice],
        token_topk,
        compress_ratio,
        decode_query_len,
        block_indices[decode_slice],
    )
    prefill_slice = slice(num_decode_tokens, rows)
    qsa_indexer_ops.qsa_select_paged_prefill(
        q[prefill_slice],
        cache,
        page_table[2:],
        query_start_loc[2:],
        visible_blocks[prefill_slice],
        token_topk,
        compress_ratio,
        query_lens[-1],
        block_indices[prefill_slice],
        max_seq_len=sequence_lengths.max().item(),
    )
    # +1: the packed trailing count column (never a token index; excluded
    # from the comparison).
    actual = torch.empty(
        (rows, token_topk + compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        compress_ratio,
        token_topk,
        actual,
    )
    expected_blocks = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )
    expected = _expand_qsa_indices_reference(
        expected_blocks,
        query_positions,
        sequence_lengths.index_select(0, token_to_req.long()),
        compress_ratio,
        token_topk,
    )

    torch.testing.assert_close(
        actual[:, : token_topk + compress_ratio - 1].sort().values,
        expected.sort().values,
    )


@requires_qsa_kernels
def test_qsa_selection_handles_no_complete_compressed_blocks(workspace_init) -> None:
    q = torch.zeros(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 8, device="cuda", dtype=torch.bfloat16)
    page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    visible_blocks = torch.zeros(2, device="cuda", dtype=torch.int32)

    block_indices = torch.empty((2, 512), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.qsa_select_paged_prefill(
        q,
        cache,
        page_table,
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        visible_blocks,
        token_topk=2048,
        compress_ratio=4,
        max_query_len=2,
        block_indices=block_indices,
        max_seq_len=64,  # clamps to the page-table capacity
    )
    selected = torch.empty((2, 2052), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        compress_ratio=4,
        token_topk=2048,
        out=selected,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.all(selected[0, 2:2051] == -1)
    assert torch.all(selected[1, 3:2051] == -1)
    # The packed trailing column holds each row's valid-entry count.
    assert selected[:, 2051].tolist() == [2, 3]


@requires_qsa_kernels
def test_qsa_streaming_compression_and_compressor_state_store_match_reference() -> None:
    head_dim = 8
    current_pairs = [
        *((0, position) for position in range(2, 9)),
        *((1, position) for position in range(5, 11)),
    ]

    def key_row(request: int, position: int) -> torch.Tensor:
        return (
            torch.arange(head_dim, dtype=torch.float32) + request * 1000 + position * 10
        )

    def position_row(request: int, position: int) -> torch.Tensor:
        return torch.tensor(
            [
                request * 1000 + position,
                request * 1000 + position + 100,
                request * 1000 + position + 200,
            ],
            dtype=torch.int64,
        )

    raw_keys = (
        torch.stack([key_row(request, position) for request, position in current_pairs])
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    raw_positions = (
        torch.stack(
            [position_row(request, position) for request, position in current_pairs]
        )
        .unsqueeze(1)
        .to(device="cuda")
    )
    token_to_req = torch.tensor(
        [request for request, _ in current_pairs],
        dtype=torch.int32,
        device="cuda",
    )
    logical_positions = torch.tensor(
        [position for _, position in current_pairs],
        dtype=torch.int64,
        device="cuda",
    )
    query_start_loc = torch.tensor([0, 7, 13], dtype=torch.int32, device="cuda")
    compressor_state_block_table = torch.tensor(
        [[1], [0]], dtype=torch.int32, device="cuda"
    )
    compressor_state_cache = torch.zeros(
        2, 4, 1, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    rope_cache = torch.zeros(2, 4, 1, 3, dtype=torch.int64, device="cuda")
    for request, position, block in ((0, 0, 1), (0, 1, 1), (1, 4, 0)):
        compressor_state_cache[block, position % 4, 0] = key_row(request, position).to(
            device="cuda", dtype=torch.bfloat16
        )
        rope_cache[block, position % 4, 0] = position_row(request, position).to("cuda")

    compressed_slots = torch.full(
        (len(current_pairs),), -1, dtype=torch.int64, device="cuda"
    )
    valid_rows = torch.tensor([1, 5, 9], dtype=torch.int64, device="cuda")
    compressed_slots[valid_rows] = torch.arange(3, device="cuda")
    pooled, first_positions = qsa_ops.qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
        rope_cache=rope_cache,
    )
    pooled_without_rope, scalar_first_positions = (
        qsa_ops.qsa_compress_groups_with_ratio(
            raw_keys,
            raw_positions,
            compressor_state_cache,
            compressor_state_block_table,
            token_to_req,
            query_start_loc,
            logical_positions,
            compressed_slots,
            compress_ratio=4,
        )
    )

    groups = [
        [(0, position) for position in range(0, 4)],
        [(0, position) for position in range(4, 8)],
        [(1, position) for position in range(4, 8)],
    ]
    expected_pooled = (
        torch.stack(
            [
                torch.stack([key_row(*pair) for pair in group]).mean(dim=0)
                for group in groups
            ]
        )
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    expected_positions = torch.stack(
        [position_row(0, 0), position_row(0, 4), position_row(1, 4)]
    ).to("cuda")
    expected_scalar_positions = torch.tensor(
        [[0, 0, 0], [4, 4, 4], [4, 4, 4]],
        dtype=torch.int64,
        device="cuda",
    )

    torch.testing.assert_close(pooled[valid_rows], expected_pooled)
    torch.testing.assert_close(pooled_without_rope[valid_rows], expected_pooled)
    torch.testing.assert_close(first_positions[valid_rows], expected_positions)
    torch.testing.assert_close(
        scalar_first_positions[valid_rows], expected_scalar_positions
    )

    compressor_state_slots = torch.tensor(
        [-1, -1, -1, 5, 6, 7, 4, -1, -1, 3, 0, 1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    qsa_ops.qsa_store_cache_rows(
        compressor_state_cache, compressor_state_slots, raw_keys
    )
    qsa_ops.qsa_store_cache_rows(rope_cache, compressor_state_slots, raw_positions)
    for request, positions, block in ((0, range(5, 9), 1), (1, range(7, 11), 0)):
        for position in positions:
            torch.testing.assert_close(
                compressor_state_cache[block, position % 4, 0],
                key_row(request, position).to(device="cuda", dtype=torch.bfloat16),
            )
            torch.testing.assert_close(
                rope_cache[block, position % 4, 0],
                position_row(request, position).to("cuda"),
            )


def _fp8_roundtrip(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize to per-tensor E4M3 and return the bytes plus a host scale.

    The reader takes its dequant scales as Python floats, the way the layer
    hands over `_k_scale_float`, so the helper returns one rather than a
    device tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (tensor.abs().amax().float() / finfo.max).clamp_min(1e-12)
    quantized = (tensor.float() / scale).clamp(finfo.min, finfo.max)
    return quantized.to(torch.float8_e4m3fn), float(scale)


def _fp8_pages_as_dispatched(quantized: torch.Tensor) -> torch.Tensor:
    """The view the forward builds for an e4m3 page on this device.

    Below compute capability 8.9 Triton cannot type an fp8 pointer, so the
    pages go in as raw bytes; from 8.9 on they go in as e4m3.
    """
    if current_platform.supports_fp8():
        return quantized
    return quantized.view(torch.uint8)


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_fp8_matches_dequantized_reference() -> None:
    """The FP8 reader must agree with attention over the dequantized pages."""
    torch.manual_seed(11)
    num_rows, num_query_heads, num_kv_heads, page_size, head_dim = 3, 8, 1, 16, 256
    num_pages = 6
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k_ref = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_ref = torch.randn_like(k_ref)
    k_bytes, k_scale = _fp8_roundtrip(k_ref)
    v_bytes, v_scale = _fp8_roundtrip(v_ref)

    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).reshape(
        1, -1
    )
    token_to_req = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
    width = 8
    logical_indices = _packed_selection(width, num_rows)
    # One gate for both calls: the gate multiplies each side identically, so
    # what this compares is still the FP8 reader against the dequantized pages.
    output_gate = torch.randn_like(q)

    quantized = qsa_ops.qsa_sparse_paged_attention(
        q,
        _fp8_pages_as_dispatched(k_bytes),
        _fp8_pages_as_dispatched(v_bytes),
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    reference = qsa_ops.qsa_sparse_paged_attention(
        q,
        (k_bytes.to(torch.float32) * k_scale).to(torch.bfloat16),
        (v_bytes.to(torch.float32) * v_scale).to(torch.bfloat16),
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        output_gate=output_gate,
    )
    torch.testing.assert_close(
        quantized.float(), reference.float(), rtol=2e-2, atol=2e-2
    )


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_nvfp4_matches_dequantized_reference() -> None:
    """The NVFP4 slot reader must agree with its own dequantized pages."""
    from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

    torch.manual_seed(12)
    # num_kv_heads != page_size on purpose: with them equal a grid axis or a
    # token/head stride taken from the wrong dimension still lines up.
    num_rows, num_query_heads, num_kv_heads, page_size, head_dim = 2, 8, 2, 16, 256
    num_pages = 4
    width_bytes = nvfp4_kv_cache_full_dim(head_dim)
    data_bytes = head_dim // 2

    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )

    # Random packed pages: the reader is checked against a torch decode of the
    # very same bytes, so the values only have to be well-formed.
    def make_slots() -> torch.Tensor:
        slots = torch.randint(
            0,
            256,
            (num_pages, num_kv_heads, page_size, width_bytes),
            device="cuda",
            dtype=torch.uint8,
        )
        # Random bytes in the scale tail would decode to E4M3 NaN, so write
        # finite scales there instead.
        scales = (
            torch.rand(
                num_pages,
                num_kv_heads,
                page_size,
                width_bytes - data_bytes,
                device="cuda",
            )
            + 0.5
        )
        slots[..., data_bytes:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
        return slots

    k_slots = make_slots()
    v_slots = make_slots()
    # Distinct non-unit global scales: the writer stores each block scale
    # relative to one, so a reader that drops it, applies it twice, or swaps K
    # for V lands somewhere else. Equal or unit values would hide all three.
    global_k_scale = 0.375
    global_v_scale = 1.75

    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).reshape(
        1, -1
    )
    token_to_req = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
    logical_indices = _packed_selection(8, num_rows)
    # One gate for both calls, for the same reason as the FP8 test above.
    output_gate = torch.randn_like(q)

    packed = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_slots,
        v_slots,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=global_k_scale,
        v_scale=global_v_scale,
        nvfp4=True,
        output_gate=output_gate,
    )

    def decode(slots: torch.Tensor, global_scale: float) -> torch.Tensor:
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
        values = sign * magnitude
        block_scale = scales.repeat_interleave(16, dim=-1)
        # Slot views are [pages, heads, tokens, width]; pages are [pages,
        # tokens, heads, dim] for the BF16 path.
        return (
            (values * block_scale * global_scale).to(torch.bfloat16).permute(0, 2, 1, 3)
        )

    reference = qsa_ops.qsa_sparse_paged_attention(
        q,
        decode(k_slots, global_k_scale),
        decode(v_slots, global_v_scale),
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
        output_gate=output_gate,
    )
    torch.testing.assert_close(packed.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
def test_qsa_packed_decoders_match_their_arithmetic_definitions() -> None:
    """Every encoding the branch-free decoders accept must land on its value."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def probe(source_ptr, fp4_ptr, scale_ptr, WIDTH: tl.constexpr):
        offsets = tl.arange(0, WIDTH)
        raw = tl.load(source_ptr + offsets)
        tl.store(fp4_ptr + offsets, qsa_ops._dequant_fp4_e2m1(raw.to(tl.int32) & 15))
        tl.store(scale_ptr + offsets, qsa_ops._dequant_e4m3fn_block_scales(raw))

    width = 256
    source = torch.arange(width, device="cuda", dtype=torch.uint8)
    decoded_fp4 = torch.empty(width, device="cuda", dtype=torch.bfloat16)
    decoded_scales = torch.empty(width, device="cuda", dtype=torch.bfloat16)
    probe[(1,)](source, decoded_fp4, decoded_scales, WIDTH=width)

    def assert_same_bits(decoded: torch.Tensor, expected: torch.Tensor) -> None:
        # Compare BF16 encodings, not values: these decoders assemble the bit
        # pattern by hand, and a float comparison would call -0.0 equal to +0.0
        # and miss a dropped sign on the zero encodings (FP4 0x8, E4M3 0x80).
        assert torch.equal(
            decoded.cpu().view(torch.uint16),
            expected.to(torch.bfloat16).view(torch.uint16),
        )

    # Both formats keep at most four significant bits, so BF16 holds every
    # value exactly and the comparison can demand equality rather than a
    # tolerance. The right-hand sides restate the FP4 E2M1 and FP8 E4M3
    # definitions the arithmetic decoders implement.
    nibbles = (source.to(torch.int32) & 15).cpu()
    exponent = (nibbles >> 1) & 3
    mantissa = (nibbles & 1).double()
    expected_fp4 = torch.where(nibbles & 8 != 0, -1.0, 1.0).double() * torch.where(
        exponent > 0,
        torch.exp2(exponent.double() - 2.0) * (2.0 + mantissa),
        mantissa * 0.5,
    )
    assert_same_bits(decoded_fp4, expected_fp4)

    byte = source.to(torch.int32).cpu()
    exponent = (byte >> 3) & 15
    mantissa = (byte & 7).double()
    expected_scales = torch.where(byte & 128 != 0, -1.0, 1.0).double() * torch.where(
        exponent > 0,
        torch.exp2(exponent.double() - 10.0) * (8.0 + mantissa),
        mantissa * 0.001953125,
    )
    assert_same_bits(decoded_scales, expected_scales)


@pytest.fixture(autouse=True)
def _clear_capability_caches():
    """Drop the capability caches around every test in this module.

    `_stub_capability` fills them through a stubbed platform; left in place
    they would answer for the real device in whatever test ran next.
    """
    qsa_ops._is_sm120.cache_clear()
    qsa_ops._fp8_pages_are_raw_bytes.cache_clear()
    yield
    qsa_ops._is_sm120.cache_clear()
    qsa_ops._fp8_pages_are_raw_bytes.cache_clear()


def _stub_capability(monkeypatch, supported: bool, sm120: bool = False) -> None:
    """Pin the two capability probes the profile reads.

    `_select_config` dispatches to the sm_120 table through `_is_sm120()`, so
    a stub that answers only `has_device_capability` leaves that call reaching
    the real platform. Both are pinned, and the `lru_cache` in front of the
    sm_120 probe is cleared so the stub is the one consulted.
    """
    # One real device per combination, so the three probes can never disagree:
    # `supported` selects whether the tuned-table gate is met, which is SM100,
    # and every answer below is derived from that one capability.
    capability = (12, 0) if sm120 else ((10, 0) if supported else (8, 0))
    packed = capability[0] * 10 + capability[1]
    monkeypatch.setattr(
        qsa_ops,
        "current_platform",
        SimpleNamespace(
            has_device_capability=lambda wanted: packed >= wanted,
            get_device_capability=lambda: capability,
            supports_fp8=lambda: capability >= (8, 9),
        ),
    )
    qsa_ops._is_sm120.cache_clear()
    qsa_ops._fp8_pages_are_raw_bytes.cache_clear()


@pytest.mark.parametrize("kv_quant", [1, 2])
@pytest.mark.parametrize("use_prefill_config", [False, True])
def test_qsa_splitk_profile_only_narrows_tiles_that_cannot_stage(
    kv_quant, use_prefill_config, monkeypatch
) -> None:
    """Packed caches keep the tuned table until its KV tile stops fitting."""
    _stub_capability(monkeypatch, supported=False)
    head_dim, selection_width = 256, 2051
    # Only the paths that rebuild values from bytes stage: raw e4m3 and NVFP4.
    raw_fp8 = kv_quant == 1
    budget = qsa_ops._qsa_staged_block_n(head_dim, kv_quant, raw_fp8)
    assert budget == (16 if kv_quant == 2 else 32)
    # A native fp8 pointer stages nothing, so it keeps the tuned tile.
    assert qsa_ops._qsa_staged_block_n(head_dim, 1, False) is None

    def profile(num_rows, quant):
        return qsa_ops._qsa_splitk_profile(
            num_rows,
            1,
            use_prefill_config,
            selection_width,
            head_dim,
            quant,
            quant == 1,
        )

    # Keyed off the table rather than a fixed row list, so a retuned
    # _select_config moves this test with it instead of pinning stale widths.
    for num_rows in (1, 4, 16, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        tuned = profile(num_rows, 0)
        narrowed = profile(num_rows, kv_quant)
        if tuned[0] <= budget:
            # The table already picks a tile that stages, so nothing may move.
            assert narrowed == tuned
            continue
        block_n, _, tiles, splits, _ = narrowed
        assert block_n == budget
        assert tiles == -(-selection_width // block_n)
        # The narrow tile must not buy its parallelism with a bigger FP32
        # partial workspace than the tuned profile already allocated.
        assert 1 <= splits <= tuned[3]
        assert splits <= max(1, tiles // qsa_ops._QSA_MIN_TILES_PER_SPLIT)


def test_qsa_splitk_profile_keeps_the_tuned_table_where_it_was_measured(
    monkeypatch,
) -> None:
    """SM100 and unquantized caches must see the profile they were tuned with."""
    _stub_capability(monkeypatch, supported=True)
    for kv_quant, raw_fp8 in ((0, False), (1, True), (1, False), (2, False)):
        tuned = qsa_ops._select_config(2048, 1, False, 2051, is_fp8=kv_quant == 1)
        assert qsa_ops._qsa_staged_block_n(256, kv_quant, raw_fp8) is None
        assert qsa_ops._qsa_splitk_profile(
            2048, 1, False, 2051, 256, kv_quant, raw_fp8
        ) == (*tuned, 2)

    _stub_capability(monkeypatch, supported=False)
    # An unquantized cache reads BF16 pages straight into the dots, and a
    # native fp8 pointer stages nothing either, so neither takes the staging
    # profile no matter how old the device is.
    bf16_tuned = qsa_ops._select_config(2048, 1, False, 2051, is_fp8=False)
    assert qsa_ops._qsa_staged_block_n(256, 0, False) is None
    assert qsa_ops._qsa_splitk_profile(2048, 1, False, 2051, 256, 0, False) == (
        *bf16_tuned,
        2,
    )
    fp8_tuned = qsa_ops._select_config(2048, 1, False, 2051, is_fp8=True)
    assert qsa_ops._qsa_staged_block_n(256, 1, False) is None
    assert qsa_ops._qsa_splitk_profile(2048, 1, False, 2051, 256, 1, False) == (
        *fp8_tuned,
        2,
    )


def test_qsa_backend_advertises_the_kv_cache_dtypes_it_serves() -> None:
    """The declared contract must match the caches the impl actually reads."""
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend

    backend = Qwen4ExpQSAFlashAttentionBackend
    for dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4"):
        assert backend.supports_kv_cache_dtype(dtype), dtype
    assert not backend.supports_kv_cache_dtype("float16")


def test_qsa_unquantized_call_allocates_no_device_scale(monkeypatch) -> None:
    """A BF16 call must not build a device scale tensor at all.

    The kernel takes its dequant scales as host floats folded into
    `softmax_scale` and `output_scale`, so an unquantized call has nothing to
    allocate. The earlier contract cached one unit tensor per device; there is
    no longer anything to cache, and this holds the allocation at zero rather
    than at one.
    """
    if not current_platform.is_cuda():
        pytest.skip("CUDA is required")
    q = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    # Built before the torch.ones counter below is installed, and with randn
    # rather than ones, so it cannot disturb what that counter measures.
    output_gate = torch.randn_like(q)
    device = q.device
    assert not hasattr(qsa_ops, "_QSA_UNIT_SCALE_BY_DEVICE"), (
        "the device unit-scale cache is gone; host floats replaced it"
    )

    builds = []
    real_ones = torch.ones

    def counting_ones(*args, **kwargs):
        if kwargs.get("device") == device or device in args:
            builds.append(1)
        return real_ones(*args, **kwargs)

    seen: list[tuple[float, float]] = []
    real_kernel = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel

    class _Recording:
        def __getitem__(self, grid):
            inner = real_kernel[grid]

            def run(*args, **kwargs):
                # softmax_scale and output_scale are positional, right after
                # the output pointer.
                seen.append((args[9], args[10]))
                return inner(*args, **kwargs)

            return run

    monkeypatch.setattr(torch, "ones", counting_ones)
    monkeypatch.setattr(qsa_ops, "_qsa_sparse_paged_gqa_splitk_kernel", _Recording())
    k = torch.randn(2, 16, 1, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(2, device=device, dtype=torch.int32).reshape(1, 2)
    token_to_req = torch.zeros(1, device=device, dtype=torch.int32)
    indices = _packed_selection(8, 1, device=device)
    for _ in range(3):
        qsa_ops.qsa_sparse_paged_attention(
            q,
            k,
            v,
            indices,
            block_table,
            token_to_req,
            use_prefill_config=False,
            output_gate=output_gate,
        )
    assert not builds, (
        f"a BF16 call built {len(builds)} device tensor(s) with torch.ones"
    )
    assert seen == [(256**-0.5, 1.0)] * 3, seen


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("nvfp4", "kv_quantized", "expected_quant"),
    [(False, False, 0), (False, True, 1), (True, False, 2)],
)
def test_qsa_warmup_compiles_the_layout_it_is_given(
    nvfp4, kv_quantized, expected_quant, monkeypatch
) -> None:
    """Warmup must bind the same kernel signature the forward path calls.

    It runs only at startup, so a signature that drifted from the kernel --
    a missing scale pointer, a missing KV_QUANT -- surfaces as a TypeError
    during engine init rather than in any forward test.
    """
    from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

    head_size, num_kv_heads, pages, page_size = 256, 2, 4, 8
    if nvfp4:
        width = nvfp4_kv_cache_full_dim(head_size)
        kv_cache = torch.zeros(
            pages, 2 * num_kv_heads, page_size, width, dtype=torch.uint8, device="cuda"
        )
    else:
        # The unquantized allocation is [pages, kv_heads, page_size, 2 * head]:
        # the transpose in the warmup is what turns it into paged NHD.
        dtype = torch.uint8 if kv_quantized else torch.bfloat16
        kv_cache = torch.zeros(
            pages, num_kv_heads, page_size, 2 * head_size, dtype=dtype, device="cuda"
        )
    block_table = torch.zeros(2, pages, dtype=torch.int32, device="cuda")

    seen: list[tuple[tuple, dict]] = []
    real_warmup = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel.warmup

    def capture(*args, **kwargs):
        seen.append((args, kwargs))
        return real_warmup(*args, **kwargs)

    monkeypatch.setattr(qsa_ops._qsa_sparse_paged_gqa_splitk_kernel, "warmup", capture)

    qsa_ops.warmup_qsa_sparse_paged_attention(
        kv_cache,
        block_table,
        num_query_heads=num_kv_heads * 2,
        selection_width=64,
        head_size=head_size,
        nvfp4=nvfp4,
        kv_quantized=kv_quantized,
    )

    assert seen, "warmup compiled no specialization"

    # The views and the six (block, token, head) strides the forward builds
    # for this layout. A warmup that derived them differently would compile a
    # kernel the forward never calls.
    if nvfp4:
        key_view, value_view = kv_cache[:, 0::2], kv_cache[:, 1::2]
        expected_strides = (
            key_view.stride(0),
            key_view.stride(2),
            key_view.stride(1),
            value_view.stride(0),
            value_view.stride(2),
            value_view.stride(1),
        )
    else:
        key_view, value_view = kv_cache.transpose(1, 2).split(head_size, dim=-1)
        expected_strides = (
            key_view.stride(0),
            key_view.stride(1),
            key_view.stride(2),
            value_view.stride(0),
            value_view.stride(1),
            value_view.stride(2),
        )
    expected_raw_fp8 = expected_quant == 1 and not current_platform.supports_fp8()
    if expected_quant == 0:
        expected_ptr_dtype = torch.bfloat16
    elif expected_quant == 2 or expected_raw_fp8:
        expected_ptr_dtype = torch.uint8
    else:
        expected_ptr_dtype = torch.float8_e4m3fn
    # Every profile the forward can reach for this layout, so a warmup that
    # compiled some other tile or stage count shows up here.
    expected_profiles = {
        qsa_ops._qsa_splitk_profile(
            num_rows,
            num_kv_heads,
            use_prefill_config,
            64,
            head_size,
            expected_quant,
            expected_raw_fp8,
        )
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    for args, kwargs in seen:
        assert kwargs["KV_QUANT"] == expected_quant
        assert kwargs["RAW_FP8"] == expected_raw_fp8
        assert kwargs["PAGE_SIZE"] == page_size
        assert kwargs["HEAD_DIM"] == head_size
        # q_ptr, k, v, indices, block_table, token_to_req, partial_out,
        # partial_lse, out, softmax_scale, output_scale, gate, q row/head
        # stride, then the six cache strides.
        assert tuple(args[14:20]) == expected_strides
        assert args[1].dtype == args[2].dtype == expected_ptr_dtype
        assert kwargs["grid"] == (16, num_kv_heads, kwargs["NUM_SPLITS"])
        assert (
            kwargs["BLOCK_N"],
            kwargs["num_warps"],
            kwargs["NUM_TILES"],
            kwargs["NUM_SPLITS"],
            kwargs["num_stages"],
        ) in expected_profiles


class _ReachedQKVParallelLinear(Exception):
    """Raised in place of the first heavy layer the QSA owner builds.

    The dtype gates run before it, so reaching this is what says a cache dtype
    was admitted -- without the test having to repeat the list of dtypes the
    constructor accepts.
    """


def _qsa_owner_config() -> SimpleNamespace:
    """The smallest text config the QSA owner's gates read."""
    return SimpleNamespace(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        is_causal=True,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        indexer_n_heads=4,
        partial_rotary_factor=1.0,
    )


def _qsa_owner_vllm_config(cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=16),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        num_speculative_tokens=0,
    )


@pytest.mark.parametrize(
    "cache_dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4"]
)
def test_qsa_owner_admits_every_kv_cache_dtype_the_backend_serves(
    cache_dtype, monkeypatch
) -> None:
    """The layer owner's own gate must admit the dtypes the backend declares.

    The backend advertises nvfp4 and the impl accepts it, but the owner keeps a
    second gate on `cache_config.cache_dtype`. It was dropped in an upstream
    merge and serving stopped at engine init with NotImplementedError while
    every kernel test still passed, because no test ran this constructor.
    """
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod

    def reached(*args, **kwargs):
        raise _ReachedQKVParallelLinear

    monkeypatch.setattr(qsa_mod, "QKVParallelLinear", reached)
    monkeypatch.setattr(qsa_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    with pytest.raises(_ReachedQKVParallelLinear):
        qsa_mod.Qwen4ExpQSAAttention(
            vllm_config=_qsa_owner_vllm_config(cache_dtype),
            config=_qsa_owner_config(),
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )


def test_qsa_owner_rejects_a_kv_cache_dtype_it_cannot_read(monkeypatch) -> None:
    """And an unsupported dtype must stop at that gate, not somewhere later."""
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod

    def reached(*args, **kwargs):
        raise _ReachedQKVParallelLinear

    monkeypatch.setattr(qsa_mod, "QKVParallelLinear", reached)
    monkeypatch.setattr(qsa_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    with pytest.raises(NotImplementedError, match="main KV cache"):
        qsa_mod.Qwen4ExpQSAAttention(
            vllm_config=_qsa_owner_vllm_config("fp8_e5m2"),
            config=_qsa_owner_config(),
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )


def _qsa_owner_for_spec(kv_cache_dtype: str, torch_dtype: torch.dtype):
    """A QSA owner carrying only what `get_kv_cache_spec` reads.

    The constructor builds projections and an indexer, none of which the spec
    depends on, so the four attributes it does read are set directly.
    """
    from torch import nn

    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    owner = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(owner)
    owner.num_kv_heads = 2
    owner.head_dim = 128
    owner.kv_cache_dtype = kv_cache_dtype
    owner.kv_cache_torch_dtype = torch_dtype
    return owner


def test_qsa_nvfp4_kv_cache_spec_sizes_its_own_slots() -> None:
    """NVFP4 keeps K and V in separate per-head slots of packed bytes.

    The spec carries that as `num_head_slots` and `state_content_bytes`; a spec
    that dropped them would size the allocation as a dense BF16 page and the
    cache would be wrong without anything raising.
    """
    from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    owner = _qsa_owner_for_spec("nvfp4", torch.uint8)
    spec = owner.get_kv_cache_spec(vllm_config)

    assert spec.num_head_slots == 2 * owner.num_kv_heads
    assert spec.state_content_bytes == nvfp4_kv_cache_full_dim(owner.head_dim)
    # The overrides have to reach the sizes the allocator actually uses.
    assert spec.num_heads == 2 * owner.num_kv_heads
    assert spec.state_content_size_bytes == nvfp4_kv_cache_full_dim(owner.head_dim)
    assert spec.page_size_bytes == (
        spec.block_size * spec.num_heads * spec.state_content_size_bytes
    )


@pytest.mark.parametrize(
    ("kv_cache_dtype", "torch_dtype"),
    [("auto", torch.bfloat16), ("fp8_e4m3", torch.uint8)],
)
def test_qsa_dense_kv_cache_spec_keeps_its_sizing(kv_cache_dtype, torch_dtype) -> None:
    """A BF16 or e4m3 cache must be sized exactly as it was before NVFP4."""
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    owner = _qsa_owner_for_spec(kv_cache_dtype, torch_dtype)
    spec = owner.get_kv_cache_spec(vllm_config)

    assert spec.num_head_slots is None
    assert spec.state_content_bytes is None
    # Dense pages hold K and V for every head at the element width.
    assert spec.num_heads == owner.num_kv_heads
    assert spec.page_size_bytes == (
        2 * spec.block_size * owner.num_kv_heads * owner.head_dim * torch_dtype.itemsize
    )
