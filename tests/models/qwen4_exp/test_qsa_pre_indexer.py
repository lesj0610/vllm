# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused QSA pre-indexer."""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer as pre_indexer_module
from vllm.models.qwen4_exp.common.qsa_cache import (
    canonical_qsa_rope_positions,
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)
from vllm.models.qwen4_exp.nvidia.indexer_qsa import apply_qsa_rope
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_store_cache_rows,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import (
    qsa_pre_indexer,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

HQ, D = 4, 128
CR = 4
MROPE_SECTION = (11, 11, 10)
EPS = 1e-6
BLOCK_SIZE = 16
COMP_PAGE = BLOCK_SIZE // CR
ROPE_POS_OFFSET = D
RTOL = 1.6e-2
ATOL = 1e-2
MIXED_BATCH = ([260, 259, 138], [1, 1, 37], [8, 8, 8])


def _make_block_table(block_counts):
    num_blocks = sum(block_counts)
    table = torch.full((len(block_counts), max(block_counts)), -1, dtype=torch.int32)
    physical_blocks = torch.randperm(num_blocks)
    offset = 0
    for request, count in enumerate(block_counts):
        table[request, :count] = physical_blocks[offset : offset + count]
        offset += count
    return table, num_blocks


def assert_fp8_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # e4m3 is sign-magnitude, so within a sign the uint8 code order matches the
    # value order and one ulp is one code step. The two paths' intermediates
    # differ in the pooling accumulation order, which at denormal magnitudes
    # (absolute grid step 2^-9) shows up as up to 2 code steps.
    code_diff = (
        actual.view(torch.uint8).int() - expected.view(torch.uint8).int()
    ).abs()
    abs_diff = (actual.float() - expected.float()).abs()
    assert bool(((code_diff <= 1) | (abs_diff <= 2**-8)).all())


@requires_qsa_kernels
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("indexer_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "mrope,is_2d_positions,cache_rope_positions,state_size,seq_lens,query_lens,history_lens",
    [
        pytest.param(True, True, True, 4, *MIXED_BATCH, id="mrope"),
        pytest.param(False, False, False, 4, *MIXED_BATCH, id="text"),
        pytest.param(True, False, True, 8, *MIXED_BATCH, id="mrope-model-1d"),
        pytest.param(True, True, False, 4, *MIXED_BATCH, id="mrope-no-position-cache"),
        pytest.param(
            True,
            False,
            False,
            8,
            *MIXED_BATCH,
            id="mrope-model-1d-no-position-cache",
        ),
        pytest.param(
            False, False, True, 4, *MIXED_BATCH, id="text-with-position-cache"
        ),
        pytest.param(True, True, True, 4, [37], [37], [0], id="fresh"),
        pytest.param(True, True, True, 4, [4097], [4097], [0], id="tiled"),
    ],
)
def test_qsa_fused_pre_indexer_matches_unfused(
    indexer_dtype,
    mrope,
    is_2d_positions,
    cache_rope_positions,
    state_size,
    seq_lens,
    query_lens,
    history_lens,
) -> None:
    from flashinfer.norm import gemma_rmsnorm

    from vllm.model_executor.layers.rotary_embedding import get_rope

    device = "cuda"
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

    token_to_req = torch.cat(
        [
            torch.full((length,), request, dtype=torch.int32)
            for request, length in enumerate(query_lens)
        ]
    ).to(device)
    logical_positions = torch.cat(
        [
            torch.arange(seq_len - query_len, seq_len, dtype=torch.int64)
            for seq_len, query_len in zip(seq_lens, query_lens)
        ]
    ).to(device)
    num_tokens = logical_positions.numel()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    ).to(device)
    positions = (
        torch.stack(
            [
                logical_positions,
                logical_positions // 7 + 3,
                logical_positions // 13 + 11,
            ]
        )
        if is_2d_positions
        else logical_positions
    )
    position_rows = (
        canonical_qsa_rope_positions(positions)
        if cache_rope_positions
        else logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
    )

    compressed_block_counts = [
        (seq_len // CR + COMP_PAGE - 1) // COMP_PAGE for seq_len in seq_lens
    ]
    raw_block_table, num_raw_blocks = _make_block_table([1] * len(seq_lens))
    compressed_block_table, num_compressed_blocks = _make_block_table(
        compressed_block_counts
    )
    raw_block_table = raw_block_table.to(device)
    raw_slots = circular_qsa_slot_mapping(
        raw_block_table,
        token_to_req,
        logical_positions,
        state_size,
        query_start_loc,
    )
    compressed_slots = compressed_qsa_slot_mapping(
        compressed_block_table.to(device),
        token_to_req,
        logical_positions,
        COMP_PAGE,
        CR,
    )
    group_counts = torch.tensor(
        [
            seq_len // CR - (seq_len - query_len) // CR
            for seq_len, query_len in zip(seq_lens, query_lens)
        ],
        dtype=torch.int32,
        device=device,
    )
    k_work_counts = torch.maximum(group_counts, torch.ones_like(group_counts))
    k_start_loc = torch.cat([k_work_counts.new_zeros(1), k_work_counts.cumsum(0)])
    work_requests = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32, device=device),
        k_work_counts,
    )
    local_work = torch.arange(
        int(k_start_loc[-1]), dtype=torch.int32, device=device
    ) - torch.repeat_interleave(k_start_loc[:-1], k_work_counts)
    max_k_work = (num_tokens + (CR - 1) * len(query_lens)) // CR
    k_work_metadata = torch.full((max_k_work, 2), -1, dtype=torch.int32, device=device)
    k_work_metadata[: work_requests.numel()] = torch.stack(
        (work_requests, local_work), dim=1
    )

    raw_width = D + 12 if cache_rope_positions else D
    # Match vLLM's padded-page cache layout: rows are contiguous, while physical
    # blocks have a larger stride than their logical contents.
    raw_page_elements = state_size * raw_width
    fused_raw_storage = torch.zeros(
        num_raw_blocks,
        raw_page_elements + 16,
        dtype=torch.bfloat16,
        device=device,
    )
    fused_raw = torch.as_strided(
        fused_raw_storage,
        (num_raw_blocks, state_size, 1, raw_width),
        (raw_page_elements + 16, raw_width, raw_width, 1),
    )
    for request, history_len in enumerate(history_lens):
        history_end = seq_lens[request] - query_lens[request]
        for position in range(history_end - history_len, history_end):
            block = int(raw_block_table[request, 0])
            row = fused_raw[block, position % state_size, 0]
            row[:D] = torch.randn(D, dtype=torch.bfloat16, device=device)
            if cache_rope_positions:
                row[ROPE_POS_OFFSET:].view(torch.int64).copy_(
                    torch.tensor(
                        [position, position // 7 + 3, position // 13 + 11],
                        dtype=torch.int64,
                        device=device,
                    )
                )
    unfused_raw = fused_raw.clone()
    # A second untouched copy: the fp8 branch reruns the fused call at the compute
    # dtype, and the kernel writes the ring in place.
    pre_call_raw = fused_raw.clone()
    compressed_page_elements = COMP_PAGE * D
    fused_compressed_storage = torch.zeros(
        num_compressed_blocks,
        compressed_page_elements + 16,
        dtype=indexer_dtype,
        device=device,
    )
    fused_compressed = torch.as_strided(
        fused_compressed_storage,
        (num_compressed_blocks, COMP_PAGE, 1, D),
        (compressed_page_elements + 16, D, D, 1),
    )
    # The reference scatters at the compute dtype and narrows once at the end. Writing
    # e4m3 through qsa_store_cache_rows would need a Triton fp8 store, which is not
    # available on every architecture this runs on -- and narrowing the pooled rows
    # early would not be the same arithmetic anyway. The kernel likewise rounds to the
    # compute dtype and converts only in its final store.
    unfused_compressed_dtype = (
        torch.bfloat16 if indexer_dtype == torch.float8_e4m3fn else indexer_dtype
    )
    unfused_compressed_storage = torch.zeros(
        num_compressed_blocks,
        compressed_page_elements + 16,
        dtype=unfused_compressed_dtype,
        device=device,
    )
    unfused_compressed = torch.as_strided(
        unfused_compressed_storage,
        (num_compressed_blocks, COMP_PAGE, 1, D),
        (compressed_page_elements + 16, D, D, 1),
    )

    projected_qk = torch.randn(
        num_tokens, (HQ + 1) * D, dtype=torch.bfloat16, device=device
    )
    q_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2
    k_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2

    fused_query = torch.empty(num_tokens, HQ, D, dtype=indexer_dtype, device=device)
    qsa_pre_indexer(
        projected_qk[:, : HQ * D],
        projected_qk[:, HQ * D :],
        positions,
        rope.cos_sin_cache,
        q_weight,
        k_weight,
        EPS,
        fused_query,
        fused_raw,
        raw_slots,
        raw_block_table,
        query_start_loc,
        logical_positions,
        fused_compressed,
        compressed_slots,
        k_work_metadata,
        compress_ratio=CR,
        mrope_section=MROPE_SECTION if mrope else None,
        rope_pos_offset=ROPE_POS_OFFSET if cache_rope_positions else None,
    )

    def _rerun_fused_wide():
        """The same fused call writing the compute dtype, on the same inputs.

        The kernel mutates both caches, so the ring and the compressed pages are
        rebuilt from the pre-call state rather than reused.
        """
        wide_query = torch.empty(num_tokens, HQ, D, dtype=torch.bfloat16, device=device)
        wide_raw = pre_call_raw.clone()
        wide_compressed_storage = torch.zeros_like(
            fused_compressed_storage, dtype=torch.bfloat16
        )
        wide_compressed = torch.as_strided(
            wide_compressed_storage,
            (num_compressed_blocks, COMP_PAGE, 1, D),
            (compressed_page_elements + 16, D, D, 1),
        )
        qsa_pre_indexer(
            projected_qk[:, : HQ * D],
            projected_qk[:, HQ * D :],
            positions,
            rope.cos_sin_cache,
            q_weight,
            k_weight,
            EPS,
            wide_query,
            wide_raw,
            raw_slots,
            raw_block_table,
            query_start_loc,
            logical_positions,
            wide_compressed,
            compressed_slots,
            k_work_metadata,
            compress_ratio=CR,
            mrope_section=MROPE_SECTION if mrope else None,
            rope_pos_offset=ROPE_POS_OFFSET if cache_rope_positions else None,
        )
        return wide_query, wide_compressed

    unfused_query = projected_qk[:, : HQ * D].reshape(num_tokens, HQ, D)
    unfused_query = gemma_rmsnorm(
        unfused_query.reshape(-1, D), q_weight, EPS
    ).reshape_as(unfused_query)
    unfused_query = apply_qsa_rope(rope, positions, unfused_query)

    raw_keys = unfused_raw[..., :D]
    rope_positions = (
        unfused_raw[..., ROPE_POS_OFFSET:].view(torch.int64)
        if cache_rope_positions
        else None
    )
    pooled, first_positions = qsa_compress_groups_with_ratio(
        projected_qk[:, HQ * D :].reshape(-1, 1, D),
        position_rows,
        raw_keys,
        raw_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        CR,
        rope_positions,
    )
    compressed_rows = gemma_rmsnorm(pooled.reshape(-1, D), k_weight, EPS).reshape(
        -1, 1, D
    )
    group_positions = (
        first_positions.transpose(0, 1) if mrope else first_positions[:, 0]
    )
    compressed_rows = apply_qsa_rope(rope, group_positions, compressed_rows)
    qsa_store_cache_rows(unfused_compressed, compressed_slots, compressed_rows)
    qsa_store_cache_rows(raw_keys, raw_slots, projected_qk[:, HQ * D :])
    if rope_positions is not None:
        qsa_store_cache_rows(rope_positions, raw_slots, position_rows)

    if indexer_dtype == torch.float8_e4m3fn:
        # What the narrowing arm has to equal is the same kernel writing wide and then
        # rounded once, not the unfused path. Those two implementations already differ
        # by up to 2**-5 at bf16, which RTOL/ATOL absorbs and a code-step rule cannot,
        # so measuring the narrowing against that difference would let a wiring mistake
        # hide inside it. The unfused comparison is still made, at the width where its
        # own tolerance applies.
        wide_query, wide_compressed = _rerun_fused_wide()
        assert_fp8_within_one_ulp(fused_query, wide_query.to(indexer_dtype))
        assert_fp8_within_one_ulp(fused_compressed, wide_compressed.to(indexer_dtype))
        torch.testing.assert_close(wide_query, unfused_query, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            wide_compressed, unfused_compressed, rtol=RTOL, atol=ATOL
        )
    else:
        torch.testing.assert_close(fused_query, unfused_query, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            fused_compressed, unfused_compressed, rtol=RTOL, atol=ATOL
        )
    assert torch.equal(fused_raw.view(torch.int16), unfused_raw.view(torch.int16))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the rotary table cache is a CUDA path"
)
def test_paired_cos_sin_is_not_keyed_on_an_address() -> None:
    """A freed table's address can come back under a different one."""
    import gc

    from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import (
        _PAIRED_COS_SIN,
        _paired_cos_sin,
    )

    first = torch.arange(2 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128)
    paired_first = _paired_cos_sin(first)
    assert _paired_cos_sin(first) is paired_first, "the same table is built once"

    held = paired_first.clone()
    del first, paired_first
    gc.collect()
    # Other tables built by other tests are alive and stay cached, so what is
    # counted here is this one's entry rather than the whole cache.
    baseline = len(_PAIRED_COS_SIN)

    # Whatever lands here next is a different table, and must not be answered
    # with the one built for whatever was here before.
    second = torch.arange(2 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128)
    second += 1
    paired_second = _paired_cos_sin(second)
    assert len(_PAIRED_COS_SIN) == baseline + 1
    assert not torch.equal(paired_second, held)
    cos, sin = second.chunk(2, dim=-1)
    expected = torch.stack((cos, sin), dim=-1).reshape_as(second)
    torch.testing.assert_close(paired_second, expected)
    del second, paired_second
    gc.collect()
    assert len(_PAIRED_COS_SIN) == baseline, "the entry goes when the table does"


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the rotary table cache is a CUDA path"
)
def test_paired_cos_sin_refuses_to_build_under_capture() -> None:
    """A table built under capture lives in the graph's pool, not the caller's."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import _paired_cos_sin

    table = torch.arange(2 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.empty(1, device="cuda")
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="before the graph is captured"),
        torch.cuda.graph(graph),
    ):
        _paired_cos_sin(table)

    # Built ahead of capture, the same table is served from the cache instead.
    built = _paired_cos_sin(table)
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        assert _paired_cos_sin(table) is built


def _minimal_pre_indexer_args(out_dtype: torch.dtype, device="cuda"):
    """The smallest argument set the pre-indexer accepts, one request, one head."""
    tokens, heads, dim, ratio, ring, page = 4, 1, 128, 4, 4, 4
    compute = torch.bfloat16
    return dict(
        q=torch.randn(tokens, heads * dim, dtype=compute, device=device),
        k=torch.randn(tokens, dim, dtype=compute, device=device),
        positions=torch.arange(tokens, dtype=torch.int64, device=device),
        cos_sin_cache=torch.randn(64, dim // 2, dtype=compute, device=device),
        q_norm_weight=torch.randn(dim, dtype=compute, device=device) * 0.2,
        k_norm_weight=torch.randn(dim, dtype=compute, device=device) * 0.2,
        eps=1e-6,
        q_out=torch.empty(tokens, heads, dim, dtype=out_dtype, device=device),
        state_cache=torch.zeros(1, ring, 1, dim, dtype=compute, device=device),
        state_slots=torch.arange(tokens, dtype=torch.int64, device=device),
        state_block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        logical_positions=torch.arange(tokens, dtype=torch.int64, device=device),
        compressed_cache=torch.zeros(1, page, 1, dim, dtype=out_dtype, device=device),
        compressed_slots=torch.zeros(tokens, dtype=torch.int64, device=device),
        k_work_metadata=torch.tensor([[0, 0]], dtype=torch.int32, device=device),
        compress_ratio=ratio,
    )


def _run_minimal_pre_indexer(out_dtype: torch.dtype):
    args = _minimal_pre_indexer_args(out_dtype)
    ratio = args.pop("compress_ratio")
    pre_indexer_module.qsa_pre_indexer(
        *[
            args[name]
            for name in (
                "q",
                "k",
                "positions",
                "cos_sin_cache",
                "q_norm_weight",
                "k_norm_weight",
                "eps",
                "q_out",
                "state_cache",
                "state_slots",
                "state_block_table",
                "query_start_loc",
                "logical_positions",
                "compressed_cache",
                "compressed_slots",
                "k_work_metadata",
            )
        ],
        compress_ratio=ratio,
        mrope_section=None,
        rope_pos_offset=None,
    )
    torch.accelerator.synchronize()
    return args["q_out"], args["compressed_cache"]


def _clear_pre_indexer_caches() -> None:
    """Both answers are cached, and a stale one passes these tests for free."""
    pre_indexer_module._has_cuda_pre_indexer.cache_clear()
    pre_indexer_module._pre_indexer_accepts_dtypes.cache_clear()


@pytest.fixture
def clean_pre_indexer_caches():
    _clear_pre_indexer_caches()
    yield
    _clear_pre_indexer_caches()


@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_pre_indexer_capability_reads_the_compiled_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask without the narrowing bit keeps fp8 off the CUDA path."""
    import flashinfer.sparse_pre_indexer as fi

    monkeypatch.setattr(pre_indexer_module, "_has_cuda_pre_indexer", lambda: True)
    monkeypatch.setattr(
        fi, "qsa_pre_indexer_dispatch_mask", lambda: fi.QSA_PRE_INDEXER_SAME_AS_COMPUTE
    )

    accepts = pre_indexer_module._pre_indexer_accepts_dtypes
    assert accepts(torch.bfloat16, torch.bfloat16)
    assert not accepts(torch.bfloat16, torch.float8_e4m3fn)


@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_pre_indexer_capability_without_the_python_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older FlashInfer keeps bf16 on CUDA and sends only fp8 to Triton."""
    import flashinfer.sparse_pre_indexer as fi

    monkeypatch.setattr(pre_indexer_module, "_has_cuda_pre_indexer", lambda: True)
    monkeypatch.delattr(fi, "qsa_pre_indexer_dispatch_mask")

    accepts = pre_indexer_module._pre_indexer_accepts_dtypes
    assert accepts(torch.bfloat16, torch.bfloat16)
    assert not accepts(torch.bfloat16, torch.float8_e4m3fn)


@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_pre_indexer_capability_with_a_stale_compiled_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current wrapper over a build that predates the arm reports the old set.

    Distinct from the missing-wrapper case: here the Python side is new and it is the
    binary that is behind, which is what a warm JIT or AOT cache produces.
    """
    import flashinfer.sparse_pre_indexer as fi

    # A compiled module resolves its exports dynamically, so the way to stand in for
    # one that lacks the symbol is to hand the wrapper a module that does not have it.
    stale = SimpleNamespace(qsa_pre_indexer=lambda *args, **kwargs: None)
    monkeypatch.setattr(fi, "get_sparse_pre_indexer_module", lambda: stale)
    monkeypatch.setattr(pre_indexer_module, "_has_cuda_pre_indexer", lambda: True)

    assert fi.qsa_pre_indexer_dispatch_mask() == fi.QSA_PRE_INDEXER_SAME_AS_COMPUTE
    accepts = pre_indexer_module._pre_indexer_accepts_dtypes
    assert accepts(torch.bfloat16, torch.bfloat16)
    assert not accepts(torch.bfloat16, torch.float8_e4m3fn)


@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_pre_indexer_capability_propagates_a_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the module has built, a failing query is a defect, not a "no"."""
    import flashinfer.sparse_pre_indexer as fi

    def boom() -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(pre_indexer_module, "_has_cuda_pre_indexer", lambda: True)
    monkeypatch.setattr(fi, "qsa_pre_indexer_dispatch_mask", boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        pre_indexer_module._pre_indexer_accepts_dtypes(
            torch.bfloat16, torch.float8_e4m3fn
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the fallback question is a CUDA one"
)
@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_warmup_refuses_fp8_when_no_path_can_write_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a capable FlashInfer, an e4m3 output has no working path here.

    Triton is not a fallback for it: ``tl.store`` into an fp8 pointer needs
    ``fp8e4nv``, which Triton compiles only from sm_89 on. Letting the capability
    query answer "no" and falling through would turn that into a compile error at the
    first forward, so the warmup says it instead, while the message can still name a
    remedy.
    """
    monkeypatch.setattr(
        pre_indexer_module, "_pre_indexer_accepts_dtypes", lambda *_: False
    )
    if pre_indexer_module._triton_can_write(torch.float8_e4m3fn):
        pytest.skip("this architecture's Triton compiles fp8e4nv")

    with pytest.raises(RuntimeError, match="cannot write"):
        pre_indexer_module.warmup_pre_indexer_capability(
            torch.bfloat16, torch.float8_e4m3fn
        )
    # bf16 has a working Triton path, so it must not be refused.
    pre_indexer_module.warmup_pre_indexer_capability(torch.bfloat16, torch.bfloat16)


@requires_qsa_kernels
@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_triton_fallback_cannot_write_fp8_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the fact the refusal above rests on, by running the fallback for real."""
    if pre_indexer_module._triton_can_write(torch.float8_e4m3fn):
        pytest.skip("this architecture's Triton compiles fp8e4nv")
    monkeypatch.setattr(pre_indexer_module, "_has_cuda_pre_indexer", lambda: False)

    from triton.compiler.errors import CompilationError

    with pytest.raises(CompilationError, match="fp8e4nv"):
        _run_minimal_pre_indexer(torch.float8_e4m3fn)

    # The same call at the compute dtype goes through, so the failure is the dtype and
    # not the arguments.
    _run_minimal_pre_indexer(torch.bfloat16)


@pytest.mark.usefixtures("clean_pre_indexer_caches")
def test_pre_indexer_capability_rejects_an_unbuildable_compute_dtype() -> None:
    """The compute axis is FlashInfer's own, and float32 is not on it."""
    assert not pre_indexer_module._pre_indexer_accepts_dtypes(
        torch.float32, torch.float32
    )


# Run in a subprocess: a fresh JIT directory is not a cold start on its own, because
# ``get_sparse_pre_indexer_module`` is a module-level cache and FlashInfer reads its
# workspace base at import time.
_COLD_START_CAPTURE_PROBE = """
import torch

import flashinfer.sparse_pre_indexer as fi
import vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer as m
from flashinfer.jit.core import JitSpec
from flashinfer.jit.sparse_pre_indexer import gen_sparse_pre_indexer_module

# build() is overridden per backend, so count it on the class the spec actually is.
SpecType = type(gen_sparse_pre_indexer_module())

loads = 0
builds = 0
original_build_and_load = JitSpec.build_and_load
original_build = SpecType.build


def counting_build_and_load(self, *args, **kwargs):
    global loads
    loads += 1
    return original_build_and_load(self, *args, **kwargs)


def counting_build(self, *args, **kwargs):
    global builds
    builds += 1
    return original_build(self, *args, **kwargs)


JitSpec.build_and_load = counting_build_and_load
SpecType.build = counting_build

OUT_DTYPE = torch.float8_e4m3fn
T, HQ, D, R, RING, PAGE = 4, 1, 128, 4, 4, 4
CT = torch.bfloat16
COS_SIN = torch.randn(64, D // 2, dtype=CT, device="cuda")
POSITIONAL = (
    torch.randn(T, HQ * D, dtype=CT, device="cuda"),
    torch.randn(T, D, dtype=CT, device="cuda"),
    torch.arange(T, dtype=torch.int64, device="cuda"),
    COS_SIN,
    torch.randn(D, dtype=CT, device="cuda") * 0.2,
    torch.randn(D, dtype=CT, device="cuda") * 0.2,
    1e-6,
    torch.empty(T, HQ, D, dtype=OUT_DTYPE, device="cuda"),
    torch.zeros(1, RING, 1, D, dtype=CT, device="cuda"),
    torch.arange(T, dtype=torch.int64, device="cuda"),
    torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
    torch.tensor([0, T], dtype=torch.int32, device="cuda"),
    torch.arange(T, dtype=torch.int64, device="cuda"),
    torch.zeros(1, PAGE, 1, D, dtype=OUT_DTYPE, device="cuda"),
    torch.zeros(T, dtype=torch.int64, device="cuda"),
    torch.tensor([[0, 0]], dtype=torch.int32, device="cuda"),
)
KEYWORDS = dict(compress_ratio=R, mrope_section=None, rope_pos_offset=None)

# What the profiling run does, and all of it.
m.warmup_pre_indexer_capability(CT, OUT_DTYPE, COS_SIN)
assert m._pre_indexer_accepts_dtypes(CT, OUT_DTYPE), "FlashInfer arm missing"
assert loads > 0, "the warmup did not reach build_and_load"
assert builds > 0, "a cold cache should have compiled the module, not loaded it"


def counters():
    return (
        fi.get_sparse_pre_indexer_module.cache_info().misses,
        loads,
        builds,
        m._has_cuda_pre_indexer.cache_info().misses,
        m._pre_indexer_accepts_dtypes.cache_info().misses,
    )


before = counters()

# Production reaches the kernel for the first time inside a capture: the profiling run
# returns before the fused branch, so the warmup above is all that precedes it. The
# kernel is not run beforehand here either, or anything a first launch does would be
# hidden.
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    torch.empty(1, device="cuda")
torch.cuda.current_stream().wait_stream(side)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    m.qsa_pre_indexer(*POSITIONAL, **KEYWORDS)
graph.replay()
torch.accelerator.synchronize()

after = counters()
assert before == after, (
    "work happened during capture: " + str(before) + " -> " + str(after)
)
print("OK")
"""


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="graph capture is a CUDA path"
)
def test_capability_is_settled_before_graph_capture(tmp_path) -> None:
    """No build, load or capability query may happen inside a capture.

    The kernel itself is captured, as it must be; what this pins is that the FlashInfer
    question is already answered by then. Counting accessor calls would not show it --
    the wrapper calls the cached getter on every launch -- so this watches cache misses
    and builds instead.
    """
    import subprocess
    import sys

    env = dict(os.environ, FLASHINFER_WORKSPACE_BASE=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", _COLD_START_CAPTURE_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
