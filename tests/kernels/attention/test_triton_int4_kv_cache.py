# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    INT4_SCALE_GROUP_SIZE,
    INT4_SCALE_DTYPE,
    KVQuantMode,
    get_int4_inline_head_size_bytes,
    get_int4_packed_head_size,
    get_int4_packed_head_size_padded,
    get_int4_scale_group_count,
    get_kv_quant_mode,
)


def _expand_scale(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if scale.ndim == x.ndim - 1:
        scale = scale.unsqueeze(-1)
    if scale.shape[-1] in (1, x.shape[-1]):
        return scale
    return scale.repeat_interleave(INT4_SCALE_GROUP_SIZE, dim=-1)[..., : x.shape[-1]]


def _quantize_int4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.round(x.float() / _expand_scale(scale, x)).clamp(-8, 7).to(
        torch.int32
    )


def _dequantize_int4_values(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * _expand_scale(scale, q)


def _pack_int4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = _quantize_int4(x, scale)
    q_u4 = torch.where(q < 0, q + 16, q).to(torch.uint8)
    if q_u4.shape[-1] % 2:
        q_u4 = torch.nn.functional.pad(q_u4, (0, 1))
    return q_u4[..., 0::2] | (q_u4[..., 1::2] << 4)


def _dequantize_packed_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    out = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        device=packed.device,
        dtype=torch.float32,
    )
    scale_expanded = scale.to(torch.float32)
    if scale_expanded.ndim == packed.ndim - 1:
        scale_expanded = scale_expanded.unsqueeze(-1)
    if scale_expanded.shape[-1] not in (1, packed.shape[-1]):
        scale_expanded = scale_expanded.repeat_interleave(
            INT4_SCALE_GROUP_SIZE // 2, dim=-1
        )[..., : packed.shape[-1]]
    out[..., 0::2] = low.to(torch.float32) * scale_expanded
    out[..., 1::2] = high.to(torch.float32) * scale_expanded
    return out[..., :head_size]


def _int4_scales_per_token_head(x: torch.Tensor) -> torch.Tensor:
    return x.abs().amax(dim=-1).float().clamp(min=1e-6) / 7.0


def _int4_grouped_scales(x: torch.Tensor) -> torch.Tensor:
    pad = (-x.shape[-1]) % INT4_SCALE_GROUP_SIZE
    x_padded = F.pad(x.float(), (0, pad))
    grouped = x_padded.view(*x.shape[:-1], -1, INT4_SCALE_GROUP_SIZE)
    return grouped.abs().amax(dim=-1).clamp(min=1e-6) / 7.0


def _get_int4_inline_scale_views(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key_cache, value_cache = kv_cache.unbind(1)
    _, block_size, num_heads, padded_hs = key_cache.shape
    assert padded_hs == get_int4_inline_head_size_bytes(head_size)
    num_groups = get_int4_scale_group_count(head_size)
    data_hs_padded = get_int4_packed_head_size_padded(head_size)
    assert num_groups >= 1

    raw = kv_cache.untyped_storage()
    scale_size = torch.tensor([], dtype=INT4_SCALE_DTYPE).element_size()
    assert data_hs_padded % scale_size == 0
    base_scale = torch.tensor(
        [], dtype=INT4_SCALE_DTYPE, device=kv_cache.device
    ).set_(raw)

    kv_half_bytes = block_size * num_heads * padded_hs
    full_block_scale = 2 * kv_half_bytes // scale_size
    slot_scale = num_heads * padded_hs // scale_size
    head_scale = padded_hs // scale_size
    scale_off = data_hs_padded // scale_size

    k_scale_cache = torch.as_strided(
        base_scale,
        size=(kv_cache.shape[0], block_size, num_heads, num_groups),
        stride=(full_block_scale, slot_scale, head_scale, 1),
        storage_offset=scale_off,
    )
    v_scale_cache = torch.as_strided(
        base_scale,
        size=(kv_cache.shape[0], block_size, num_heads, num_groups),
        stride=(full_block_scale, slot_scale, head_scale, 1),
        storage_offset=(kv_half_bytes // scale_size) + scale_off,
    )
    k_scale_cache.fill_(1.0)
    v_scale_cache.fill_(1.0)
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def _ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start_idx = 0
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    for i, query_len in enumerate(query_lens):
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        scores = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=q.device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        scores.masked_fill_(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def test_int4_kv_cache_shape_and_page_size():
    assert get_kv_quant_mode("int4_per_token_head") == KVQuantMode.INT4_PER_TOKEN_HEAD
    assert TritonAttentionBackend.get_kv_cache_shape(
        8, 16, 4, 128, "int4_per_token_head"
    ) == (
        8,
        2,
        16,
        4,
        72,
    )

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    )
    assert spec.page_size_bytes == 16 * 4 * (72 + 72)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_int4_inline_scale_views_respect_padded_block_stride():
    device = "cuda"
    num_blocks = 3
    block_size = 16
    num_kv_heads = 4
    head_size = 128
    num_groups = get_int4_scale_group_count(head_size)

    kv_cache_shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, "int4_per_token_head"
    )
    contiguous_block_elems = kv_cache_shape[1] * kv_cache_shape[2] * kv_cache_shape[3] * kv_cache_shape[4]
    padded_block_elems = contiguous_block_elems + 128
    raw = torch.zeros(num_blocks * padded_block_elems, dtype=torch.uint8, device=device)

    half_stride = kv_cache_shape[2] * kv_cache_shape[3] * kv_cache_shape[4]
    slot_stride = kv_cache_shape[3] * kv_cache_shape[4]
    head_stride = kv_cache_shape[4]
    kv_cache = torch.as_strided(
        raw,
        size=kv_cache_shape,
        stride=(padded_block_elems, half_stride, slot_stride, head_stride, 1),
    )

    impl = TritonAttentionImpl(
        num_heads=num_kv_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
        attn_type=AttentionType.DECODER,
    )
    impl._ensure_scale_caches(kv_cache)
    assert impl._k_scale_cache is not None
    assert impl._v_scale_cache is not None

    for blk in range(num_blocks):
        for group_idx in range(num_groups):
            impl._k_scale_cache[blk, :, :, group_idx].fill_(10.0 * (blk + 1) + group_idx)
            impl._v_scale_cache[blk, :, :, group_idx].fill_(20.0 * (blk + 1) + group_idx)

    scale_size = torch.tensor([], dtype=INT4_SCALE_DTYPE).element_size()
    base_scale = torch.tensor(
        [], dtype=INT4_SCALE_DTYPE, device=device
    ).set_(kv_cache.untyped_storage())
    block_scale = padded_block_elems // scale_size
    half_scale = half_stride // scale_size
    slot_scale = slot_stride // scale_size
    head_scale = head_stride // scale_size
    scale_off = get_int4_packed_head_size_padded(head_size) // scale_size

    for blk in range(num_blocks):
        for slot in (0, block_size - 1):
            for head in (0, num_kv_heads - 1):
                for group_idx in range(num_groups):
                    k_idx = blk * block_scale + slot * slot_scale + head * head_scale + scale_off + group_idx
                    v_idx = blk * block_scale + half_scale + slot * slot_scale + head * head_scale + scale_off + group_idx
                    assert base_scale[k_idx].item() == pytest.approx(
                        10.0 * (blk + 1) + group_idx
                    )
                    assert base_scale[v_idx].item() == pytest.approx(
                        20.0 * (blk + 1) + group_idx
                    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_triton_reshape_and_cache_flash_int4():
    device = "cuda"
    set_random_seed(0)
    torch.set_default_device(device)

    num_tokens = 23
    num_heads = 4
    head_size = 97
    block_size = 16
    num_blocks = 8
    packed_head_size = get_int4_packed_head_size(head_size)
    packed_head_size_padded = get_int4_packed_head_size_padded(head_size)
    num_groups = get_int4_scale_group_count(head_size)

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slot_mapping = torch.randperm(num_blocks * block_size, device=device)[:num_tokens]

    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_heads,
        get_int4_inline_head_size_bytes(head_size),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache, k_scale_cache, v_scale_cache = _get_int4_inline_scale_views(
        kv_cache, head_size
    )

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "int4_per_token_head",
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        k_scale_cache,
        v_scale_cache,
    )

    got_key = _dequantize_packed_int4(
        key_cache[..., :packed_head_size], k_scale_cache, head_size
    )
    got_value = _dequantize_packed_int4(
        value_cache[..., :packed_head_size], v_scale_cache, head_size
    )

    expected_key = torch.zeros(
        num_blocks, block_size, num_heads, head_size, device=device, dtype=torch.float32
    )
    expected_value = torch.zeros_like(expected_key)
    expected_k_scales = torch.ones(
        num_blocks,
        block_size,
        num_heads,
        num_groups,
        device=device,
        dtype=INT4_SCALE_DTYPE,
    )
    expected_v_scales = torch.ones_like(expected_k_scales)
    key_scales = _int4_grouped_scales(key)
    value_scales = _int4_grouped_scales(value)
    key_dequant_scales = key_scales.to(INT4_SCALE_DTYPE)
    value_dequant_scales = value_scales.to(INT4_SCALE_DTYPE)
    key_roundtrip = _dequantize_int4_values(
        _quantize_int4(key, key_scales), key_dequant_scales
    )
    value_roundtrip = _dequantize_int4_values(
        _quantize_int4(value, value_scales), value_dequant_scales
    )

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        block_idx = slot // block_size
        block_offset = slot % block_size
        expected_key[block_idx, block_offset] = key_roundtrip[token_idx]
        expected_value[block_idx, block_offset] = value_roundtrip[token_idx]
        expected_k_scales[block_idx, block_offset] = key_dequant_scales[token_idx]
        expected_v_scales[block_idx, block_offset] = value_dequant_scales[token_idx]

    atol = (
        max(
            float(key_dequant_scales.float().max().item()),
            float(value_dequant_scales.float().max().item()),
        )
        + 1e-4
    )
    torch.testing.assert_close(got_key, expected_key, atol=atol, rtol=0.0)
    torch.testing.assert_close(got_value, expected_value, atol=atol, rtol=0.0)
    torch.testing.assert_close(
        k_scale_cache.float(), expected_k_scales.float(), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        v_scale_cache.float(), expected_v_scales.float(), atol=1e-4, rtol=1e-4
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_triton_reshape_and_cache_flash_int4_rejects_mismatched_v_head_size():
    device = "cuda"
    set_random_seed(7)
    torch.set_default_device(device)

    num_tokens = 2
    num_heads = 2
    head_size = 128
    head_size_v = 64
    block_size = 16
    num_blocks = 2
    num_groups = get_int4_scale_group_count(head_size)

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size_v, dtype=torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)

    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        get_int4_inline_head_size_bytes(head_size),
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        get_int4_inline_head_size_bytes(head_size_v),
        dtype=torch.uint8,
        device=device,
    )
    k_scale_cache = torch.ones(
        num_blocks,
        block_size,
        num_heads,
        num_groups,
        dtype=INT4_SCALE_DTYPE,
        device=device,
    )
    v_scale_cache = torch.ones_like(k_scale_cache)

    with pytest.raises(ValueError, match="head_size_v == head_size"):
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "int4_per_token_head",
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            k_scale_cache,
            v_scale_cache,
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_triton_unified_attention_int4():
    device = "cuda"
    set_random_seed(1)
    torch.set_default_device(device)

    query_lens = [1, 5, 3]
    kv_lens = [33, 18, 27]
    num_blocks = 64
    block_size = 16
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)

    k_quant_scale = _int4_grouped_scales(key_cache)
    v_quant_scale = _int4_grouped_scales(value_cache)
    k_scale = k_quant_scale.to(INT4_SCALE_DTYPE)
    v_scale = v_quant_scale.to(INT4_SCALE_DTYPE)
    key_cache_int4 = _pack_int4(key_cache, k_quant_scale)
    value_cache_int4 = _pack_int4(value_cache, v_quant_scale)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32, device=device)
    cu_query_lens = cu_query_lens.cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (len(query_lens), max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_int4,
        v=value_cache_int4,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max_kv_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )

    ref_output = _ref_paged_attn(
        query=query.float(),
        key_cache=_dequantize_packed_int4(key_cache_int4, k_scale, head_size),
        value_cache=_dequantize_packed_int4(value_cache_int4, v_scale, head_size),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=head_size**-0.5,
    )
    torch.testing.assert_close(output.float(), ref_output.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@torch.inference_mode()
def test_triton_unified_attention_int4_decode_only_grouped_scale():
    device = "cuda"
    set_random_seed(2)
    torch.set_default_device(device)

    query_lens = [1, 1, 1]
    kv_lens = [33, 18, 27]
    num_blocks = 64
    block_size = 16
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)

    k_quant_scale = _int4_grouped_scales(key_cache)
    v_quant_scale = _int4_grouped_scales(value_cache)
    k_scale = k_quant_scale.to(INT4_SCALE_DTYPE)
    v_scale = v_quant_scale.to(INT4_SCALE_DTYPE)
    key_cache_int4 = _pack_int4(key_cache, k_quant_scale)
    value_cache_int4 = _pack_int4(value_cache, v_quant_scale)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32, device=device)
    cu_query_lens = cu_query_lens.cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (len(query_lens), max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    output = torch.empty_like(query)
    num_par_softmax_segments = 16
    head_size_padded = 128
    softmax_segm_output = torch.empty(
        (8, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_max = torch.empty(
        (8, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_expsum = torch.empty(
        (8, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device=device,
    )

    unified_attention(
        q=query,
        k=key_cache_int4,
        v=value_cache_int4,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max_kv_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=8,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )

    ref_output = _ref_paged_attn(
        query=query.float(),
        key_cache=_dequantize_packed_int4(key_cache_int4, k_scale, head_size),
        value_cache=_dequantize_packed_int4(value_cache_int4, v_scale, head_size),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=head_size**-0.5,
    )
    torch.testing.assert_close(output.float(), ref_output.float(), atol=2e-2, rtol=2e-2)


def test_int4_grouped_scale_reduces_outlier_mse():
    head_size = 128
    x = torch.full((1, 1, head_size), 0.25, dtype=torch.float32)
    x[..., 0] = 6.0
    x[..., 1] = -5.5

    grouped_scale = _int4_grouped_scales(x)
    grouped_dequant_scale = grouped_scale.to(INT4_SCALE_DTYPE)
    grouped_roundtrip = _dequantize_int4_values(
        _quantize_int4(x, grouped_scale), grouped_dequant_scale
    )

    baseline_scale = _int4_scales_per_token_head(x)
    baseline_roundtrip = _dequantize_int4_values(
        _quantize_int4(x, baseline_scale), baseline_scale
    )

    non_outlier = slice(2, None)
    grouped_mse = torch.mean(
        (grouped_roundtrip[..., non_outlier] - x[..., non_outlier]) ** 2
    )
    baseline_mse = torch.mean(
        (baseline_roundtrip[..., non_outlier] - x[..., non_outlier]) ** 2
    )

    assert grouped_mse < baseline_mse
