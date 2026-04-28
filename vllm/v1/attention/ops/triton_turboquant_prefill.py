# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton TurboQuant native prefill attention.

This module intentionally reads the compressed TurboQuant KV cache directly.
It is the long-prefill counterpart to ``triton_turboquant_decode.py`` and avoids
materializing dense K/V pages or calling FlashAttention in the production path.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_decode import _QJL_ALPHA, _get_layout

logger = init_logger(__name__)

_MAX_Q_PROJECTION_BYTES = 1 << 30

def _select_gqa_group_tile(head_dim: int, kv_group_size: int) -> int:
    """Return GQA group tile for native prefill, or 1 for scalar path."""
    if kv_group_size <= 1:
        return 1
    if head_dim == 512:
        return 2 if kv_group_size % 2 == 0 else 1
    if head_dim != 256:
        return 1
    # GROUP_TILE=2 is the only currently validated speedup point. GROUP_TILE=3
    # compiles with padded rows but is slower on the Qwen3.6 profile shape.
    if kv_group_size % 2 == 0:
        return 2
    return 1


@triton.jit
def _tq_load_slot_bases(
    Block_table_ptr,
    stride_bt_n,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    start_n,
    offs_n,
    end_n,
    kv_head,
    BLOCK_SIZE: tl.constexpr,
):
    kv_offs = start_n + offs_n
    kv_mask = kv_offs < end_n
    page_idx = kv_offs // BLOCK_SIZE
    page_off = kv_offs % BLOCK_SIZE
    block_nums = tl.load(
        Block_table_ptr + page_idx * stride_bt_n,
        mask=kv_mask,
        other=0,
    )
    slot_bases = (
        block_nums * stride_cache_block
        + page_off * stride_cache_pos
        + kv_head * stride_cache_head
    )
    return kv_offs, kv_mask, slot_bases


@triton.jit
def _tq_apply_prefill_mask(
    scores,
    q_mask,
    kv_mask,
    q_pos,
    kv_offs,
    SLIDING_WINDOW: tl.constexpr,
):
    causal_mask = kv_offs[None, :] <= q_pos[:, None]
    valid_mask = q_mask[:, None] & kv_mask[None, :] & causal_mask
    if SLIDING_WINDOW > 0:
        window_start = q_pos - SLIDING_WINDOW + 1
        valid_mask = valid_mask & (kv_offs[None, :] >= window_start[:, None])
    return tl.where(valid_mask, scores, -float("inf"))


@triton.jit
def _tq_compute_scores_from_cache(
    q_rot,
    q_qjl,
    Key_norm_lut_ptr,
    KV_cache_ptr,
    Centroids_ptr,
    slot_bases,
    kv_mask,
    d_mask,
    offs_d,
    HEAD_DIM: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KEY_NORM_BYTES: tl.constexpr,
    KEY_NORM_OFFSET: tl.constexpr,
    RES_NORM_OFFSET: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    QJL_ALPHA: tl.constexpr,
    RES_NORM_SCALE: tl.constexpr,
):
    mse_bit_off = offs_d * MSE_BITS
    mse_byte_idx = mse_bit_off // 8
    mse_bit_shift = mse_bit_off % 8
    mse_mask = (1 << MSE_BITS) - 1

    mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
    mse_raw0 = tl.load(
        KV_cache_ptr + mse_addrs0,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0,
    ).to(tl.int32)
    mse_raw1 = tl.load(
        KV_cache_ptr + mse_addrs0 + 1,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0,
    ).to(tl.int32)
    raw16 = mse_raw0 | (mse_raw1 << 8)
    mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
    c_vals = tl.load(
        Centroids_ptr + mse_idx,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    term_base = tl.dot(q_rot, tl.trans(c_vals), input_precision="ieee")

    qjl_byte_idx = offs_d // 8
    qjl_bit_shift = offs_d % 8
    qjl_addrs = slot_bases[:, None] + MSE_BYTES + qjl_byte_idx[None, :]
    qjl_raw = tl.load(
        KV_cache_ptr + qjl_addrs,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0,
    ).to(tl.int32)
    qjl_bit = (qjl_raw >> qjl_bit_shift[None, :]) & 0x1
    qjl_sign = (qjl_bit.to(tl.float32) * 2.0 - 1.0).to(tl.float16)
    term_qjl = tl.dot(q_qjl, tl.trans(qjl_sign), input_precision="ieee")

    kn_base = slot_bases + KEY_NORM_OFFSET
    if KEY_NORM_BYTES == 1:
        kn_raw = tl.load(KV_cache_ptr + kn_base, mask=kv_mask, other=0).to(tl.int32)
        key_norm = tl.load(Key_norm_lut_ptr + kn_raw, mask=kv_mask, other=0.0)
    else:
        kn_lo = tl.load(KV_cache_ptr + kn_base, mask=kv_mask, other=0).to(tl.uint16)
        kn_hi = tl.load(KV_cache_ptr + kn_base + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        key_norm = (kn_lo | (kn_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    rn_base = slot_bases + RES_NORM_OFFSET
    rn_raw = tl.load(KV_cache_ptr + rn_base, mask=kv_mask, other=0).to(tl.float32)
    residual_norm = rn_raw * (RES_NORM_SCALE / 255.0)

    qjl_corr = (QJL_ALPHA / HEAD_DIM) * residual_norm[None, :] * term_qjl
    return key_norm[None, :] * (term_base + qjl_corr) * ATTN_SCALE


@triton.jit
def _tq_load_values_from_cache(
    KV_cache_ptr,
    slot_bases,
    kv_mask,
    d_mask,
    offs_d,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
):
    val_bases = slot_bases + KPS
    if VQB == 3:
        val_bit_off = offs_d * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
        val_raw0 = tl.load(
            KV_cache_ptr + val_addrs0,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_addrs0 + 1,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
    else:
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4
        val_addrs = val_bases[:, None] + vb_idx[None, :]
        val_raw = tl.load(
            KV_cache_ptr + val_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

    sc_bases = val_bases + VAL_DATA_BYTES
    sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
    sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
    v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
    zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
    v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    values = v_idx * v_scales[:, None] + v_zeros[:, None]
    return tl.where(d_mask[None, :], values, 0.0)


@triton.jit
def _tq_update_online_acc(acc, m_i, l_i, scores, values):
    m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp(m_i - m_ij)
    p = tl.exp(scores - m_ij[:, None])
    acc = acc * alpha[:, None] + tl.dot(
        p.to(tl.float16), values.to(tl.float16), input_precision="ieee"
    )
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return acc, l_i, m_ij


@triton.jit
def _tq_prefill_kernel(
    Q_rot_ptr,  # [Q, Hq, D] fp16
    Q_qjl_ptr,  # [Q, Hq, D] fp16
    Key_norm_lut_ptr,  # [256] float32
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [1, max_num_blocks] int32
    Centroids_ptr,  # [n_centroids] float32
    Out_ptr,  # [Q, Hq, D] fp16/bf16
    stride_qb,
    stride_qh,
    stride_qq_b,
    stride_qq_h,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_bt_n,
    stride_ob,
    stride_oh,
    Q_LEN,
    SEQ_LEN,
    CACHED_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KEY_NORM_BYTES: tl.constexpr,
    KEY_NORM_OFFSET: tl.constexpr,
    RES_NORM_OFFSET: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    QJL_ALPHA: tl.constexpr,
    RES_NORM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    hid = tl.program_id(0)
    q_block = tl.program_id(1)

    kv_head = hid // KV_GROUP_SIZE
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < Q_LEN
    d_mask = offs_d < HEAD_DIM
    q_pos = CACHED_LEN + offs_m

    q_base = offs_m[:, None] * stride_qb + hid * stride_qh
    q_rot = tl.load(
        Q_rot_ptr + q_base + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    qq_base = offs_m[:, None] * stride_qq_b + hid * stride_qq_h
    q_qjl = tl.load(
        Q_qjl_ptr + qq_base + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_block_end = tl.minimum(Q_LEN, (q_block + 1) * BLOCK_M)
    end_n = tl.minimum(SEQ_LEN, CACHED_LEN + q_block_end)

    start_n_limit = 0
    if SLIDING_WINDOW > 0:
        q_block_start = q_block * BLOCK_M
        start_n_limit = tl.maximum(0, CACHED_LEN + q_block_start - SLIDING_WINDOW + 1)

    for start_n in range(start_n_limit, end_n, BLOCK_N):
        kv_offs, kv_mask, slot_bases = _tq_load_slot_bases(
            Block_table_ptr,
            stride_bt_n,
            stride_cache_block,
            stride_cache_pos,
            stride_cache_head,
            start_n,
            offs_n,
            end_n,
            kv_head,
            BLOCK_SIZE,
        )

        scores = _tq_compute_scores_from_cache(
            q_rot,
            q_qjl,
            Key_norm_lut_ptr,
            KV_cache_ptr,
            Centroids_ptr,
            slot_bases,
            kv_mask,
            d_mask,
            offs_d,
            HEAD_DIM,
            MSE_BITS,
            MSE_BYTES,
            KEY_NORM_BYTES,
            KEY_NORM_OFFSET,
            RES_NORM_OFFSET,
            ATTN_SCALE,
            QJL_ALPHA,
            RES_NORM_SCALE,
        )
        scores = _tq_apply_prefill_mask(
            scores, q_mask, kv_mask, q_pos, kv_offs, SLIDING_WINDOW
        )

        values = _tq_load_values_from_cache(
            KV_cache_ptr,
            slot_bases,
            kv_mask,
            d_mask,
            offs_d,
            KPS,
            VQB,
            VAL_DATA_BYTES,
        )
        acc, l_i, m_i = _tq_update_online_acc(acc, m_i, l_i, scores, values)

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    out_base = offs_m[:, None] * stride_ob + hid * stride_oh
    tl.store(
        Out_ptr + out_base + offs_d[None, :],
        out,
        mask=q_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _tq_prefill_gqa_kernel(
    Q_rot_ptr,  # [Q, Hq, D] fp16
    Q_qjl_ptr,  # [Q, Hq, D] fp16
    Key_norm_lut_ptr,  # [256] float32
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [1, max_num_blocks] int32
    Centroids_ptr,  # [n_centroids] float32
    Out_ptr,  # [Q, Hq, D] fp16/bf16
    stride_qb,
    stride_qh,
    stride_qq_b,
    stride_qq_h,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_bt_n,
    stride_ob,
    stride_oh,
    Q_LEN,
    SEQ_LEN,
    CACHED_LEN,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    ROWS: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KEY_NORM_BYTES: tl.constexpr,
    KEY_NORM_OFFSET: tl.constexpr,
    RES_NORM_OFFSET: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    QJL_ALPHA: tl.constexpr,
    RES_NORM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    kv_head = tl.program_id(0)
    group_tile_id = tl.program_id(1)
    q_block = tl.program_id(2)

    offs_r = tl.arange(0, ROWS)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    group_row = offs_r // BLOCK_M
    token_row = offs_r - group_row * BLOCK_M
    group_offset = group_tile_id * GROUP_TILE + group_row
    q_head = kv_head * KV_GROUP_SIZE + group_offset
    offs_m = q_block * BLOCK_M + token_row

    q_mask = (
        (offs_m < Q_LEN)
        & (group_row < GROUP_TILE)
        & (q_head < NUM_Q_HEADS)
        & (group_offset < KV_GROUP_SIZE)
    )
    d_mask = offs_d < HEAD_DIM
    q_pos = CACHED_LEN + offs_m

    q_base = offs_m[:, None] * stride_qb + q_head[:, None] * stride_qh
    q_rot = tl.load(
        Q_rot_ptr + q_base + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    qq_base = offs_m[:, None] * stride_qq_b + q_head[:, None] * stride_qq_h
    q_qjl = tl.load(
        Q_qjl_ptr + qq_base + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    m_i = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([ROWS], dtype=tl.float32)
    acc = tl.zeros([ROWS, BLOCK_D], dtype=tl.float32)

    q_block_end = tl.minimum(Q_LEN, (q_block + 1) * BLOCK_M)
    end_n = tl.minimum(SEQ_LEN, CACHED_LEN + q_block_end)

    start_n_limit = 0
    if SLIDING_WINDOW > 0:
        q_block_start = q_block * BLOCK_M
        start_n_limit = tl.maximum(0, CACHED_LEN + q_block_start - SLIDING_WINDOW + 1)

    for start_n in range(start_n_limit, end_n, BLOCK_N):
        kv_offs, kv_mask, slot_bases = _tq_load_slot_bases(
            Block_table_ptr,
            stride_bt_n,
            stride_cache_block,
            stride_cache_pos,
            stride_cache_head,
            start_n,
            offs_n,
            end_n,
            kv_head,
            BLOCK_SIZE,
        )

        scores = _tq_compute_scores_from_cache(
            q_rot,
            q_qjl,
            Key_norm_lut_ptr,
            KV_cache_ptr,
            Centroids_ptr,
            slot_bases,
            kv_mask,
            d_mask,
            offs_d,
            HEAD_DIM,
            MSE_BITS,
            MSE_BYTES,
            KEY_NORM_BYTES,
            KEY_NORM_OFFSET,
            RES_NORM_OFFSET,
            ATTN_SCALE,
            QJL_ALPHA,
            RES_NORM_SCALE,
        )
        scores = _tq_apply_prefill_mask(
            scores, q_mask, kv_mask, q_pos, kv_offs, SLIDING_WINDOW
        )

        values = _tq_load_values_from_cache(
            KV_cache_ptr,
            slot_bases,
            kv_mask,
            d_mask,
            offs_d,
            KPS,
            VQB,
            VAL_DATA_BYTES,
        )
        acc, l_i, m_i = _tq_update_online_acc(acc, m_i, l_i, scores, values)

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    out_base = offs_m[:, None] * stride_ob + q_head[:, None] * stride_oh
    tl.store(
        Out_ptr + out_base + offs_d[None, :],
        out,
        mask=q_mask[:, None] & d_mask[None, :],
    )


def triton_turboquant_prefill_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    seq_len: int,
    cached_len: int,
    PiT: torch.Tensor,
    ST: torch.Tensor,
    key_norm_lut: torch.Tensor | None,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_norm_log_min: torch.Tensor | None = None,
    key_norm_log_max: torch.Tensor | None = None,
    max_q_projection_bytes: int = _MAX_Q_PROJECTION_BYTES,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """Run native TurboQuant prefill for one request.

    Returns ``None`` when the q-side projection buffers would exceed the safety
    budget. Callers should fall back to the stream/decode path in that case.
    """
    q_len, hq, d = query.shape
    # q-side projection uses temporaries:
    #   q_float fp32 when the input query is fp16/bf16, plus fp16 q_rot/q_qjl.
    # Keep this guard aligned with the actual allocations below so
    # single-shot long prefill falls back before transient OOM.
    q_float_bytes = 0 if query.dtype == torch.float32 else 4
    projected_bytes = q_len * hq * d * (q_float_bytes + 2 + 2)
    if projected_bytes > max_q_projection_bytes:
        logger.warning_once(
            "TurboQuant native prefill needs %.2f GiB for q-side projection "
            "buffers; "
            "falling back to stream prefill. Reduce prefill chunk size or "
            "increase the projection budget to use the native path.",
            projected_bytes / float(1 << 30),
        )
        return None

    hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = hq // hk
    cfg = _get_layout(d, mse_bits, value_quant_bits, key_packed_size)

    if cfg["key_norm_bytes"] == 1 and key_norm_lut is None:
        assert key_norm_log_min is not None
        assert key_norm_log_max is not None
        lut_idx = torch.arange(256, device=query.device, dtype=torch.float32)
        key_norm_lut = torch.exp2(
            key_norm_log_min.to(device=query.device, dtype=torch.float32)
            + lut_idx
            * (
                key_norm_log_max.to(device=query.device, dtype=torch.float32)
                - key_norm_log_min.to(device=query.device, dtype=torch.float32)
            )
            / 255.0
        ).contiguous()
    elif key_norm_lut is None:
        key_norm_lut = torch.empty(1, device=query.device, dtype=torch.float32)

    q_float = query if query.dtype == torch.float32 else query.float()
    q_rot = (q_float @ PiT).to(torch.float16).contiguous()
    q_qjl = (q_float @ ST).to(torch.float16).contiguous()
    output = torch.empty_like(query)

    group_tile = _select_gqa_group_tile(d, kv_group_size)
    if group_tile > 1 and d >= 512:
        block_m = 8
        block_n = 64
        num_warps = 8
    elif group_tile > 1:
        block_m = 8 if group_tile >= 3 else 16
        block_n = 16
        num_warps = 4
    else:
        block_m = 8 if d >= 512 else (16 if d >= 256 else 32)
        block_n = 32 if d <= 256 else 16
        num_warps = 8 if d >= 512 else 4
    if group_tile > 1:
        rows = triton.next_power_of_2(group_tile * block_m)
        grid = (hk, triton.cdiv(kv_group_size, group_tile), triton.cdiv(q_len, block_m))
        _tq_prefill_gqa_kernel[grid](
            q_rot,
            q_qjl,
            key_norm_lut,
            kv_cache,
            block_table,
            centroids,
            output,
            q_rot.stride(0),
            q_rot.stride(1),
            q_qjl.stride(0),
            q_qjl.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            block_table.stride(1),
            output.stride(0),
            output.stride(1),
            q_len,
            seq_len,
            cached_len,
            NUM_Q_HEADS=hq,
            HEAD_DIM=d,
            BLOCK_SIZE=block_size,
            KV_GROUP_SIZE=kv_group_size,
            GROUP_TILE=group_tile,
            ROWS=rows,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KEY_NORM_BYTES=cfg["key_norm_bytes"],
            KEY_NORM_OFFSET=cfg["key_norm_offset"],
            RES_NORM_OFFSET=cfg["res_norm_offset"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            QJL_ALPHA=_QJL_ALPHA,
            RES_NORM_SCALE=cfg["res_norm_scale"],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=cfg["BLOCK_D"],
            SLIDING_WINDOW=sliding_window or 0,
            num_warps=num_warps,
            num_stages=1,
        )
        return output

    grid = (hq, triton.cdiv(q_len, block_m))
    _tq_prefill_kernel[grid](
        q_rot,
        q_qjl,
        key_norm_lut,
        kv_cache,
        block_table,
        centroids,
        output,
        q_rot.stride(0),
        q_rot.stride(1),
        q_qjl.stride(0),
        q_qjl.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        q_len,
        seq_len,
        cached_len,
        HEAD_DIM=d,
        BLOCK_SIZE=block_size,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KEY_NORM_BYTES=cfg["key_norm_bytes"],
        KEY_NORM_OFFSET=cfg["key_norm_offset"],
        RES_NORM_OFFSET=cfg["res_norm_offset"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        QJL_ALPHA=_QJL_ALPHA,
        RES_NORM_SCALE=cfg["res_norm_scale"],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=cfg["BLOCK_D"],
        SLIDING_WINDOW=sliding_window or 0,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
