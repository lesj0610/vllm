# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention for serving-tuned `_nc` presets."""

import math
from typing import Any

import torch

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

# The paper-scale QJL correction is unbiased in expectation, but in serving
# workloads the added variance is amplified by the attention softmax. A damped
# correction preserves the residual signal while materially improving quality
# for the current `_nc` presets.
_QJL_ALPHA = 0.25 * math.sqrt(math.pi / 2.0)


@triton.jit
def _tq_decode_stage1(
    Q_rot_ptr,  # [B, Hq, D] float32
    Q_qjl_ptr,  # [B, Hq, D] float32
    Key_norm_lut_ptr,  # [256] float32
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    Centroids_ptr,  # [n_centroids] float32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    stride_qb,
    stride_qh,
    stride_qq_b,
    stride_qq_h,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    KEY_NORM_BYTES: tl.constexpr,
    KEY_NORM_OFFSET: tl.constexpr,
    RES_NORM_OFFSET: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    QJL_ALPHA: tl.constexpr,
    RES_NORM_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)
    if SLIDING_WINDOW > 0:
        effective_seq_len = tl.minimum(seq_len, SLIDING_WINDOW)
        window_start = seq_len - effective_seq_len
    else:
        effective_seq_len = seq_len
        window_start = 0

    split_len = tl.cdiv(effective_seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, effective_seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    qq_base = bid * stride_qq_b + hid * stride_qq_h
    q_qjl = tl.load(Q_qjl_ptr + qq_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    mse_bit_off = d_offs * MSE_BITS
    mse_byte_idx = mse_bit_off // 8
    mse_bit_shift = mse_bit_off % 8
    mse_mask = (1 << MSE_BITS) - 1

    qjl_byte_idx = d_offs // 8
    qjl_bit_shift = d_offs % 8

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        global_kv_offs = kv_offs + window_start

        page_idx = global_kv_offs // BLOCK_SIZE
        page_off = global_kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        )
        slot_bases = (
            block_nums * stride_cache_block
            + page_off * stride_cache_pos
            + kv_head * stride_cache_head
        )

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
        )
        term_base = tl.sum(
            tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
            axis=1,
        )

        qjl_addrs = slot_bases[:, None] + MSE_BYTES + qjl_byte_idx[None, :]
        qjl_raw = tl.load(
            KV_cache_ptr + qjl_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        qjl_bit = (qjl_raw >> qjl_bit_shift[None, :]) & 0x1
        qjl_sign = qjl_bit.to(tl.float32) * 2.0 - 1.0
        term_qjl = tl.sum(
            tl.where(d_mask[None, :], q_qjl[None, :] * qjl_sign, 0.0),
            axis=1,
        )

        kn_base = slot_bases + KEY_NORM_OFFSET
        if KEY_NORM_BYTES == 1:
            kn_raw = tl.load(KV_cache_ptr + kn_base, mask=kv_mask, other=0).to(
                tl.int32
            )
            key_norm = tl.load(Key_norm_lut_ptr + kn_raw, mask=kv_mask, other=0.0)
        else:
            kn_lo = tl.load(KV_cache_ptr + kn_base, mask=kv_mask, other=0).to(
                tl.uint16
            )
            kn_hi = tl.load(KV_cache_ptr + kn_base + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            key_norm = (kn_lo | (kn_hi << 8)).to(tl.float16, bitcast=True).to(
                tl.float32
            )

        rn_base = slot_bases + RES_NORM_OFFSET
        rn_raw = tl.load(KV_cache_ptr + rn_base, mask=kv_mask, other=0).to(
            tl.float32
        )
        residual_norm = rn_raw * (RES_NORM_SCALE / 255.0)

        qjl_corr = (QJL_ALPHA / HEAD_DIM) * residual_norm * term_qjl
        scores = key_norm * (term_base + qjl_corr) * ATTN_SCALE
        scores = tl.where(kv_mask, scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        val_bases = slot_bases + KPS
        if VQB == 3:
            val_bit_off = d_offs * 3
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
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

        sc_bases = val_bases + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(
            tl.float32
        )
        zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
            tl.uint16
        )
        zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(
            tl.float32
        )
        values = v_idx * v_scales[:, None] + v_zeros[:, None]

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


_layout_cache: dict = {}


def _get_layout(d: int, mse_bits: int, value_quant_bits: int, key_packed_size: int):
    key = (d, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        qjl_bytes = math.ceil(d / 8)
        val_data_bytes = math.ceil(d * value_quant_bits / 8)
        key_norm_bytes = key_packed_size - math.ceil(d * mse_bits / 8) - qjl_bytes - 1
        if key_norm_bytes not in (1, 2):
            raise ValueError(
                "Unexpected TurboQuant key-norm payload width: "
                f"key_norm_bytes={key_norm_bytes}"
            )
        cfg = {
            "mse_bytes": math.ceil(d * mse_bits / 8),
            "qjl_bytes": qjl_bytes,
            "key_norm_bytes": key_norm_bytes,
            "key_norm_offset": math.ceil(d * mse_bits / 8) + qjl_bytes,
            "res_norm_offset": math.ceil(d * mse_bits / 8) + qjl_bytes + key_norm_bytes,
            "val_data_bytes": val_data_bytes,
            "res_norm_scale": TurboQuantConfig.get_residual_norm_quant_max(
                mse_bits,
                value_quant_bits,
            ),
            "BLOCK_D": triton.next_power_of_2(d),
        }
        _layout_cache[key] = cfg
    return cfg


def full_dequant_turboquant_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    ST: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_norm_log_min: torch.Tensor | None = None,
    key_norm_log_max: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dequantizer for tests and numerical validation."""
    if S is None:
        S = ST.T.contiguous()
    bsz = block_table.shape[0]
    hk = kv_cache.shape[2]
    d = Pi.shape[0]
    cfg = _get_layout(d, mse_bits, value_quant_bits, key_packed_size)
    max_seq = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    k_out = torch.zeros(
        bsz, hk, max_seq, d, device=kv_cache.device, dtype=torch.float32
    )
    v_out = torch.zeros_like(k_out)

    for b in range(bsz):
        seq_len = int(seq_lens[b].item())
        for pos in range(seq_len):
            page_idx = pos // kv_cache.shape[1]
            page_off = pos % kv_cache.shape[1]
            block_num = int(block_table[b, page_idx].item())
            slot = kv_cache[block_num, page_off]  # [Hk, padded_slot]

            mse_bytes = slot[:, :cfg["mse_bytes"]]
            qjl_bytes = slot[
                :, cfg["mse_bytes"] : cfg["mse_bytes"] + cfg["qjl_bytes"]
            ]
            if cfg["key_norm_bytes"] == 1:
                assert key_norm_log_min is not None
                assert key_norm_log_max is not None
                key_norm_q = slot[:, cfg["key_norm_offset"]].float()
                key_norm_log = key_norm_log_min + (
                    key_norm_q * (key_norm_log_max - key_norm_log_min) / 255.0
                )
                key_norm = torch.exp2(key_norm_log)
            else:
                key_norm = (
                    slot[:, cfg["key_norm_offset"] : cfg["key_norm_offset"] + 2]
                    .contiguous()
                    .view(torch.float16)
                    .float()
                    .squeeze(-1)
                )
            residual_norm = (
                slot[:, cfg["res_norm_offset"]]
                .float()
                .mul(cfg["res_norm_scale"] / 255.0)
            )

            if mse_bits == 3:
                packed24 = (
                    mse_bytes[:, 0::3].to(torch.int32)
                    | (mse_bytes[:, 1::3].to(torch.int32) << 8)
                    | (mse_bytes[:, 2::3].to(torch.int32) << 16)
                )
                shifts = (
                    torch.arange(8, device=kv_cache.device, dtype=torch.int32) * 3
                )
                idx = ((packed24.unsqueeze(-1) >> shifts) & 0x7).reshape(hk, d)
            elif mse_bits == 2:
                shifts = (
                    torch.arange(4, device=kv_cache.device, dtype=torch.int32) * 2
                )
                idx = (
                    (mse_bytes.to(torch.int32).unsqueeze(-1) >> shifts) & 0x3
                ).reshape(hk, d)
            else:
                raise ValueError(f"Unsupported mse_bits={mse_bits}")

            y_tilde = centroids[idx]
            xhat_mse = y_tilde @ Pi
            qjl_shifts = torch.arange(8, device=kv_cache.device, dtype=torch.int32)
            qjl_bits = (
                (qjl_bytes.to(torch.int32).unsqueeze(-1) >> qjl_shifts) & 0x1
            ).reshape(hk, d)
            qjl_sign = qjl_bits.float() * 2.0 - 1.0
            # Materialized K reconstruction is S^T @ sign. With row-vector
            # PyTorch notation that is qjl_sign @ S. Decode still projects the
            # query as query @ ST and computes the equivalent dot product.
            xhat_qjl = (_QJL_ALPHA / d) * residual_norm[:, None] * (qjl_sign @ S)
            k_out[b, :, pos] = key_norm[:, None] * (xhat_mse + xhat_qjl)

            val_base = key_packed_size
            val_bytes = slot[:, val_base:val_base + cfg["val_data_bytes"]]
            scale = (
                slot[
                    :,
                    val_base + cfg["val_data_bytes"] : val_base
                    + cfg["val_data_bytes"]
                    + 2,
                ]
                .contiguous()
                .view(torch.float16)
                .float()
                .squeeze(-1)
            )
            zero = (
                slot[
                    :,
                    val_base
                    + cfg["val_data_bytes"]
                    + 2 : val_base
                    + cfg["val_data_bytes"]
                    + 4,
                ]
                .contiguous()
                .view(torch.float16)
                .float()
                .squeeze(-1)
            )
            if value_quant_bits == 4:
                shifts = torch.tensor(
                    [0, 4], device=kv_cache.device, dtype=torch.int32
                )
                v_idx = (
                    (val_bytes.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
                ).reshape(hk, d)
            elif value_quant_bits == 3:
                packed24 = (
                    val_bytes[:, 0::3].to(torch.int32)
                    | (val_bytes[:, 1::3].to(torch.int32) << 8)
                    | (val_bytes[:, 2::3].to(torch.int32) << 16)
                )
                shifts = (
                    torch.arange(8, device=kv_cache.device, dtype=torch.int32) * 3
                )
                v_idx = ((packed24.unsqueeze(-1) >> shifts) & 0x7).reshape(hk, d)
            else:
                raise ValueError(f"Unsupported value_quant_bits={value_quant_bits}")
            v_out[b, :, pos] = v_idx.float() * scale[:, None] + zero[:, None]

    return k_out, v_out


def _decode_turboquant_slots(
    slots: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    ST: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_norm_log_min: torch.Tensor | None = None,
    key_norm_log_max: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized PyTorch dequant for TurboQuant slots.

    Args:
        slots: uint8 tensor with shape ``[..., Hk, slot_size]``.

    Returns:
        ``(k, v)`` with shape ``[..., Hk, D]`` in float32.
    """
    del ST  # ST is for decode query projection; materialized K uses S.
    d = Pi.shape[0]
    cfg = _get_layout(d, mse_bits, value_quant_bits, key_packed_size)
    hk = slots.shape[-2]
    device = slots.device

    mse_bytes = slots[..., : cfg["mse_bytes"]]
    if mse_bits == 3:
        packed24 = (
            mse_bytes[..., 0::3].to(torch.int32)
            | (mse_bytes[..., 1::3].to(torch.int32) << 8)
            | (mse_bytes[..., 2::3].to(torch.int32) << 16)
        )
        shifts = torch.arange(8, device=device, dtype=torch.int32) * 3
        idx = ((packed24.unsqueeze(-1) >> shifts) & 0x7).reshape(
            *slots.shape[:-2], hk, d
        )
    elif mse_bits == 2:
        shifts = torch.arange(4, device=device, dtype=torch.int32) * 2
        idx = ((mse_bytes.to(torch.int32).unsqueeze(-1) >> shifts) & 0x3).reshape(
            *slots.shape[:-2], hk, d
        )
    else:
        raise ValueError(f"Unsupported mse_bits={mse_bits}")

    y_tilde = centroids[idx]
    xhat_mse = y_tilde @ Pi

    qjl_bytes = slots[..., cfg["mse_bytes"] : cfg["mse_bytes"] + cfg["qjl_bytes"]]
    qjl_shifts = torch.arange(8, device=device, dtype=torch.int32)
    qjl_bits = (
        (qjl_bytes.to(torch.int32).unsqueeze(-1) >> qjl_shifts) & 0x1
    ).reshape(*slots.shape[:-2], hk, d)
    qjl_sign = qjl_bits.float() * 2.0 - 1.0

    if cfg["key_norm_bytes"] == 1:
        assert key_norm_log_min is not None
        assert key_norm_log_max is not None
        key_norm_q = slots[..., cfg["key_norm_offset"]].float()
        key_norm_log = key_norm_log_min + (
            key_norm_q * (key_norm_log_max - key_norm_log_min) / 255.0
        )
        key_norm = torch.exp2(key_norm_log)
    else:
        key_norm = (
            slots[..., cfg["key_norm_offset"] : cfg["key_norm_offset"] + 2]
            .contiguous()
            .view(torch.float16)
            .float()
            .squeeze(-1)
        )

    residual_norm = (
        slots[..., cfg["res_norm_offset"]].float().mul(cfg["res_norm_scale"] / 255.0)
    )
    xhat_qjl = (_QJL_ALPHA / d) * residual_norm[..., None] * (qjl_sign @ S)
    k_out = key_norm[..., None] * (xhat_mse + xhat_qjl)

    val_base = key_packed_size
    val_bytes = slots[..., val_base : val_base + cfg["val_data_bytes"]]
    scale_start = val_base + cfg["val_data_bytes"]
    scale = (
        slots[..., scale_start : scale_start + 2]
        .contiguous()
        .view(torch.float16)
        .float()
        .squeeze(-1)
    )
    zero = (
        slots[..., scale_start + 2 : scale_start + 4]
        .contiguous()
        .view(torch.float16)
        .float()
        .squeeze(-1)
    )
    if value_quant_bits == 4:
        shifts = torch.tensor([0, 4], device=device, dtype=torch.int32)
        v_idx = ((val_bytes.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF).reshape(
            *slots.shape[:-2], hk, d
        )
    elif value_quant_bits == 3:
        packed24 = (
            val_bytes[..., 0::3].to(torch.int32)
            | (val_bytes[..., 1::3].to(torch.int32) << 8)
            | (val_bytes[..., 2::3].to(torch.int32) << 16)
        )
        shifts = torch.arange(8, device=device, dtype=torch.int32) * 3
        v_idx = ((packed24.unsqueeze(-1) >> shifts) & 0x7).reshape(
            *slots.shape[:-2], hk, d
        )
    else:
        raise ValueError(f"Unsupported value_quant_bits={value_quant_bits}")
    v_out = v_idx.float() * scale[..., None] + zero[..., None]
    return k_out, v_out


def dequant_turboquant_cache_pages(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    Pi: torch.Tensor,
    S: torch.Tensor,
    ST: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_norm_log_min: torch.Tensor | None = None,
    key_norm_log_max: torch.Tensor | None = None,
    page_chunk_size: int = 1,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize the used TurboQuant pages into compact FA-compatible pages.

    The returned block table maps the original used physical blocks to compact
    contiguous page IDs, so FlashAttention can read from the transient K/V
    tensors without touching the compressed cache.
    """
    if block_table.shape[0] != 1:
        raise ValueError("TurboQuant transient FA dequant currently handles one req")
    block_size = kv_cache.shape[1]
    num_pages = math.ceil(seq_len / block_size)
    if num_pages <= 0:
        raise ValueError("Cannot dequantize empty TurboQuant page set")

    device = kv_cache.device
    dtype = output_dtype or torch.float16
    physical_blocks = block_table[0, :num_pages].to(device=device, dtype=torch.long)
    hk = kv_cache.shape[2]
    d = Pi.shape[0]
    k_pages = torch.empty(
        num_pages, block_size, hk, d, device=device, dtype=dtype
    )
    v_pages = torch.empty_like(k_pages)

    page_chunk_size = max(1, page_chunk_size)
    for start in range(0, num_pages, page_chunk_size):
        end = min(start + page_chunk_size, num_pages)
        slots = kv_cache.index_select(0, physical_blocks[start:end])
        k_chunk, v_chunk = _decode_turboquant_slots(
            slots=slots,
            Pi=Pi,
            S=S,
            ST=ST,
            centroids=centroids,
            mse_bits=mse_bits,
            key_packed_size=key_packed_size,
            value_quant_bits=value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
        )
        k_pages[start:end] = k_chunk.to(dtype)
        v_pages[start:end] = v_chunk.to(dtype)

    compact_block_table = torch.arange(
        num_pages, device=device, dtype=block_table.dtype
    ).view(1, num_pages)
    return k_pages, v_pages, compact_block_table


def triton_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    ST: torch.Tensor,
    key_norm_lut: torch.Tensor | None,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_norm_log_min: torch.Tensor | None = None,
    key_norm_log_max: torch.Tensor | None = None,
    PiT: torch.Tensor | None = None,
    query_rot: torch.Tensor | None = None,
    query_qjl: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sliding_window: int | None = None,
) -> torch.Tensor:
    bsz, hq, d = query.shape
    hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = hq // hk
    device = query.device
    cfg = _get_layout(d, mse_bits, value_quant_bits, key_packed_size)
    if cfg["key_norm_bytes"] == 1 and key_norm_lut is None:
        assert key_norm_log_min is not None
        assert key_norm_log_max is not None
        lut_idx = torch.arange(256, device=device, dtype=torch.float32)
        key_norm_lut = torch.exp2(
            key_norm_log_min.to(device=device, dtype=torch.float32)
            + lut_idx
            * (
                key_norm_log_max.to(device=device, dtype=torch.float32)
                - key_norm_log_min.to(device=device, dtype=torch.float32)
            )
            / 255.0
        ).contiguous()
    elif key_norm_lut is None:
        key_norm_lut = torch.empty(1, device=device, dtype=torch.float32)

    q_rot = query_rot
    if q_rot is None:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()
    q_qjl = query_qjl
    if q_qjl is None:
        q_qjl = (query.float() @ ST).contiguous()

    num_kv_splits = max_num_kv_splits
    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= bsz
        and mid_o_buf.shape[2] >= num_kv_splits
    ):
        mid_o = mid_o_buf[:bsz, :hq, :num_kv_splits, :]
    else:
        mid_o = torch.empty(
            bsz, hq, num_kv_splits, d + 1, dtype=torch.float32, device=device
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    grid = (bsz, hq, num_kv_splits)
    _tq_decode_stage1[grid](
        q_rot,
        q_qjl,
        key_norm_lut,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        q_qjl.stride(0),
        q_qjl.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=hk,
        HEAD_DIM=d,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=num_kv_splits,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        QJL_BYTES=cfg["qjl_bytes"],
        KEY_NORM_BYTES=cfg["key_norm_bytes"],
        KEY_NORM_OFFSET=cfg["key_norm_offset"],
        RES_NORM_OFFSET=cfg["res_norm_offset"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        QJL_ALPHA=_QJL_ALPHA,
        RES_NORM_SCALE=cfg["res_norm_scale"],
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=4,
        SLIDING_WINDOW=sliding_window or 0,
        num_warps=1,
        num_stages=1,
    )

    if output_buf is not None and output_buf.shape[0] >= bsz:
        output = output_buf[:bsz, :hq, :d]
    else:
        output = torch.empty(bsz, hq, d, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= bsz:
        lse = lse_buf[:bsz, :hq]
    else:
        lse = torch.empty(bsz, hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (bsz, hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=d,
        num_warps=4,
        num_stages=1,
    )
    return output.to(query.dtype)
