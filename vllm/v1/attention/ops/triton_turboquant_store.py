# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant store path.

This file keeps the public launcher name used by the attention backend, but the
paper-faithful TurboQuant_prod key path is implemented with vectorized PyTorch
ops for quantization/packing and a final scatter into the combined KV cache.
"""


import torch

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)


def _pack_levels(levels: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack per-coordinate integer levels into uint8 bytes."""
    levels = levels.to(torch.int32).contiguous()
    n, d = levels.shape
    if bits == 4:
        assert d % 2 == 0
        pairs = levels.view(n, d // 2, 2)
        packed = (pairs[..., 0] & 0xF) | ((pairs[..., 1] & 0xF) << 4)
        return packed.to(torch.uint8)
    if bits == 3:
        assert d % 8 == 0
        groups = levels.view(n, d // 8, 8)
        shifts = (torch.arange(8, device=levels.device, dtype=torch.int32) * 3)
        packed24 = ((groups & 0x7) << shifts).sum(dim=-1)
        out = torch.empty(n, d // 8 * 3, device=levels.device, dtype=torch.uint8)
        out[:, 0::3] = (packed24 & 0xFF).to(torch.uint8)
        out[:, 1::3] = ((packed24 >> 8) & 0xFF).to(torch.uint8)
        out[:, 2::3] = ((packed24 >> 16) & 0xFF).to(torch.uint8)
        return out
    if bits == 2:
        assert d % 4 == 0
        groups = levels.view(n, d // 4, 4)
        shifts = (torch.arange(4, device=levels.device, dtype=torch.int32) * 2)
        packed = ((groups & 0x3) << shifts).sum(dim=-1)
        return packed.to(torch.uint8)
    raise ValueError(f"Unsupported pack bits: {bits}")


def _pack_sign_bits(sign_mask: torch.Tensor) -> torch.Tensor:
    sign_mask = sign_mask.to(torch.int32).contiguous()
    n, d = sign_mask.shape
    assert d % 8 == 0
    groups = sign_mask.view(n, d // 8, 8)
    shifts = torch.arange(8, device=sign_mask.device, dtype=torch.int32)
    packed = ((groups & 0x1) << shifts).sum(dim=-1)
    return packed.to(torch.uint8)


def _pack_fp16_scalars(values: torch.Tensor) -> torch.Tensor:
    return (
        values.to(torch.float16)
        .contiguous()
        .view(torch.uint8)
        .view(values.shape[0], 2)
    )


def _pack_uint8_scalars(values: torch.Tensor, *, scale_max: float) -> torch.Tensor:
    q = torch.clamp(
        torch.round(values.float() * (255.0 / scale_max)),
        0,
        255,
    )
    return q.to(torch.uint8).view(values.shape[0], 1)


def _pack_log_uint8_scalars(
    values: torch.Tensor,
    *,
    log_min: torch.Tensor,
    log_max: torch.Tensor,
) -> torch.Tensor:
    values = values.float().clamp_min(1e-8)
    log_values = torch.log2(values)
    step = (log_max - log_min) / 255.0
    q = torch.clamp(
        torch.round((log_values - log_min) / step),
        0,
        255,
    )
    return q.to(torch.uint8).view(values.shape[0], 1)


def _quantize_and_pack_values(
    value: torch.Tensor, value_quant_bits: int
) -> torch.Tensor:
    value = value.float().contiguous()
    n, d = value.shape
    v_min = value.min(dim=1).values
    v_max = value.max(dim=1).values
    denom = (2**value_quant_bits) - 1
    scale = (v_max - v_min) / max(denom, 1)
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round((value - v_min[:, None]) / scale[:, None]), 0, denom)
    packed = _pack_levels(q.to(torch.int32), value_quant_bits)
    return torch.cat([
        packed,
        _pack_fp16_scalars(scale),
        _pack_fp16_scalars(v_min),
    ], dim=1)


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    Pi: torch.Tensor,  # [D, D] float32
    PiT: torch.Tensor,  # [D, D] float32
    ST: torch.Tensor,  # [D, D] float32 (S^T)
    centroids: torch.Tensor,  # [2^mse_bits] float32
    midpoints: torch.Tensor,  # [2^mse_bits-1] float32
    key_norm_log_min: torch.Tensor,  # [] float32
    key_norm_log_max: torch.Tensor,  # [] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
):
    """Quantize and scatter K/V into the combined TurboQuant cache."""
    n, h, d = key.shape
    if n == 0:
        return

    valid = slot_mapping >= 0
    if not torch.all(valid):
        key = key[valid]
        value = value[valid]
        slot_mapping = slot_mapping[valid]
        n = int(valid.sum().item())
        if n == 0:
            return

    nh = n * h
    key_f = key.float().reshape(nh, d)
    value_f = value.float().reshape(nh, d)

    key_norm = key_f.norm(dim=1)
    x_hat = key_f / key_norm.clamp_min(1e-8).unsqueeze(1)

    y = x_hat @ PiT
    idx = torch.bucketize(y, midpoints)
    y_tilde = centroids[idx]
    xhat_mse = y_tilde @ Pi
    residual = x_hat - xhat_mse
    residual_norm = residual.norm(dim=1)
    qjl = (residual @ ST) >= 0
    residual_norm_scale = TurboQuantConfig.get_residual_norm_quant_max(
        mse_bits,
        value_quant_bits,
    )

    mse_bytes = _pack_levels(idx, mse_bits)
    qjl_bytes = _pack_sign_bits(qjl)
    key_bytes = torch.cat(
        [
            mse_bytes,
            qjl_bytes,
            _pack_log_uint8_scalars(
                key_norm,
                log_min=key_norm_log_min.to(
                    device=key_norm.device, dtype=torch.float32
                ),
                log_max=key_norm_log_max.to(
                    device=key_norm.device, dtype=torch.float32
                ),
            ),
            _pack_uint8_scalars(
                residual_norm,
                scale_max=residual_norm_scale,
            ),
        ],
        dim=1,
    )
    assert key_bytes.shape[1] == key_packed_size, (
        key_bytes.shape[1],
        key_packed_size,
    )

    value_bytes = _quantize_and_pack_values(value_f, value_quant_bits)
    slot_bytes = torch.cat([key_bytes, value_bytes], dim=1).view(n, h, -1)

    block_size = kv_cache.shape[1]
    blocks = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offs = slot_mapping % block_size
    head_idx = torch.arange(h, device=kv_cache.device)
    kv_cache[
        blocks[:, None], offs[:, None], head_idx[None, :], : slot_bytes.shape[-1]
    ] = slot_bytes
