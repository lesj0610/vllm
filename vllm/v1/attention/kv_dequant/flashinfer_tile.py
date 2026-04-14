# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer staged KV dequant helpers.

These helpers use eager PyTorch tensor ops. They follow the same inline scale
layout contract as the Triton path, but they are callable from the Python-side
FlashInfer forward path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import INT4_CODEBOOK_LEVELS
from vllm.v1.kv_cache_interface import (
    INT4_CHANNELS_PER_SCALE,
    KVQuantMode,
    get_per_token_head_scale_count,
    get_per_token_head_scale_dtype,
)

SUPPORTED_MODES: frozenset[KVQuantMode] = frozenset(
    {
        KVQuantMode.NONE,
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.INT4_PER_TOKEN_HEAD,
    }
)

_INT4_CODEBOOK_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _get_int4_codebook(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    codebook = _INT4_CODEBOOK_CACHE.get(key)
    if codebook is None:
        codebook = torch.tensor(
            INT4_CODEBOOK_LEVELS, dtype=torch.float32, device=device
        )
        _INT4_CODEBOOK_CACHE[key] = codebook
    return codebook


def _get_inline_scale_view(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    kv_quant_mode: KVQuantMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the inline payload slice and typed scale view."""
    scale_dtype = get_per_token_head_scale_dtype(kv_quant_mode)
    assert scale_dtype is not None

    dtype_sz = kv_cache.element_size()
    scale_dtype_sz = get_dtype_size(scale_dtype)
    scale_count = get_per_token_head_scale_count(head_size, kv_quant_mode)
    scale_pad = scale_count * scale_dtype_sz // dtype_sz
    data_head_size = kv_cache.shape[-1] - scale_pad

    raw = kv_cache.untyped_storage()
    scale_storage = torch.tensor([], dtype=scale_dtype, device=kv_cache.device).set_(
        raw
    )

    strides = kv_cache.stride()
    stride_names: dict[str, int] = {
        f"dim_{idx}": stride for idx, stride in enumerate(strides[:-1])
    }
    stride_names["scale_offset"] = data_head_size * strides[-1]
    for name, stride in stride_names.items():
        byte_stride = stride * dtype_sz
        assert byte_stride % scale_dtype_sz == 0, (
            f"{name} byte stride must align with {scale_dtype}, got {byte_stride} bytes"
        )

    scale_stride: Sequence[int] = tuple(
        stride * dtype_sz // scale_dtype_sz for stride in strides[:-1]
    ) + (1,)
    scale_offset = data_head_size * strides[-1] * dtype_sz // scale_dtype_sz
    scales = torch.as_strided(
        scale_storage,
        size=kv_cache.shape[:-1] + (scale_count,),
        stride=tuple(scale_stride),
        storage_offset=scale_offset,
    )
    return kv_cache[..., :data_head_size], scales


def _staged_dequant_int8_kv_cache(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    payload, scales = _get_inline_scale_view(
        kv_cache,
        head_size=head_size,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    assert payload.shape[-1] == head_size, (
        f"Expected logical INT8 head size {head_size}, got {payload.shape[-1]}"
    )
    return (payload.to(torch.float32) * scales.to(torch.float32)).to(out_dtype)


def _staged_dequant_int4_kv_cache(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    payload, scales = _get_inline_scale_view(
        kv_cache,
        head_size=head_size,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    )
    low = (payload & 0x0F).to(torch.int32)
    high = ((payload >> 4) & 0x0F).to(torch.int32)
    packed = torch.stack((low, high), dim=-1).reshape(*payload.shape[:-1], -1)[
        ..., :head_size
    ]
    levels = _get_int4_codebook(kv_cache.device)[packed]
    group_idx = (
        torch.arange(head_size, device=kv_cache.device) // INT4_CHANNELS_PER_SCALE
    ).view(*([1] * (levels.dim() - 1)), head_size)
    group_scales = torch.take_along_dim(
        scales.to(torch.float32),
        group_idx.expand(*levels.shape[:-1], head_size),
        dim=-1,
    )
    return (levels * group_scales).to(out_dtype)


def staged_dequantize_paged_kv_cache(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    kv_quant_mode: KVQuantMode,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize paged KV cache into a FlashInfer-compatible dense tensor."""
    if kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
        return _staged_dequant_int8_kv_cache(
            kv_cache, head_size=head_size, out_dtype=out_dtype
        )
    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        return _staged_dequant_int4_kv_cache(
            kv_cache, head_size=head_size, out_dtype=out_dtype
        )
    raise ValueError(
        f"staged dequant is not defined for kv quant mode {kv_quant_mode.name}"
    )
