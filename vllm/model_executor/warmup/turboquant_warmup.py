# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up TurboQuant attention kernels before serving requests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.worker.workspace import is_workspace_manager_initialized

logger = init_logger(__name__)


@dataclass(frozen=True)
class _TurboQuantWarmupKey:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    block_table_stride: int
    num_kv_splits: int
    kv_group_size: int
    scale: float
    mse_bits: int
    key_packed_size: int
    value_quant_bits: int
    key_fp8: bool
    norm_correction: bool
    output_fp16: bool
    sliding_window: int | None


def _iter_turboquant_attention_layers(
    model: torch.nn.Module,
) -> Iterable[tuple[Attention, TurboQuantAttentionImpl]]:
    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        if not isinstance(layer.kv_cache_dtype, str):
            continue
        if not layer.kv_cache_dtype.startswith("turboquant_"):
            continue
        if not isinstance(layer.impl, TurboQuantAttentionImpl):
            continue
        yield layer, layer.impl


def _make_warmup_key(
    impl: TurboQuantAttentionImpl,
    *,
    block_size: int,
    block_table_stride: int,
    model_dtype: torch.dtype,
) -> _TurboQuantWarmupKey:
    return _TurboQuantWarmupKey(
        num_heads=impl.num_heads,
        num_kv_heads=impl.num_kv_heads,
        head_dim=impl.head_size,
        block_size=block_size,
        # Triton specializes scalar stride arguments unless told otherwise.
        # Keep synthetic block-table layout aligned with runtime and dedupe on
        # it so warmup covers the same launch family.
        block_table_stride=block_table_stride,
        num_kv_splits=impl.max_num_kv_splits,
        kv_group_size=impl.num_kv_groups,
        scale=impl.scale,
        mse_bits=impl.tq_config.key_mse_bits,
        key_packed_size=impl.tq_config.key_packed_size,
        value_quant_bits=impl.tq_config.effective_value_quant_bits,
        key_fp8=impl.tq_config.key_fp8,
        norm_correction=impl.tq_config.norm_correction,
        output_fp16=model_dtype == torch.float16,
        sliding_window=impl.sliding_window,
    )


def _make_decode_metadata(
    *,
    batch_size: int,
    device: torch.device,
    block_table_stride: int,
) -> TurboQuantMetadata:
    block_table = torch.zeros(
        (batch_size, block_table_stride), dtype=torch.int32, device=device
    )
    # Block 0 is the null block. Use block 1 so the kernel takes the normal
    # cache-read path without touching runtime block-pool state.
    block_table[:, 0] = 1
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    return TurboQuantMetadata(
        seq_lens=seq_lens,
        slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
        block_table=block_table,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=1,
        is_prefill=False,
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
    )


def _warmup_turboquant_layer(
    layer: Attention,
    impl: TurboQuantAttentionImpl,
    *,
    device: torch.device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype: torch.dtype,
) -> None:
    impl._ensure_on_device(layer, device)

    batch_size = max_num_decode_tokens
    query = torch.zeros(
        (batch_size, impl.num_heads, impl.head_size),
        dtype=model_dtype,
        device=device,
    )
    kv_cache = torch.zeros(
        (
            2,
            block_size,
            impl.num_kv_heads,
            impl.tq_config.slot_size_aligned,
        ),
        dtype=torch.uint8,
        device=device,
    )
    attn_metadata = _make_decode_metadata(
        batch_size=batch_size,
        device=device,
        block_table_stride=block_table_stride,
    )

    # Runtime decode path: warms _tq_decode_stage1/_tq_decode_stage2 and
    # reserves any decode workspace before CUDA graph capture locks growth.
    impl._decode_attention(
        query=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        Pi=layer._tq_Pi,
        centroids=layer._tq_centroids,
        PiT=layer._tq_PiT,
        layer=layer,
    )

    # Continuation/full-dequant path: runtime logs showed _tq_full_dequant_kv
    # can still compile on first use if only decode is warmed. This path uses
    # WorkspaceManager; skip only in unit tests or unusual init states where the
    # workspace manager is not available yet.
    if is_workspace_manager_initialized():
        impl._dequant_cached_kv(
            layer,
            kv_cache,
            attn_metadata.block_table[:1],
            cache_len=block_size,
            output_dtype=model_dtype,
        )


@torch.inference_mode()
def turboquant_attention_warmup(
    model: torch.nn.Module,
    *,
    device: torch.device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype: torch.dtype,
) -> None:
    """Compile TurboQuant decode/full-dequant kernels before real traffic.

    V1 dummy/profile runs may avoid TurboQuant decode and continuation
    dequant paths, which leaves Triton kernels to compile on the first real
    request under the JIT monitor. This warmup calls the backend helpers with
    synthetic tensors whose launch-time constants match runtime attention.
    """
    if max_num_decode_tokens <= 0:
        return
    if block_table_stride <= 0:
        return

    seen: set[_TurboQuantWarmupKey] = set()
    num_warmups = 0

    for layer, impl in _iter_turboquant_attention_layers(model):
        key = _make_warmup_key(
            impl,
            block_size=block_size,
            block_table_stride=block_table_stride,
            model_dtype=model_dtype,
        )
        if key in seen:
            continue
        seen.add(key)
        _warmup_turboquant_layer(
            layer,
            impl,
            device=device,
            block_size=block_size,
            block_table_stride=block_table_stride,
            max_num_decode_tokens=max_num_decode_tokens,
            model_dtype=model_dtype,
        )
        num_warmups += 1

    if num_warmups > 0:
        torch.accelerator.synchronize()
        logger.info("Warmed up %d TurboQuant attention variant(s).", num_warmups)
