# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper-faithful TurboQuant attention backend for vLLM.

Prefill / continuation / decode all read compressed KV through the
TurboQuant Triton path. Keys use TurboQuant_prod:
  * TurboQuant_mse indices on a randomly rotated normalized key
  * QJL sign sketch on the residual
  * stored key norm and residual norm scalars

Values remain uniformly quantized per slot.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    dequant_turboquant_cache_pages,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_prefill import (
    triton_turboquant_prefill_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128
# Long prefill chunks larger than the continuation threshold amortize decode
# kernel launch overhead without requiring prohibitively large scratch buffers.
# Keep this module-level so local experiments can retune it easily.
_LONG_PREFILL_DECODE_CHUNK_SIZE = 1024
_LONG_PREFILL_FA_MIN_SEQ_LEN = 8192
_LONG_PREFILL_FA_MIN_FREE_HEADROOM_BYTES = 1 << 30
_LONG_PREFILL_FA_PAGE_CHUNK_SIZE = 4

logger = init_logger(__name__)


def _get_turboquant_decode_workspace_shapes(
    batch_size: int,
    num_heads: int,
    head_size: int,
    max_num_kv_splits: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    return (
        ((batch_size, num_heads, max_num_kv_splits, head_size + 1), torch.float32),
        ((batch_size, num_heads, head_size), torch.float32),
        ((batch_size, num_heads), torch.float32),
    )


def reserve_turboquant_decode_workspace(
    *,
    vllm_config,
    num_heads: int,
    head_size: int,
) -> None:
    if not is_workspace_manager_initialized():
        return

    batch_size = max(
        vllm_config.scheduler_config.max_num_seqs,
        _CONTINUATION_DECODE_THRESHOLD,
    )
    max_num_kv_splits = vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
    current_workspace_manager().get_simultaneous(
        *_get_turboquant_decode_workspace_shapes(
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
            max_num_kv_splits=max_num_kv_splits,
        )
    )


def _build_key_norm_lut(
    log_min: torch.Tensor, log_max: torch.Tensor, device: torch.device
) -> torch.Tensor:
    lut_idx = torch.arange(256, device=device, dtype=torch.float32)
    step = (log_max - log_min) / 255.0
    return torch.exp2(log_min + lut_idx * step).contiguous()


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = (
            (sliding_window - 1, 0) if sliding_window is not None else None
        )

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = math.ceil(head_size * cfg.key_mse_bits / 8)
        self._qjl_bytes = math.ceil(head_size / 8)
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

    def _local_window_size(self) -> int | None:
        if self.sliding_window is None:
            return None
        return self.sliding_window[0] + 1

    def _ensure_on_device(self, layer, device):
        """Move paper-faithful TurboQuant state to the active device once."""
        cached_device = getattr(layer, "_tq_cached_device", None)
        if cached_device == device:
            return

        layer._tq_Pi = layer._tq_Pi.to(device=device, dtype=torch.float32)
        layer._tq_PiT = layer._tq_PiT.to(device=device, dtype=torch.float32)
        layer._tq_S = layer._tq_S.to(device=device, dtype=torch.float32)
        layer._tq_ST = layer._tq_ST.to(device=device, dtype=torch.float32)
        layer._tq_centroids = layer._tq_centroids.to(device=device, dtype=torch.float32)
        layer._tq_midpoints = layer._tq_midpoints.to(device=device, dtype=torch.float32)
        layer._tq_key_norm_log_min = layer._tq_key_norm_log_min.to(
            device=device, dtype=torch.float32
        )
        layer._tq_key_norm_log_max = layer._tq_key_norm_log_max.to(
            device=device, dtype=torch.float32
        )
        layer._tq_key_norm_lut = _build_key_norm_lut(
            layer._tq_key_norm_log_min,
            layer._tq_key_norm_log_max,
            device,
        )
        layer._tq_cached_device = device

    def _get_decode_workspace(
        self, batch_size: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not is_workspace_manager_initialized():
            return None, None, None

        return tuple(
            current_workspace_manager().get_simultaneous(
                *_get_turboquant_decode_workspace_shapes(
                    batch_size=batch_size,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    max_num_kv_splits=self.max_num_kv_splits,
                )
            )
        )

    def _select_num_kv_splits(self, max_seq_len: int) -> int:
        # Keep the fixed upper bound while using CUDA graphs. In eager mode,
        # very long decode/prefill chunks benefit from fewer splits because the
        # reduction overhead of many small partitions dominates.
        if is_forward_context_available():
            runtime_mode = get_forward_context().cudagraph_runtime_mode
            if runtime_mode != CUDAGraphMode.NONE:
                return self.max_num_kv_splits

        if max_seq_len >= 32768:
            return min(self.max_num_kv_splits, 8)
        if max_seq_len >= 16384:
            return min(self.max_num_kv_splits, 16)
        return self.max_num_kv_splits

    def _select_long_prefill_chunk_size(self, seq_len: int) -> int:
        del seq_len  # reserved for future heuristics
        if is_forward_context_available():
            runtime_mode = get_forward_context().cudagraph_runtime_mode
            if runtime_mode != CUDAGraphMode.NONE:
                return _CONTINUATION_DECODE_THRESHOLD
        return _LONG_PREFILL_DECODE_CHUNK_SIZE

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        ST = tq_layer._tq_ST
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, ST, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                ST,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens],
                kv_cache,
                decode_meta,
                Pi,
                ST,
                centroids,
                PiT,
                layer,
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use CPU-side max to avoid GPU→CPU sync from .item()
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                ST,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_Pi,
            layer._tq_PiT,
            layer._tq_ST,
            layer._tq_centroids,
            layer._tq_midpoints,
            layer._tq_key_norm_log_min,
            layer._tq_key_norm_log_max,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
        )

    # ------------------------------------------------------------------ #
    #  Prefill / continuation: compressed-KV Triton path                  #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        # key/value are consumed by do_kv_cache_update before this call.
        # Attention itself reads only compressed KV from the cache.
        del key, value
        return self._tq_prefill_attention(
            query=query,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            Pi=Pi,
            ST=ST,
            centroids=centroids,
            PiT=PiT,
            layer=layer,
        )

    def _tq_prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        n, hq, d = query.shape
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(n, hq, d, device=query.device, dtype=query.dtype)

        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            cached_len = seq_len - q_len
            q_seq = query[q_start:q_end]
            synth_seq_lens = torch.arange(
                cached_len + 1,
                seq_len + 1,
                device=query.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
            block_table = attn_metadata.block_table[i : i + 1]

            if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                synth_bt = block_table.expand(q_len, -1)
                out = self._decode_prefill_chunk_from_cache(
                    query=q_seq,
                    kv_cache=kv_cache,
                    block_table=synth_bt,
                    seq_lens=synth_seq_lens,
                    max_seq_len=seq_len,
                    Pi=Pi,
                    ST=ST,
                    centroids=centroids,
                    PiT=PiT,
                    layer=layer,
                )
            else:
                out = None
                backend = envs.VLLM_TQ_LONG_PREFILL_BACKEND
                if backend == "native":
                    out = self._try_long_prefill_native_from_cache(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        cached_len=cached_len,
                        PiT=PiT if PiT is not None else Pi.T.contiguous(),
                        ST=ST,
                        centroids=centroids,
                        layer=layer,
                    )
                elif backend == "fa":
                    out = self._try_long_prefill_flash_from_cache(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        cached_len=cached_len,
                        Pi=Pi,
                        ST=ST,
                        centroids=centroids,
                        layer=layer,
                    )
                if out is None:
                    out = self._stream_long_prefill_from_cache(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        seq_lens_dtype=attn_metadata.seq_lens.dtype,
                        cached_len=cached_len,
                        Pi=Pi,
                        ST=ST,
                        centroids=centroids,
                        PiT=PiT,
                        layer=layer,
                    )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _try_long_prefill_native_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        PiT: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor | None:
        if layer is None:
            return None
        if is_forward_context_available():
            runtime_mode = get_forward_context().cudagraph_runtime_mode
            if runtime_mode != CUDAGraphMode.NONE:
                return None

        q_len = query.shape[0]
        target_seq_len = cached_len + q_len
        try:
            return triton_turboquant_prefill_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_len=target_seq_len,
                cached_len=cached_len,
                PiT=PiT,
                ST=ST,
                key_norm_lut=layer._tq_key_norm_lut,
                key_norm_log_min=layer._tq_key_norm_log_min,
                key_norm_log_max=layer._tq_key_norm_log_max,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                sliding_window=self._local_window_size(),
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning_once(
                "TurboQuant native prefill ran out of memory; falling back to "
                "stream prefill."
            )
            return None

    def _can_use_long_prefill_flash(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> bool:
        if seq_len < _LONG_PREFILL_FA_MIN_SEQ_LEN:
            return False
        if not is_flash_attn_varlen_func_available():
            return False
        if not query.is_cuda or not kv_cache.is_cuda:
            return False
        if is_forward_context_available():
            runtime_mode = get_forward_context().cudagraph_runtime_mode
            if runtime_mode != CUDAGraphMode.NONE:
                return False

        block_size = kv_cache.shape[1]
        num_pages = math.ceil(seq_len / block_size)
        if num_pages <= 0 or block_table.shape[1] < num_pages:
            return False

        element_size = torch.empty((), dtype=query.dtype, device="meta").element_size()
        transient_bytes = (
            num_pages
            * block_size
            * self.num_kv_heads
            * self.head_size
            * element_size
            * 2
        )
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(query.device)
        except RuntimeError:
            return True
        return transient_bytes + _LONG_PREFILL_FA_MIN_FREE_HEADROOM_BYTES < free_bytes

    def _try_long_prefill_flash_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor | None:
        if layer is None:
            return None

        q_len = query.shape[0]
        target_seq_len = cached_len + q_len
        if not self._can_use_long_prefill_flash(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=target_seq_len,
        ):
            return None

        try:
            k_cache, v_cache, compact_block_table = dequant_turboquant_cache_pages(
                kv_cache=kv_cache,
                block_table=block_table,
                seq_len=target_seq_len,
                Pi=Pi,
                S=layer._tq_S,
                ST=ST,
                centroids=centroids,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_norm_log_min=layer._tq_key_norm_log_min,
                key_norm_log_max=layer._tq_key_norm_log_max,
                page_chunk_size=_LONG_PREFILL_FA_PAGE_CHUNK_SIZE,
                output_dtype=query.dtype,
            )
            output = torch.empty_like(query)
            cu_seqlens_q = torch.tensor(
                [0, q_len], device=query.device, dtype=torch.int32
            )
            seqused_k = torch.tensor(
                [target_seq_len], device=query.device, dtype=torch.int32
            )
            sliding_window = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )
            fa_version = get_flash_attn_version(head_size=self.head_size)
            num_splits = (
                self._select_num_kv_splits(target_seq_len)
                if fa_version not in (None, 2)
                else 1
            )
            flash_attn_varlen_func(
                q=query.contiguous(),
                k=k_cache,
                v=v_cache,
                out=output,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=target_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=sliding_window,
                block_table=compact_block_table,
                fa_version=fa_version,
                num_splits=num_splits,
            )
            return output
        except torch.cuda.OutOfMemoryError:
            return None

    def _decode_prefill_chunk_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        query_rot: torch.Tensor | None = None,
        query_qjl: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
        mid_o_buf: torch.Tensor | None = None,
        output_buf: torch.Tensor | None = None,
        lse_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            mid_o_buf is None
            and output_buf is None
            and lse_buf is None
            and query.shape[0] <= _CONTINUATION_DECODE_THRESHOLD
        ):
            mid_o_buf, output_buf, lse_buf = self._get_decode_workspace(query.shape[0])
        buf_holder = (
            layer
            if layer is not None
            and (mid_o_buf is None or output_buf is None or lse_buf is None)
            else None
        )
        num_kv_splits = self._select_num_kv_splits(max_seq_len)

        return triton_turboquant_decode_attention(
            query=query,
            query_rot=query_rot,
            query_qjl=query_qjl,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            ST=ST,
            key_norm_lut=layer._tq_key_norm_lut if layer is not None else None,
            key_norm_log_min=(
                layer._tq_key_norm_log_min if layer is not None else None
            ),
            key_norm_log_max=(
                layer._tq_key_norm_log_max if layer is not None else None
            ),
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=buf_holder,
            max_num_kv_splits=num_kv_splits,
            sliding_window=self._local_window_size(),
        )

    def _stream_long_prefill_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens_dtype: torch.dtype,
        cached_len: int,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        q_len = query.shape[0]
        output = torch.empty_like(query)
        target_seq_len = cached_len + q_len
        chunk_size = self._select_long_prefill_chunk_size(target_seq_len)
        seq_lens_full = torch.arange(
            cached_len + 1,
            target_seq_len + 1,
            device=query.device,
            dtype=seq_lens_dtype,
        )
        if PiT is None:
            PiT = Pi.T.contiguous()

        temp_mid_o = temp_output = temp_lse = None
        if chunk_size > _CONTINUATION_DECODE_THRESHOLD:
            num_kv_splits = self._select_num_kv_splits(target_seq_len)
            temp_mid_o = torch.empty(
                chunk_size,
                self.num_heads,
                num_kv_splits,
                self.head_size + 1,
                dtype=torch.float32,
                device=query.device,
            )
            temp_output = torch.empty(
                chunk_size,
                self.num_heads,
                self.head_size,
                dtype=torch.float32,
                device=query.device,
            )
            temp_lse = torch.empty(
                chunk_size,
                self.num_heads,
                dtype=torch.float32,
                device=query.device,
            )

        for chunk_start in range(0, q_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, q_len)
            chunk_len = chunk_end - chunk_start
            synth_seq_lens = seq_lens_full[chunk_start:chunk_end]
            synth_bt = block_table.expand(chunk_len, -1)
            q_chunk = query[chunk_start:chunk_end]
            q_chunk_float = q_chunk.float()
            output[chunk_start:chunk_end] = self._decode_prefill_chunk_from_cache(
                query=q_chunk,
                kv_cache=kv_cache,
                block_table=synth_bt,
                seq_lens=synth_seq_lens,
                max_seq_len=target_seq_len,
                Pi=Pi,
                ST=ST,
                centroids=centroids,
                PiT=PiT,
                query_rot=(q_chunk_float @ PiT).contiguous(),
                query_qjl=(q_chunk_float @ ST).contiguous(),
                layer=layer,
                mid_o_buf=(temp_mid_o[:chunk_len] if temp_mid_o is not None else None),
                output_buf=(
                    temp_output[:chunk_len] if temp_output is not None else None
                ),
                lse_buf=(temp_lse[:chunk_len] if temp_lse is not None else None),
            )

        return output

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        ST: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        mid_o_buf, output_buf, lse_buf = self._get_decode_workspace(query.shape[0])
        buf_holder = (
            layer
            if layer is not None
            and (mid_o_buf is None or output_buf is None or lse_buf is None)
            else None
        )
        num_kv_splits = self._select_num_kv_splits(attn_metadata.max_seq_len)

        return triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            ST=ST,
            key_norm_lut=layer._tq_key_norm_lut if layer is not None else None,
            key_norm_log_min=(
                layer._tq_key_norm_log_min if layer is not None else None
            ),
            key_norm_log_max=(
                layer._tq_key_norm_log_max if layer is not None else None
            ),
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=buf_holder,
            max_num_kv_splits=num_kv_splits,
            sliding_window=self._local_window_size(),
        )
