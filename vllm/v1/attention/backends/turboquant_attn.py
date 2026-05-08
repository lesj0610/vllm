# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import triton
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
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128


def _get_turboquant_decode_workspace_shapes(
    batch_size: int,
    num_heads: int,
    head_size: int,
    max_num_kv_splits: int,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """Workspace views required by one TurboQuant decode kernel call."""
    return (
        ((batch_size, num_heads, max_num_kv_splits, head_size + 1), torch.float32),
        ((batch_size, num_heads, head_size), output_dtype),
        ((batch_size, num_heads), torch.float32),
    )


def reserve_turboquant_decode_workspace(
    *,
    vllm_config: Any,
    num_heads: int,
    head_size: int,
) -> bool:
    """Pre-grow WorkspaceManager for TurboQuant decode scratch buffers.

    WorkspaceManager has no separate reservation API; the supported way to
    size it is to request the largest views during model initialization or
    profiling, before ``lock_workspace()`` freezes workspace size. The output
    buffer is reserved as float32 so runtime requests with fp16/bf16 query
    dtypes cannot exceed the reservation.
    """
    if not is_workspace_manager_initialized():
        return False

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
    return True


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
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

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

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
        return kv_cache_dtype.startswith("turboquant_")

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
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    # CPU-resident copies used by the prefill path for per-request iteration
    # without per-step D2H syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None

    @staticmethod
    def compute_mm_prefix_range_tensor(
        mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
        num_seqs: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Convert mm-prefix ranges to a padded device tensor.

        Shape is (num_seqs, max_ranges, 2). Empty rows use (0, 0),
        which kernels treat as invalid because start == end.
        """
        if mm_prefix_range is None:
            return None

        range_lists = [
            mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]
        if all(r == [(0, 0)] for r in range_lists):
            return None

        max_ranges = max(len(r) for r in range_lists)
        padded = [list(r) + [(0, 0)] * (max_ranges - len(r)) for r in range_lists]
        return torch.tensor(padded, dtype=torch.int32, device=device).view(
            num_seqs, max_ranges, 2
        )


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
            mm_prefix_range=getattr(cam, "mm_prefix_range", None),
            mm_prefix_range_tensor=getattr(cam, "mm_prefix_range_tensor", None),
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
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
        self.sliding_window = sliding_window

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Detect flash-attn version (FA2/3/4) for prefill paths.
        self.fa_version = get_flash_attn_version(head_size=head_size)
        # vllm_flash_attn FA2 rejects head dimensions above 256 at runtime.
        # Gemma4 global-attention layers use global_head_dim=512, so TQ must
        # fall back to SDPA for those prefill/continuation paths.
        self._can_use_flash_attn = _HAS_FLASH_ATTN and head_size <= 256

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        reserve_turboquant_decode_workspace(
            vllm_config=vllm_config,
            num_heads=self.num_heads,
            head_size=self.head_size,
        )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def _get_decode_workspace(
        self,
        batch_size: int,
        output_dtype: torch.dtype,
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
                    output_dtype=output_dtype,
                )
            )
        )

    @staticmethod
    def _slice_mm_prefix_range(
        mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
        start: int,
        length: int,
    ) -> dict[int, list[tuple[int, int]]] | None:
        """Slice metadata rows and remap them to a zero-based sub-batch."""
        if not mm_prefix_range or length <= 0:
            return None
        sliced = {}
        for new_idx in range(length):
            ranges = mm_prefix_range.get(start + new_idx)
            if ranges:
                sliced[new_idx] = ranges
        return sliced or None

    @staticmethod
    def _slice_mm_prefix_range_tensor(
        mm_prefix_range_tensor: torch.Tensor | None,
        start: int,
        length: int,
    ) -> torch.Tensor | None:
        if mm_prefix_range_tensor is None or length <= 0:
            return None
        return mm_prefix_range_tensor[start : start + length]

    @staticmethod
    def _get_mm_prefix_ranges(
        attn_metadata: TurboQuantMetadata,
        req_idx: int,
    ) -> list[tuple[int, int]]:
        mm_prefix_range = getattr(attn_metadata, "mm_prefix_range", None)
        if not mm_prefix_range:
            return []
        return mm_prefix_range.get(req_idx, []) or []

    @staticmethod
    def _mm_prefix_active_for_queries(
        ranges: list[tuple[int, int]],
        query_start_pos: int,
        q_len: int,
    ) -> bool:
        if not ranges or q_len <= 0:
            return False
        query_end_pos = query_start_pos + q_len - 1
        return any(
            start < end and query_start_pos <= end and query_end_pos >= start
            for start, end in ranges
        )

    def _request_needs_explicit_mask(
        self,
        *,
        query_start_pos: int,
        q_len: int,
        seq_len: int,
        mm_prefix_ranges: list[tuple[int, int]],
    ) -> bool:
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None and seq_len > sliding_window:
            return True
        return self._mm_prefix_active_for_queries(
            mm_prefix_ranges,
            query_start_pos,
            q_len,
        )

    def _metadata_needs_explicit_prefill_mask(
        self,
        attn_metadata: TurboQuantMetadata,
    ) -> bool:
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None and attn_metadata.max_seq_len > sliding_window:
            return True
        mm_prefix_range = getattr(attn_metadata, "mm_prefix_range", None)
        return bool(mm_prefix_range and any(mm_prefix_range.values()))

    def _metadata_needs_explicit_decode_mask(
        self,
        attn_metadata: TurboQuantMetadata,
    ) -> bool:
        """Return True only when Python fallback is still required.

        Sliding-window decode and mm-prefix decode are handled by the TQ Triton
        kernel when mm_prefix_range_tensor is available. Avoid seq_lens.tolist()
        here: decode can run during CUDA graph capture.
        """
        mm_prefix_range = getattr(attn_metadata, "mm_prefix_range", None)
        if not (mm_prefix_range and any(mm_prefix_range.values())):
            return False
        return getattr(attn_metadata, "mm_prefix_range_tensor", None) is None

    def _build_explicit_attention_mask(
        self,
        *,
        q_len: int,
        kv_len: int,
        query_start_pos: int,
        key_start_pos: int,
        device: torch.device,
        mm_prefix_ranges: list[tuple[int, int]],
    ) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + query_start_pos
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0) + key_start_pos
        mask = k_pos <= q_pos

        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            mask &= (q_pos - k_pos) < sliding_window

        for start, end in mm_prefix_ranges:
            if start >= end:
                continue
            q_in_range = (q_pos >= start) & (q_pos <= end)
            k_in_range = (k_pos >= start) & (k_pos <= end)
            mask |= q_in_range & k_in_range

        return mask

    def _masked_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start_pos: int,
        key_start_pos: int = 0,
        mm_prefix_ranges: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        q_len = query.shape[0]
        kv_len = key.shape[0]
        mask = self._build_explicit_attention_mask(
            q_len=q_len,
            kv_len=kv_len,
            query_start_pos=query_start_pos,
            key_start_pos=key_start_pos,
            device=query.device,
            mm_prefix_ranges=mm_prefix_ranges or [],
        )
        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = key.transpose(0, 1).unsqueeze(0)
        v_t = value.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=self.scale,
            enable_gqa=(key.shape[1] < query.shape[1]),
        )
        return out[0].transpose(0, 1)

    def _get_dequant_kv_buffers(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        alloc_len: int,
        head_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buf_shape = (batch_size, num_kv_heads, alloc_len, head_size)
        if is_workspace_manager_initialized():
            buffers = current_workspace_manager().get_simultaneous(
                (buf_shape, torch.float16),
                (buf_shape, torch.float16),
            )
            return buffers[0], buffers[1]
        return (
            torch.empty(buf_shape, device=device, dtype=torch.float16),
            torch.empty(buf_shape, device=device, dtype=torch.float16),
        )

    def _dequant_kv_cache_prefix(
        self,
        layer: Any,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        centroids: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a single request's cached prefix from TQ KV cache."""
        Hk = kv_cache.shape[2]
        D = self.head_size
        device = kv_cache.device
        if seq_len <= 0:
            empty = torch.empty(0, Hk, D, dtype=output_dtype, device=device)
            return empty, empty

        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(seq_len / block_size) * block_size
        BLOCK_D = triton.next_power_of_2(D)

        k_buf, v_buf = self._get_dequant_kv_buffers(
            batch_size=1,
            num_kv_heads=Hk,
            alloc_len=alloc_len,
            head_size=D,
            device=device,
        )
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        grid = (alloc_len, Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        if not self.tq_config.key_fp8:
            assert layer is not None, "TurboQuant MSE dequant requires layer buffers"
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :seq_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, seq_len, D).transpose(0, 1)
        else:
            k_cached_trim = k_cached[0, :, :seq_len, :].transpose(0, 1)

        v_cached_trim = v_cached[0, :, :seq_len, :].transpose(0, 1)
        return k_cached_trim.to(output_dtype), v_cached_trim.to(output_dtype)

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
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
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
            mm_prefix_range = getattr(attn_metadata, "mm_prefix_range", None)
            mm_prefix_range_tensor = getattr(
                attn_metadata, "mm_prefix_range_tensor", None
            )
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
                mm_prefix_range=self._slice_mm_prefix_range(
                    mm_prefix_range, 0, num_decodes
                ),
                mm_prefix_range_tensor=self._slice_mm_prefix_range_tensor(
                    mm_prefix_range_tensor, 0, num_decodes
                ),
                query_start_loc_cpu=attn_metadata.query_start_loc_cpu[: num_decodes + 1]
                if attn_metadata.query_start_loc_cpu is not None
                else None,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[:num_decodes]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
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
                mm_prefix_range=self._slice_mm_prefix_range(
                    mm_prefix_range,
                    num_decodes,
                    prefill_seq_lens.shape[0],
                ),
                mm_prefix_range_tensor=self._slice_mm_prefix_range_tensor(
                    mm_prefix_range_tensor,
                    num_decodes,
                    prefill_seq_lens.shape[0],
                ),
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
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
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if (
            self._can_use_flash_attn
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
        ):
            if self._metadata_needs_explicit_prefill_mask(attn_metadata):
                # flash_attn_varlen cannot represent Gemma4's
                # (causal AND sliding_window) OR mm_prefix mask.
                pass
            else:
                return self._flash_attn_varlen(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.max_query_len,
                )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)
            mm_prefix_ranges = self._get_mm_prefix_ranges(attn_metadata, i)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self._request_needs_explicit_mask(
                    query_start_pos=0,
                    q_len=q_len,
                    seq_len=seq_len,
                    mm_prefix_ranges=mm_prefix_ranges,
                ):
                    out = self._masked_sdpa(
                        q_seq,
                        k_seq,
                        v_seq,
                        query_start_pos=0,
                        mm_prefix_ranges=mm_prefix_ranges,
                    )
                elif self._can_use_flash_attn:
                    # Assign to slice to avoid gpu/cpu sync.
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly when the
                # standard causal mask is sufficient. Explicit mm-prefix or
                # sliding-window masks require dequant + masked SDPA.
                cached_len = seq_len - q_len
                needs_explicit_mask = self._request_needs_explicit_mask(
                    query_start_pos=cached_len,
                    q_len=q_len,
                    seq_len=seq_len,
                    mm_prefix_ranges=mm_prefix_ranges,
                )
                if q_len <= _CONTINUATION_DECODE_THRESHOLD and not needs_explicit_mask:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    # Slice from pre-built arange (no kernel launch)
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = triton_turboquant_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        Pi=Pi,
                        centroids=centroids,
                        scale=self.scale,
                        mse_bits=self.tq_config.key_mse_bits,
                        key_packed_size=self.tq_config.key_packed_size,
                        value_quant_bits=(self.tq_config.effective_value_quant_bits),
                        key_fp8=self.tq_config.key_fp8,
                        norm_correction=self.tq_config.norm_correction,
                        PiT=PiT,
                    )
                else:
                    # Large continuation, or explicit mask required.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        mm_prefix_ranges=mm_prefix_ranges,
                        force_explicit_mask=needs_explicit_mask,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        mm_prefix_ranges: list[tuple[int, int]] | None = None,
        force_explicit_mask: bool = False,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device

        k_cached_trim, v_cached_trim = self._dequant_kv_cache_prefix(
            layer,
            kv_cache,
            block_table,
            cached_len,
            centroids,
            query.dtype,
        )

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if force_explicit_mask:
            return self._masked_sdpa(
                query,
                k_full,
                v_full,
                query_start_pos=cached_len,
                mm_prefix_ranges=mm_prefix_ranges,
            )

        if self._can_use_flash_attn:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            # Assigning to slice uses fill_ which avoids cpu/gpu sync.
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            cu_seqlens_q = self._cu_2_q
            cu_seqlens_k = self._cu_2_k
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            return self._masked_sdpa(
                query,
                k_full,
                v_full,
                query_start_pos=cached_len,
                mm_prefix_ranges=mm_prefix_ranges,
            )

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _masked_decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        centroids: torch.Tensor,
        layer: torch.nn.Module | None,
    ) -> torch.Tensor:
        """Decode fallback for masks not supported by the TQ Triton kernel."""
        output = torch.empty_like(query)
        seq_lens = attn_metadata.seq_lens.tolist()

        for req_idx, seq_len in enumerate(seq_lens):
            block_table = attn_metadata.block_table[req_idx : req_idx + 1]
            k_full, v_full = self._dequant_kv_cache_prefix(
                layer,
                kv_cache,
                block_table,
                seq_len,
                centroids,
                query.dtype,
            )
            mm_prefix_ranges = self._get_mm_prefix_ranges(attn_metadata, req_idx)
            output[req_idx : req_idx + 1] = self._masked_sdpa(
                query[req_idx : req_idx + 1],
                k_full,
                v_full,
                query_start_pos=seq_len - 1,
                mm_prefix_ranges=mm_prefix_ranges,
            ).to(query.dtype)

        return output

    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        Hq = self.num_heads
        assert query.shape[-1] == D
        assert Hq == query.shape[1]

        if self._metadata_needs_explicit_decode_mask(attn_metadata):
            return self._masked_decode_attention(
                query,
                kv_cache,
                attn_metadata,
                centroids,
                layer,
            )

        # output_buf in query dtype — matches the in-kernel cast in stage2.
        mid_o_buf, output_buf, lse_buf = self._get_decode_workspace(B, query.dtype)
        buf_holder = (
            layer
            if layer is not None
            and (mid_o_buf is None or output_buf is None or lse_buf is None)
            else None
        )

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            mm_prefix_range=getattr(attn_metadata, "mm_prefix_range_tensor", None),
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=buf_holder,
            max_num_kv_splits=self.max_num_kv_splits,
            sliding_window=getattr(self, "sliding_window", None),
        )
        return result
