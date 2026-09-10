# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ..common.ple import PLEVocabParallelEmbedding, copy_ple_embedding_shard_
from .ops.ple import ple_ngram_ids

logger = init_logger(__name__)


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        # Kept for the packed formats: an NVFP4 lookup returns one row per
        # n-gram head and has to unflatten them before dequantizing.
        self.num_ngram_heads = num_ngram_heads
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""

        def unquantized_or_declared_fp8() -> "Qwen4ExpPLEEmbeddingMethod":
            """Fall back to the dtype the model config declares for the table.

            A checkpoint may quantize the model to NVFP4 while serializing the
            PLE table in FP8 and excluding it from the NVFP4 config. The
            exclusion only says the table is not NVFP4; the declared dtype is
            what says it is still quantized.
            """
            if _ple_dtype_is_fp8(embedding_dtype):
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()

        if quant_config is None:
            return unquantized_or_declared_fp8()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            algo = quant_config._resolve_quant_algo(prefix)
            # Both "NVFP4" and "W4A16_NVFP4" name the packed format, matching
            # how ModelOptNvFp4Config itself reads the algorithm string. The
            # packed lookup is a PLE table format, so it only applies to a PLE
            # table: other prefixes reach this selector as well.
            if (
                algo is not None
                and "NVFP4" in algo
                and prefix.endswith(".ple_embedding.ngram_embedding")
            ):
                logger.info_once(
                    "PLE embedding %s uses the runtime NVFP4 method", prefix
                )
                return Qwen4ExpPLENVFp4EmbeddingMethod()
            if algo == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return unquantized_or_declared_fp8()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return unquantized_or_declared_fp8()
        if isinstance(quant_config, ModelOptNvFp4Config):
            # A non-excluded layer of an NVFP4 checkpoint keeps its packed
            # rows; the declared table dtype does not override that.
            if not quant_config.is_checkpoint_nvfp4_serialized:
                return unquantized_or_declared_fp8()
            logger.info_once("PLE embedding %s uses the runtime NVFP4 method", prefix)
            return Qwen4ExpPLENVFp4EmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return unquantized_or_declared_fp8()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return unquantized_or_declared_fp8()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


_NVFP4_BLOCK_SIZE = 16

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_nvfp4_codes(
    packed: torch.Tensor,
    scale: torch.Tensor,
    scale_2: torch.Tensor,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack NVFP4 codes with FP8 block scales and an FP32 global scale."""
    half = packed.shape[-1]
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(
        *packed.shape[:-1], half * 2
    )
    if lut is None:
        lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = lut[(codes & 0x7).long()]
    sign = ((codes >> 3) & 1).to(torch.float32)
    fp4 = (magnitude * (1 - 2 * sign)).reshape(
        *packed.shape[:-1], -1, _NVFP4_BLOCK_SIZE
    )
    output = fp4 * scale.float().unsqueeze(-1) * scale_2.float()
    return output.reshape(*packed.shape[:-1], half * 2)


def _dequant_nvfp4_rows(
    packed_rows: torch.Tensor,
    head_dim: int,
    scale_2: torch.Tensor,
    output_dtype: torch.dtype,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Dequantize rows containing NVFP4 codes followed by block scales."""
    half = head_dim // 2
    codes = packed_rows[..., :half]
    scales = packed_rows[..., half:].contiguous().view(torch.float8_e4m3fn)
    return _dequant_nvfp4_codes(codes, scales, scale_2, lut).to(output_dtype)


_SHARD_PREFIX = "ngram_embedding.shard_"


def _nvfp4_shard_sink(
    suffix: str,
    packed_codes: dict[int, torch.Tensor],
    packed_scales: dict[int, torch.Tensor],
    packed_outer_scales: dict[int, torch.Tensor],
) -> tuple[int, dict[int, torch.Tensor]] | None:
    """Route one ``shard_<n>.<leaf>`` name to its packed-tensor collection.

    Returns ``None`` when the name is not one of the three packed tensors, so
    the caller can fall back to the generic loader.
    """
    for ending, sink in (
        (".weight_scale_2", packed_outer_scales),
        (".weight_scale", packed_scales),
        (".weight", packed_codes),
    ):
        if not suffix.endswith(ending):
            continue
        shard_text = suffix[: -len(ending)]
        if not shard_text.isdigit():
            return None
        return int(shard_text), sink
    return None


def _get_shared_nvfp4_outer_scale(
    outer_scales: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Validate and return the global scale shared by NVFP4 PLE shards."""
    first_index, reference = next(iter(outer_scales.items()))
    reference = reference.reshape(())
    for shard_index, outer_scale in outer_scales.items():
        if not torch.equal(outer_scale.reshape(()), reference):
            raise ValueError(
                "NVFP4 PLE shards must share the same global scale, but "
                f"shards {first_index} and {shard_index} differ"
            )
    return reference


def _ple_dtype_is_fp8(ple_embedding_dtype: object) -> bool:
    """Return whether the model config declares FP8 PLE checkpoint weights."""

    if ple_embedding_dtype is None:
        return False
    if isinstance(ple_embedding_dtype, torch.dtype):
        return ple_embedding_dtype == torch.float8_e4m3fn
    return str(ple_embedding_dtype).rsplit(".", 1)[-1] == "float8_e4m3fn"


class Qwen4ExpPLENVFp4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """NVFP4 PLE embedding kept packed at runtime.

    The table stays as uint8 codes with FP8 block scales (16 values per
    block) and one FP32 global scale. A lookup only gathers the packed rows;
    the PLE layer dequantizes them after lookup.
    """

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, extra_weight_attrs
        if input_size_per_partition % _NVFP4_BLOCK_SIZE:
            raise ValueError(
                "NVFP4 PLE embedding requires the embedding dim to be a "
                f"multiple of {_NVFP4_BLOCK_SIZE}, got {input_size_per_partition}"
            )
        self.params_dtype = params_dtype
        self.head_dim = input_size_per_partition
        self.packed_row_width = (
            input_size_per_partition // 2
            + input_size_per_partition // _NVFP4_BLOCK_SIZE
        )
        rows = sum(output_partition_sizes)
        if isinstance(layer, Qwen4ExpPLEEmbedding):
            codes = layer.allocate_embedding_weight(
                rows, input_size_per_partition // 2, torch.uint8
            )
        else:
            # A plain VocabParallelEmbedding carrying this method allocates on
            # the device it is constructed under, which is what the resident
            # PLE table's hook does.
            codes = torch.empty(rows, input_size_per_partition // 2, dtype=torch.uint8)
        weight = nn.Parameter(codes, requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

        weight_scale = nn.Parameter(
            torch.empty(
                rows,
                input_size_per_partition // _NVFP4_BLOCK_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_scale", weight_scale)

        weight_scale_2 = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)
        self._lut: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Build the lookup table before CUDA graph capture.
        self._lut = torch.tensor(
            _FP4_VALUES, dtype=torch.float32, device=layer.weight.device
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Gather packed rows for deferred dequantization."""
        codes = F.embedding(input_, layer.weight)
        scales = F.embedding(input_, layer.weight_scale)
        return torch.cat((codes, scales.view(torch.uint8)), dim=-1)

    def dequantize_rows(
        self,
        packed_rows: torch.Tensor,
        scale_2: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize rows containing NVFP4 codes and block scales."""
        lut = self._lut
        if lut is None or lut.device != packed_rows.device:
            lut = torch.tensor(
                _FP4_VALUES, dtype=torch.float32, device=packed_rows.device
            )
            self._lut = lut
        return _dequant_nvfp4_rows(
            packed_rows, self.head_dim, scale_2, output_dtype, lut
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Unpack a lookup result that still holds one packed row per head."""
        scale_2 = getattr(layer, "weight_scale_2", None)
        if scale_2 is None:
            raise RuntimeError("NVFP4 PLE embedding is missing its global scale")
        num_heads = getattr(layer, "num_ngram_heads", 1)
        packed_rows = embeddings.unflatten(-1, (num_heads, self.packed_row_width))
        return self.dequantize_rows(packed_rows, scale_2, output_dtype).flatten(-2)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows through a CUDA view of pinned host memory."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._prefetch_stream = torch.cuda.Stream(device=self._uva_weight.device)
        self._prefetch_buffer = torch.empty(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self._uva_weight.device,
        )
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self.embedding_dim,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        active_output = self._prefetch_buffer[: gathered_ids.shape[0]]
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        embeddings = self._reduce_etp_embeddings(active_output)
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        output = self._prefetch_buffer.new_empty(
            (hidden_states.shape[0], self._output_dim)
        )
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output


class Qwen4ExpNGramEmbedding(PleOffloadLayer):
    # Only a GPU-worker placeholder carries a quant method here: it has no
    # submodules to read one from. Every other configuration leaves it None and
    # the offload metadata accessors fall back to ``ngram_embedding``.
    _offload_quant_method: QuantizeMethodBase | None = None

    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        pinned_host = engram_config is not None and engram_config.cpu_offload
        if pinned_host and envs.VLLM_PLE_CPU_OFFLOAD:
            # VLLM_PLE_CPU_OFFLOAD is this branch's switch for the offload
            # worker and upstream's legacy fallback for
            # EngramConfig.cpu_offload, so setting it asks for both. The
            # worker owns the table in that case; the pinned-host lookup needs
            # an explicit engram_config.cpu_offload with the variable unset.
            pinned_host = False
        if pinned_host and isinstance(
            embedding_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod
        ):
            # The pinned-host table stores one contiguous row per entry, while
            # NVFP4 splits a row into codes and block scales. Nothing reads the
            # second tensor out of pinned storage yet, so refuse the pair
            # instead of looking up half a row.
            raise NotImplementedError(
                "NVFP4 PLE embeddings do not support the pinned-host table; "
                "use the process-level PLE offload (VLLM_PLE_CPU_OFFLOAD)"
            )
        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if pinned_host
            else Qwen4ExpPLEDeviceEmbedding
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        # The model runner sends the CUDA-graph padded token count together with
        # an unpadded query_start_loc. Stale padding must not enter the scatter:
        # its clamped indices would overwrite the last real token.
        num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)
        packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
            input_ids[:num_valid_tokens]
        )
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        ngram_ids = torch.cat(id_blocks, dim=-1)
        if output is None:
            return ngram_ids
        # The custom op discards this return value, so the CPU branch has to
        # honour ``output`` the same way ``ple_ngram_ids`` does on CUDA. The
        # PLE offload subprocess is the only CPU caller, and it would otherwise
        # read an uninitialized scratch buffer.
        output.copy_(ngram_ids)
        return output

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = getattr(self, "ngram_embedding", None)
        if embedding is not None and getattr(embedding, "supports_prefetch", False):
            # The pinned-host table consumed the IDs in start_prefetch and
            # keyed its result on hidden_states; the offload buffer path is
            # refused at construction, so there is nothing to write out here.
            return embedding(hidden_states)
        # Past this point ``hidden_states`` only carries the offload dependency
        # edge; the embedding lookup does not read it.
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        if output_buffer is None:
            return self.ngram_embedding(ngram_ids).flatten(-2)

        # ``output_buffer`` is the cross-process embedding buffer, sized by
        # get_offload_output_dim / get_offload_output_dtype. It is never the
        # n-gram ID scratch above.
        num_tokens = input_ids.numel()
        output = output_buffer[
            :num_tokens, : self.get_offload_output_dim(self.embedding_dim)
        ]
        quant_method = getattr(self.ngram_embedding, "quant_method", None)
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            # Packed rows carry codes and block scales; keep them packed.
            output.copy_(self.ngram_embedding(ngram_ids).flatten(-2))
            return output
        torch.index_select(
            self.ngram_embedding.weight,
            0,
            ngram_ids.reshape(-1),
            out=output.reshape(-1, self.head_dim),
        )
        return output

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = getattr(self, "ngram_embedding", None)
        if embedding is None or not getattr(embedding, "supports_prefetch", False):
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Keep quantized lookup results in their embedding storage dtype."""
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.dtype
        if isinstance(self._offload_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            return torch.uint8
        if isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            return torch.float8_e4m3fn
        return default_dtype

    def get_offload_output_dim(self, default_dim: int) -> int:
        """Keep NVFP4 lookup rows packed while transferring them to the GPU."""
        quant_method = getattr(
            getattr(self, "ngram_embedding", None), "quant_method", None
        )
        if quant_method is None:
            quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            if default_dim % _NVFP4_BLOCK_SIZE:
                raise ValueError(
                    "NVFP4 PLE output dim must be a multiple of "
                    f"{_NVFP4_BLOCK_SIZE}, got {default_dim}"
                )
            return default_dim // 2 + default_dim // _NVFP4_BLOCK_SIZE
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Initialize quantization metadata skipped by the dummy loader."""
        quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
        elif isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale_2",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_offload_nvfp4_lut",
                torch.tensor(_FP4_VALUES, dtype=torch.float32, device=device),
                persistent=False,
            )

    def _shard_row_span(self, shard_index: int) -> tuple[int, int]:
        """Return the checkpoint row offset and row count of one PLE shard."""
        if shard_index >= self.split_ngram_parts:
            raise ValueError(
                f"PLE embedding shard index {shard_index} exceeds "
                f"split_ngram_parts={self.split_ngram_parts}"
            )
        embedding = self.ngram_embedding
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        checkpoint_start = shard_index * shard_size
        expected_rows = max(
            0, min(shard_size, embedding.org_vocab_size - checkpoint_start)
        )
        return checkpoint_start, expected_rows

    def _load_offload_metadata(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Keep only the dequantization metadata a GPU worker still needs.

        The offload process owns the embedding table and ships quantized lookup
        rows unchanged, so a GPU worker never materializes the table. Prefix
        grouping may invoke the caller repeatedly for one module, so the scales
        are treated as incrementally loaded state.
        """
        retained: set[str] = set()
        quant_method = self._offload_quant_method
        device = torch.accelerator.current_accelerator()

        if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            for name, loaded_weight in weights:
                if name != "ngram_embedding.weight_scale":
                    continue
                self.register_buffer(
                    "_offload_weight_scale",
                    loaded_weight.to(device=device),
                    persistent=False,
                )
                retained.add(name)
            return retained

        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            outer_scales: dict[int, torch.Tensor] = {}
            suffix = ".weight_scale_2"
            for name, loaded_weight in weights:
                if not name.startswith(_SHARD_PREFIX) or not name.endswith(suffix):
                    continue
                shard_index = int(name[len(_SHARD_PREFIX) : -len(suffix)])
                outer_scales[shard_index] = loaded_weight
                retained.add(name)
            if not retained:
                raise ValueError(
                    "NVFP4 PLE offload checkpoint is missing its global scale"
                )
            scale_2 = _get_shared_nvfp4_outer_scale(outer_scales).to(device=device)
            self.register_buffer("_offload_weight_scale_2", scale_2, persistent=False)
            self.register_buffer(
                "_offload_nvfp4_lut",
                torch.tensor(_FP4_VALUES, dtype=torch.float32, device=scale_2.device),
                persistent=False,
            )
            return retained

        # Dense offload keeps no metadata; drain the iterator either way.
        for _ in weights:
            pass
        return retained

    def _load_packed_nvfp4_shards(
        self,
        packed_codes: dict[int, torch.Tensor],
        packed_scales: dict[int, torch.Tensor],
        packed_outer_scales: dict[int, torch.Tensor],
    ) -> None:
        """Assemble the runtime NVFP4 table from per-shard codes and scales."""
        if not packed_codes:
            raise ValueError("NVFP4 PLE checkpoint contains no packed shards")
        for shard_index in packed_codes:
            if (
                shard_index not in packed_scales
                or shard_index not in packed_outer_scales
            ):
                raise ValueError(
                    f"NVFP4 PLE shard {shard_index} is missing its scale tensors"
                )
        shared_outer_scale = _get_shared_nvfp4_outer_scale(
            {index: packed_outer_scales[index] for index in packed_codes}
        )

        embedding = self.ngram_embedding
        for shard_index, codes in packed_codes.items():
            checkpoint_start, expected_rows = self._shard_row_span(shard_index)
            expected_codes_shape = (expected_rows, embedding.embedding_dim // 2)
            if tuple(codes.shape) != expected_codes_shape:
                raise ValueError(
                    f"Shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_codes_shape}, got {tuple(codes.shape)}"
                )
            scales = packed_scales[shard_index]
            expected_scales_shape = (
                expected_rows,
                embedding.embedding_dim // _NVFP4_BLOCK_SIZE,
            )
            if tuple(scales.shape) != expected_scales_shape:
                raise ValueError(
                    f"Scale shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_scales_shape}, got {tuple(scales.shape)}"
                )
            tp_bounds = {
                "checkpoint_start": checkpoint_start,
                "tp_start": embedding.shard_indices.org_vocab_start_index,
                "tp_end": embedding.shard_indices.org_vocab_end_index,
            }
            copy_ple_embedding_shard_(embedding.weight.data, codes, **tp_bounds)
            copy_ple_embedding_shard_(embedding.weight_scale.data, scales, **tp_bounds)

        embedding.weight_scale_2.data.copy_(shared_outer_scale)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            return self._load_offload_metadata(weights)

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        quant_method = getattr(self.ngram_embedding, "quant_method", None)
        nvfp4_runtime = isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod)
        packed_codes: dict[int, torch.Tensor] = {}
        packed_scales: dict[int, torch.Tensor] = {}
        packed_outer_scales: dict[int, torch.Tensor] = {}

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue

            if nvfp4_runtime and name.startswith(_SHARD_PREFIX):
                # Packed shards arrive as three tensors that only become a
                # runtime table once every shard has been seen.
                suffix = name[len(_SHARD_PREFIX) :]
                sink = _nvfp4_shard_sink(
                    suffix, packed_codes, packed_scales, packed_outer_scales
                )
                if sink is None:
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index, target = sink
                self._shard_row_span(shard_index)
                target[shard_index] = loaded_weight
                continue

            if (
                not nvfp4_runtime
                and name.startswith(_SHARD_PREFIX)
                and name.endswith(".weight")
            ):
                shard_text = name[len(_SHARD_PREFIX) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                embedding = self.ngram_embedding
                checkpoint_start, expected_rows = self._shard_row_span(int(shard_text))
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_text}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if nvfp4_runtime:
            self._load_packed_nvfp4_shards(
                packed_codes, packed_scales, packed_outer_scales
            )
            loaded.update(
                {
                    "ngram_embedding.weight",
                    "ngram_embedding.weight_scale",
                    "ngram_embedding.weight_scale_2",
                }
            )

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLENVFp4EmbeddingMethod",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
]
