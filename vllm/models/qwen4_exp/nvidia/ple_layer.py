# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

from collections.abc import Sequence

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

from .ngram_embedding import (
    _NVFP4_BLOCK_SIZE,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENVFp4EmbeddingMethod,
    _dequant_nvfp4_rows,
)
from .ops.ple import ple_conv, ple_gate


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        # NVFP4 dequantization unpacks lookup rows per head, so the outer layer
        # keeps the head geometry even when the embedding lives in another
        # process and its submodules were never constructed here.
        self.ple_embedding_dim = int(config.ple_embed_dim)
        self.ple_ngram_heads = (int(config.ngram_size) - 1) * int(
            config.heads_per_ngram
        )
        self.ple_head_dim = self.ple_embedding_dim // self.ple_ngram_heads
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        # The offload process builds the surrounding model on meta while this
        # subtree must own real CPU storage. GPU workers skip the subclass
        # constructor and retain only an empty IPC placeholder.
        with torch.device(PleOffloadLayer.get_target_device()):
            ple_embedding = Qwen4ExpNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                vllm_config.scheduler_config.max_num_batched_tokens,
                data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
                prefix=f"{prefix}.ple_embedding",
                quant_config=quant_config,
                params_dtype=model_config.dtype,
            )
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            # The GPU placeholder never runs the subclass constructor, so the
            # quant method has to be attached from out here.
            ple_embedding._offload_quant_method = (
                Qwen4ExpPLEEmbeddingMethod.from_quant_config(
                    quant_config,
                    f"{prefix}.ple_embedding.ngram_embedding",
                    getattr(config, "ple_embedding_dtype", None),
                )
            )
        self.ple_embedding: nn.Module = ple_embedding
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _get_embedding_weight_scale(self) -> torch.Tensor | None:
        embedding = getattr(self.ple_embedding, "ngram_embedding", None)
        weight_scale = getattr(embedding, "weight_scale", None)
        if weight_scale is not None:
            return weight_scale
        # A GPU worker in offload mode has no embedding submodule; it keeps only
        # the scale the loader retained.
        return getattr(self.ple_embedding, "_offload_weight_scale", None)

    def _dequantize_nvfp4_rows(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
        scale_2: torch.Tensor,
        lut: torch.Tensor,
        packed_row_width: int,
    ) -> torch.Tensor:
        """Unpack ``[codes | block scales]`` rows into ``output_dtype``."""
        packed_rows = embeddings.unflatten(-1, (self.ple_ngram_heads, packed_row_width))
        dequantized = _dequant_nvfp4_rows(
            packed_rows, self.ple_head_dim, scale_2, output_dtype, lut
        )
        return dequantized.flatten(-2)

    def _dequantize_embeddings(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize PLE lookup output."""

        offload_quant_method = getattr(
            self.ple_embedding, "_offload_quant_method", None
        )
        if isinstance(offload_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            scale_2 = getattr(self.ple_embedding, "_offload_weight_scale_2", None)
            if scale_2 is None:
                raise RuntimeError("NVFP4 PLE offload is missing its global scale")
            return self._dequantize_nvfp4_rows(
                embeddings,
                output_dtype,
                scale_2,
                self.ple_embedding._offload_nvfp4_lut,
                self.ple_head_dim // 2 + self.ple_head_dim // _NVFP4_BLOCK_SIZE,
            )

        embedding = getattr(self.ple_embedding, "ngram_embedding", None)
        if embedding is not None:
            return embedding.dequantize(embeddings, output_dtype)

        # A GPU worker in offload mode has no embedding submodule, so the
        # method-owned conversion above is out of reach: the rows arrive
        # already looked up, with only the retained scale to apply.
        if not is_fp8(embeddings):
            return embeddings
        weight_scale = self._get_embedding_weight_scale()
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned PLE lookup while the preceding decoder layer runs."""
        self.ple_embedding.start_prefetch(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.non_spec_query_start_loc
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            query_start_loc = query_start_loc[-num_prefills - 1 :] - num_decode_tokens
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    # State routing consumes the current request metadata on every replay.
    @eager_break_during_capture
    def _short_conv(self, inputs: torch.Tensor, residual: torch.Tensor) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. The residual
        # already contains the gated output, so short convolution is a no-op.
        if attn_metadata is None:
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            state_size = conv_state.size(-1)
            if state_size < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {state_size}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = torch.ops.vllm.qwen4_exp_ple_embed(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
            self.prefix,
        )
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        self._short_conv(conv_input, gated_output)
        return gated_output


# Keep data-dependent n-gram hashing and the large table gather out of the
# compiled model graph, where Inductor can miscompile the embedding indices.


# Keep data-dependent n-gram hashing and the large table gather out of the
# compiled model graph, where Inductor can miscompile the embedding indices.
def qwen4_exp_ple_embed(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Run the PLE embedding lookup eagerly."""
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.ple_embedding(hidden_states, input_ids, query_start_loc, ngram_context)


def qwen4_exp_ple_embed_fake(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Return PLE lookup metadata while tracing the custom op.

    The dtype and last dimension come from the same two accessors the offload
    worker uses to size its cross-process buffer, so the traced metadata cannot
    drift from the real allocation.
    """
    del query_start_loc, ngram_context
    layer = get_forward_context().no_compile_layers[layer_name]
    embedding = layer.ple_embedding
    return torch.empty(
        (
            input_ids.shape[0],
            embedding.get_offload_output_dim(layer.ple_embedding_dim),
        ),
        device=hidden_states.device,
        dtype=embedding.get_offload_output_dtype(hidden_states.dtype),
    )


direct_register_custom_op(
    op_name="qwen4_exp_ple_embed",
    op_func=qwen4_exp_ple_embed,
    mutates_args=[],
    fake_impl=qwen4_exp_ple_embed_fake,
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
    "Qwen4ExpPLENVFp4EmbeddingMethod",
]
