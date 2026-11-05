# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_type,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cutlass_fused_moe,
    flashinfer_expand_csf_scales,
    has_flashinfer_csf_scales,
    has_flashinfer_cutlass_fused_moe,
)

logger = init_logger(__name__)

# Expansion targets of the resident CSF scales, shared by all layers of one
# shape. Each call expands its routed experts' scales right before its MoE
# kernel on the same stream, so sharing is safe while layers run one at a time.
_CSF_SCALE_SCRATCH: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def is_valid_flashinfer_cutlass_fused_moe(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Check if the given problem size is supported by the FlashInfer CUTLASS MoE
    kernel.
    """
    if not has_flashinfer_cutlass_fused_moe():
        logger.debug_once(
            "FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available."
        )
        return False
    # Data type checks
    if (
        w1.dtype != torch.uint8
        or w2.dtype != torch.uint8
        or hidden_states.dtype not in [torch.float32, torch.float16, torch.bfloat16]
    ):
        logger.debug_once(
            "FlashInferExperts disabled: w1/w2 must be torch.uint8 "
            f"(got w1={w1.dtype}, w2={w2.dtype}), hidden_states must be "
            f"float32, float16, or bfloat16 (got {hidden_states.dtype})."
        )
        return False
    return True


class FlashInferExperts(mk.FusedMoEExpertsModular):
    kept_constants: tuple[str, ...] = (
        "gemm1_clamp_limit",
        "gemm1_alpha",
        "gemm1_beta",
        "fake_input_scale",
    )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.use_nvfp4_w4a4:
            layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
            layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)
        if self.csf_scales:
            self._keep_csf_scales(layer)

    def _keep_csf_scales(self, layer: torch.nn.Module) -> None:
        """Replace the layer's NVFP4 block scales by their CSF encoding."""
        csf = (layer.w13_weight_scale_csf, layer.w2_weight_scale_csf)
        scales = (layer.w13_weight_scale, layer.w2_weight_scale)
        if sum(t.nbytes for c in csf for t in c[:3]) >= sum(s.nbytes for s in scales):
            # Scales that compress poorly stay expanded.
            del layer.w13_weight_scale_csf, layer.w2_weight_scale_csf
            self.csf_scales = False
            return
        from flashinfer import SfLayout

        self._csf = csf
        self._csf_layout = SfLayout.layout_128x4
        key = (scales[0].device, scales[0].shape, scales[1].shape)
        if key not in _CSF_SCALE_SCRATCH:
            _CSF_SCALE_SCRATCH[key] = (
                torch.zeros_like(scales[0]),
                torch.zeros_like(scales[1]),
            )
        self.quant_config._w1.scale, self.quant_config._w2.scale = _CSF_SCALE_SCRATCH[
            key
        ]
        for scale in scales:
            scale.data = scale.data.new_empty(0)

    def _expand_csf_scales(self, topk_ids: torch.Tensor | None) -> None:
        """Expand the scales of the routed experts, or of all without ids."""
        # Expanding only the routed experts compares each id with all the ids
        # before it, which outgrows expanding every expert as tokens grow. The
        # two crossed at 1.9-2.2 ids per expert on SM80 (512 experts, top-10).
        ids = None
        if topk_ids is not None and topk_ids.numel() < 2 * self.num_experts:
            ids = topk_ids.to(torch.int)
        w13_csf, w2_csf = self._csf
        # The checkpoint's [w1; w3] rows become the kernels' [w3; w1].
        flashinfer_expand_csf_scales(
            w13_csf, self.w1_scale, ids, self._csf_layout, swap_row_halves=True
        )
        flashinfer_expand_csf_scales(w2_csf, self.w2_scale, ids, self._csf_layout)

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        csf_scales: bool = False,
    ):
        super().__init__(moe_config, quant_config)
        # Keep NVFP4 block scales CSF-compressed and expand them per call.
        self.csf_scales = csf_scales

        assert quant_config.weight_quant_dtype in (
            "mxfp4",
            "nvfp4",
            torch.float8_e4m3fn,
            None,
        ), (
            "Only mxfp4, nvfp4, fp8, bfloat16 and"
            " float16 quantization are currently supported."
        )
        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype
        self.use_dp = moe_config.moe_parallel_config.dp_size > 1
        # Enables DeepSeek-style FP8 block-scale path:
        # - pass per-block weight scales to the kernel
        # - skip input activation quantization (kernel applies scaling)
        self.use_deepseek_fp8_block_scale = quant_config.is_block_quantized

        def _per_expert(value: float | None) -> torch.Tensor | None:
            if value is None:
                return None
            return torch.full(
                (self.num_experts,),
                float(value),
                dtype=torch.float32,
                device=self.device,
            )

        self.gemm1_clamp_limit = _per_expert(quant_config.gemm1_clamp_limit)
        self.gemm1_alpha = _per_expert(quant_config.gemm1_alpha)
        self.gemm1_beta = _per_expert(quant_config.gemm1_beta)

        if (
            quant_config.weight_quant_dtype == "mxfp4"
            and quant_config.quant_dtype == "mxfp8"
        ):
            self.fake_input_scale = torch.ones(
                self.num_experts,
                device=self.device,
                dtype=torch.float32,
            )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self.quant_config.use_fp8_w8a8 and self.quant_config.is_block_quantized

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and (
                p.is_device_capability(90)
                or p.is_device_capability_family(100)
                # SM110 excluded: flashinfer-ai/flashinfer#3134
                or p.is_device_capability_family(120)
            )
            and has_flashinfer_cutlass_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def supports_csf_scales(moe_config: mk.FusedMoEConfig) -> bool:
        """Whether NVFP4 experts can keep CSF block scales resident."""
        hidden = moe_config.hidden_dim
        intermediate = moe_config.intermediate_size_per_partition
        return (
            has_flashinfer_csf_scales()
            # Routing ids index the rank's experts only without EP.
            and not moe_config.moe_parallel_config.use_ep
            and moe_config.is_act_and_mul
            # Per expert, w13's [2I, H/16] and w2's [H, I/16] scales need even
            # columns, w13 halves of whole 16-row slabs and at most 2**24 bytes.
            and hidden % 32 == 0
            and intermediate % 32 == 0
            and 2 * intermediate * (hidden // 16) <= 1 << 24
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        p = current_platform
        scheme = (weight_key, activation_key)
        # The following are supported by FlashInferExperts:
        return (
            # unquantized and fp8 static per-tensor on 9.0+
            (
                scheme
                in [
                    (None, None),
                    (kFp8StaticTensorSym, kFp8StaticTensorSym),
                ]
                and p.has_device_capability(90)
            )
            # fp8 block-scale, wmxfp4a16 on 9.0
            or (
                scheme
                in [
                    (kMxfp4Static, None),
                    (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                ]
                and p.is_device_capability(90)
            )
            # nvfp4, wmxfp4amxfp8 on 10.0+
            or (
                scheme
                in [
                    (kMxfp4Static, kMxfp8Dynamic),
                    (kNvfp4Static, kNvfp4Dynamic),
                ]
                and p.has_device_capability(100)
            )
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU_TANH,
            MoEActivation.RELU2_NO_MUL,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # FLASHINFER_CUTLASS currently uses its down P/F, which does not
        # work with SP. This will be removed in follow up after we get
        # rid of the FlashInfer specific P/F function.
        # TODO: the per-tensor fp8 kernels don't work with MNNVL FI A2As.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        """Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        workspace1 = (M, K)
        workspace2 = (0,)
        # For NVFP4, the output is stored in a packed int8 format,
        # so the actual hidden dim is 2x the size of K here.
        output_shape = (M, K * 2 if self.quant_dtype == "nvfp4" else K)
        # The workspace is determined by `aq`, since it comes after any
        # potential communication op and is involved in the expert computation.
        return (workspace1, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        quant_scales = None
        fc1_expert_weights = None
        fc2_expert_weights = None
        fc1_expert_biases = None
        fc2_expert_biases = None
        swiglu_alpha = None
        swiglu_beta = None
        swiglu_limit = None
        if activation == MoEActivation.SILU:
            swiglu_limit = self.gemm1_clamp_limit
        elif activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            swiglu_alpha = self.gemm1_alpha
            swiglu_beta = self.gemm1_beta
            swiglu_limit = self.gemm1_clamp_limit
        use_mxfp8_act_scaling = False
        use_w4_group_scaling = False
        # Select quantization metadata based on FP8 format/path
        if (
            self.quant_dtype == torch.float8_e4m3fn
            and not self.use_deepseek_fp8_block_scale
        ):
            # FP8 per-tensor path: use global alphas/scales; do not pass input_sf
            quant_scales = [
                self.g1_alphas,  # w13_weight_scale * w13_input_scale
                self.a2_gscale,  # 1.0 / w2_input_scale
                self.g2_alphas,  # w2_weight_scale * w2_input_scale
                self.a1_scale,
            ]

            a1q_scale = None  # not passing input_sf in fp8
            fc1_expert_weights = w1
            fc2_expert_weights = w2
        elif self.quant_dtype == "nvfp4":
            # Ensure w1_scale and w2_scale are not None before calling view
            assert self.w1_scale is not None and self.w2_scale is not None, (
                "w1_scale and w2_scale must not be None for FlashInferExperts"
            )
            # Flashinfer CUTLASS kernel takes scalar global scales,
            # min because inv_scale.
            quant_scales = [
                self.a1_gscale,
                self.w1_scale.view(torch.int32),
                self.g1_alphas,
                self.a2_gscale,
                self.w2_scale.view(torch.int32),
                self.g2_alphas,
            ]
            # FlashInfer API requires weight to be long for nvfp4
            fc1_expert_weights = w1.view(torch.long)
            fc2_expert_weights = w2.view(torch.long)
        elif self.weight_quant_dtype == "mxfp4":
            assert self.w1_scale is not None and self.w2_scale is not None
            assert w1.is_contiguous() and w2.is_contiguous()
            assert topk_ids.is_contiguous()

            fc1_expert_biases = self.w1_bias
            fc2_expert_biases = self.w2_bias
            swiglu_alpha = self.gemm1_alpha
            swiglu_beta = self.gemm1_beta
            swiglu_limit = self.gemm1_clamp_limit

            if self.quant_dtype == "mxfp8":
                assert self.fake_input_scale is not None
                fc1_expert_weights = w1.view(torch.long)
                fc2_expert_weights = w2.view(torch.long)

                quant_scales = [
                    self.w1_scale.view(torch.int32),
                    self.fake_input_scale,
                    self.w2_scale.view(torch.int32),
                    self.fake_input_scale,
                ]
                use_mxfp8_act_scaling = True
            else:
                assert hidden_states.dtype == torch.bfloat16
                fc1_expert_weights = w1
                fc2_expert_weights = w2
                quant_scales = [
                    self.w1_scale,
                    self.w2_scale,
                ]
                a1q_scale = None
                use_w4_group_scaling = True

        elif self.use_deepseek_fp8_block_scale:
            # FP8 block-scale path: provide block-scale weights, omit a1q_scale
            quant_scales = [
                self.w1_scale,
                self.w2_scale,
            ]
            a1q_scale = None
            fc1_expert_weights = w1
            fc2_expert_weights = w2
        else:
            quant_scales = []
            a1q_scale = None
            fc1_expert_weights = w1
            fc2_expert_weights = w2

        if self.csf_scales:
            self._expand_csf_scales(topk_ids)

        _ = flashinfer_cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            fc1_expert_biases=fc1_expert_biases,
            fc2_expert_biases=fc2_expert_biases,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            output=output,
            output_dtype=self.out_dtype,
            quant_scales=quant_scales,
            input_sf=a1q_scale,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            activation_type=activation_to_flashinfer_type(activation),
            # Informs FlashInfer to use the block-scale decoding path when True
            use_deepseek_fp8_block_scale=self.use_deepseek_fp8_block_scale,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            use_w4_group_scaling=use_w4_group_scaling,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        # No support for LoRA in flashinfer_cutlass_fused_moe.
        # See TODOs in flashinfer functions runMoe and runMoeMinLatency.
        raise NotImplementedError("LoRA is not supported for flashinfer_cutlass_moe")
