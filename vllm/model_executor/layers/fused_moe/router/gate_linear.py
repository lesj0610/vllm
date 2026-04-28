# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with three-tier GEMM dispatch:

    1. DSV3 specialized kernel (SM90+, batch<=16, supported dims)
    2. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 + fp32 out_dtype)
    3. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        is_hopper_or_blackwell = current_platform.is_device_capability(
            (9, 0)
        ) or current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and is_hopper_or_blackwell and not bias
        )

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so Tier 3 computes in fp32 natively.
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        # Quantized fallback dequantization helpers expect LinearBase-style
        # partition metadata. ReplicatedLinear is effectively a single shard.
        self.input_size_per_partition = input_size
        self.out_dtype = out_dtype
        router_weight = getattr(self, "weight", None)
        self.register_buffer("_dequant_router_weight", None, persistent=False)

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
        )

        # cuBLAS bf16→fp32 eligibility
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and router_weight is not None
            and router_weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init.

        Useful when the required dtype depends on the expert quantization
        method which is only known after the gate is constructed.
        """
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self.allow_specialized_router_gemm
            and out_dtype == torch.float32
        ):
            router_weight = getattr(self, "weight", None)
            self.allow_cublas_router_gemm = (
                router_weight is not None
                and router_weight.dtype == torch.bfloat16
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        import vllm._custom_ops as ops

        if (
            self.out_dtype == torch.float32
            and getattr(self, "weight", None) is None
            and self.quant_method is not None
        ):
            dequant_router_weight = self._dequant_router_weight
            if dequant_router_weight is None:
                eye_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
                eye = torch.eye(
                    self.input_size_per_partition,
                    dtype=eye_dtype,
                    device=x.device,
                )
                dequant_router_weight = (
                    self.quant_method.apply(self, eye, bias=None)
                    .to(torch.float32)
                    .T
                )
                self._dequant_router_weight = dequant_router_weight
            output = F.linear(x.to(torch.float32), dequant_router_weight)
            return output, None

        # Tier 1: DSV3 specialized kernel
        if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 2: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = ops.router_gemm_bf16_fp32(x, self.weight)
            return output, None

        # Tier 3: F.linear (ReplicatedLinear)
        router_weight = getattr(self, "weight", None)
        if (
            self.out_dtype is not None
            and router_weight is not None
            and x.dtype != router_weight.dtype
        ):
            x = x.to(router_weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
