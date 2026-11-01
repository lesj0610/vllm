# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config, make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    nvfp4_backend_supports_csf_scales,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_csf_scales,
    has_flashinfer_cutlass_fused_moe,
)
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and nvfp4 support",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
def test_flashinfer_swigluoai_params_are_forwarded(activation, monkeypatch):
    from flashinfer.fused_moe.core import ActivationType

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = FlashInferExperts(moe_config=moe_config, quant_config=quant_config)

    call_args = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        call_args.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )
    experts.apply(
        output=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        hidden_states=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        w1=torch.empty((2, 256, 128), device="cuda", dtype=torch.bfloat16),
        w2=torch.empty((2, 128, 128), device="cuda", dtype=torch.bfloat16),
        topk_weights=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert experts._supports_activation(activation)
    assert call_args["activation_type"] == ActivationType.Swiglu
    for name, value in (
        ("swiglu_alpha", 1.702),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        torch.testing.assert_close(
            call_args[name],
            torch.full((2,), value, device="cuda", dtype=torch.float32),
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        flashinfer_experts = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None
        assert quant_config.g1_alphas is not None
        assert quant_config.g2_alphas is not None
        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=activation
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )


def test_csf_scales_stay_resident_only_on_flashinfer_cutlass(monkeypatch):
    config = make_dummy_moe_config(hidden_dim=2560, intermediate_size=640)
    backend = NvFp4MoeBackend.FLASHINFER_CUTLASS
    has_csf_ops = (
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.has_flashinfer_csf_scales"
    )
    # A FlashInfer without the CSF ops decodes the scales at load.
    monkeypatch.setattr(has_csf_ops, lambda: False)
    assert not nvfp4_backend_supports_csf_scales(backend, config)
    monkeypatch.setattr(has_csf_ops, lambda: True)
    assert nvfp4_backend_supports_csf_scales(backend, config)
    assert not nvfp4_backend_supports_csf_scales(
        NvFp4MoeBackend.FLASHINFER_TRTLLM, config
    )
    # With EP the routing ids are global expert ids.
    monkeypatch.setattr(config.moe_parallel_config, "use_ep", True)
    assert not nvfp4_backend_supports_csf_scales(backend, config)


@pytest.mark.skipif(
    not has_flashinfer_csf_scales(), reason="requires FlashInfer CSF scales"
)
# Qwen3.8-Flash-Next routed-expert shards at TP1 and TP2.
@pytest.mark.parametrize("intermediate_size", [640, 320])
@torch.inference_mode()
def test_flashinfer_resident_csf_scales_match_decoded_scales(
    intermediate_size: int, workspace_init
) -> None:
    from vllm.v1.worker.workspace import lock_workspace

    set_random_seed(29)
    num_experts, hidden_size, topk = 512, 2560, 10
    backend = NvFp4MoeBackend.FLASHINFER_CUTLASS
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
    )
    w13 = torch.randint(
        256, (num_experts, 2 * intermediate_size, hidden_size // 2), device="cuda"
    ).to(torch.uint8)
    w2 = torch.randint(
        256, (num_experts, hidden_size, intermediate_size // 2), device="cuda"
    ).to(torch.uint8)
    # Within one 16-code window, as checkpoint scales mostly are, but for every
    # 61st scale, which lies above it as a CSF exception.
    w13_scale, w2_scale = (
        torch.randint(0x30, 0x40, (*w.shape[:2], w.shape[2] // 8), device="cuda")
        .to(torch.uint8)
        .index_fill_(2, torch.arange(0, w.shape[2] // 8, 61, device="cuda"), 0x48)
        .view(torch.float8_e4m3fn)
        for w in (w13, w2)
    )
    input_scale = torch.ones(num_experts, device="cuda")

    def convert(layer: SimpleNamespace, csf_scales: bool):
        # Copies, as the conversion reorders the weights and scales in place.
        return convert_to_nvfp4_moe_kernel_format(
            backend,
            layer,
            w13.clone(),
            w13_scale.clone(),
            torch.full((num_experts,), 1 / 64, device="cuda"),
            input_scale,
            w2.clone(),
            w2_scale.clone(),
            torch.full((num_experts,), 1 / 64, device="cuda"),
            input_scale,
            is_act_and_mul=True,
            csf_scales=csf_scales,
        )

    kernels = {}
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        for csf_scales in (False, True):
            layer = SimpleNamespace(
                activation=MoEActivation.SILU, moe_config=moe_config
            )
            (
                w1_q,
                layer.w13_weight_scale,
                layer.w13_weight_scale_2,
                layer.w13_input_scale,
                w2_q,
                layer.w2_weight_scale,
                layer.w2_weight_scale_2,
                layer.w2_input_scale,
            ) = convert(layer, csf_scales)
            quant_config = make_nvfp4_moe_quant_config(
                backend,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w13_scale_2=layer.w13_weight_scale_2,
                w2_scale_2=layer.w2_weight_scale_2,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            kernel = make_nvfp4_moe_kernel(
                quant_config,
                moe_config,
                FlashInferExperts,
                backend,
                csf_scales=csf_scales,
            )
            if csf_scales:
                expected_scales = (
                    layer.w13_weight_scale.clone(),
                    layer.w2_weight_scale.clone(),
                )
            kernel.fused_experts.process_weights_after_loading(layer)
            kernels[csf_scales] = (kernel, w1_q, w2_q)

        # The resident experts released the expanded scales.
        assert layer.w13_weight_scale.numel() == layer.w2_weight_scale.numel() == 0
        experts = kernels[True][0].fused_experts
        assert isinstance(experts, FlashInferExperts)
        assert all(csf.exceptions.numel() for csf in experts._csf)
        w1_scratch, w2_scratch = experts.w1_scale, experts.w2_scale
        assert w1_scratch is not None and w2_scratch is not None
        scratch = (w1_scratch, w2_scratch)
        experts._expand_csf_scales(None)
        for expanded, expected in zip(scratch, expected_scales):
            assert torch.equal(expanded.view(torch.uint8), expected.view(torch.uint8))
        with pytest.raises(NotImplementedError, match="reloading"):
            convert(layer, True)

        hidden_states = torch.randn(
            256, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        scores = torch.randn(256, num_experts, device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids, _ = fused_topk(hidden_states, scores, topk, False)

        def apply(csf_scales: bool, tokens: int) -> torch.Tensor:
            kernel, w1, w2 = kernels[csf_scales]
            return kernel.apply(
                hidden_states=hidden_states[:tokens],
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[:tokens],
                topk_ids=topk_ids[:tokens],
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        def poison_scales() -> None:
            for expanded in scratch:
                expanded.view(torch.uint8).fill_(0x7F)

        # One token expands its routed experts' scales, 256 tokens all of them.
        token_counts = (1, 256)
        for tokens in token_counts:
            expected = apply(False, tokens)
            assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
            poison_scales()
            assert torch.equal(apply(True, tokens), expected)

        lock_workspace()
        graphs, outputs = [], []
        for tokens in token_counts:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                outputs.append(apply(True, tokens))
            graphs.append(graph)
        # Replays must expand the newly routed experts' scales themselves.
        hidden_states.neg_()
        topk_ids.copy_(topk_ids.roll(1, dims=0))
        for tokens, graph, output in zip(token_counts, graphs, outputs):
            poison_scales()
            graph.replay()
            assert torch.equal(output, apply(False, tokens))


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
