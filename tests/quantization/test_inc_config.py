# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe as fused_moe
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.inc import (
    INCConfig,
    INCGPTQRowParallelTailLinearMethod,
)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.models.gemma4 import (
    Gemma4Router,
    _dequantize_autoround_gptq_router_weight,
    _infer_autoround_gptq_router_config,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.scalar_type import scalar_types


class DummyLayer:
    pass


class _FakeQuantMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, extra_weight_attrs
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bias
        return torch.zeros(
            *x.shape[:-1],
            layer.output_size,
            dtype=x.dtype,
            device=x.device,
        )


class _FakeQuantConfig:
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase:
        del layer, prefix
        return _FakeQuantMethod()


def _make_inc_config(extra_config: dict[str, dict[str, int]]) -> INCConfig:
    config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        extra_config=extra_config,
    )
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    return config


def test_inc_extra_config_maps_regex_keys_for_fused_qkv() -> None:
    config = _make_inc_config(
        {
            r".*model\.language_model\.layers\.\d+\.self_attn\..*": {
                "bits": 8,
            },
            "model.language_model.layers.5.self_attn.q_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.k_proj": {"bits": 8},
        }
    )
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
        }
    )

    config.apply_vllm_mapper(mapper)

    assert config.get_layer_config(
        DummyLayer(), "language_model.model.layers.5.self_attn.qkv_proj"
    ) == (8, 128, True)


def test_inc_fused_qkv_still_rejects_real_mixed_configs() -> None:
    config = _make_inc_config(
        {
            r".*model\.language_model\.layers\.\d+\.self_attn\..*": {
                "bits": 8,
            },
            "model.language_model.layers.5.self_attn.q_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.k_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.v_proj": {"bits": 4},
        }
    )
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
        }
    )

    config.apply_vllm_mapper(mapper)

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.get_layer_config(
            DummyLayer(), "language_model.model.layers.5.self_attn.qkv_proj"
        )


def test_gate_linear_accepts_quant_config_for_router_weights() -> None:
    layer = GateLinear(
        4,
        3,
        bias=False,
        out_dtype=torch.float32,
        quant_config=_FakeQuantConfig(),
        prefix="layers.0.router.proj",
        disable_tp=True,
    )

    assert hasattr(layer, "qweight")
    assert not hasattr(layer, "weight")
    assert not layer.allow_specialized_router_gemm
    assert not layer.allow_dsv3_router_gemm
    assert not layer.allow_cublas_router_gemm

    output, bias = layer(torch.ones(2, 4))
    assert output.shape == (2, 3)
    assert output.dtype == torch.float32
    assert bias is None


@pytest.mark.parametrize(
    ("num_bits", "weight_type", "zero_point"),
    [
        (4, scalar_types.uint4b8, 7),
        (8, scalar_types.uint8b128, 127),
    ],
)
def test_autoround_gptq_router_weight_dequantizes_symmetric_zero_point(
    num_bits: int,
    weight_type,
    zero_point: int,
) -> None:
    pack_factor = 32 // num_bits
    qweight_unpacked = (
        (torch.arange(8 * 8, dtype=torch.int32).reshape(8, 8) % pack_factor)
        + zero_point
        + 1
    )
    qzeros_unpacked = torch.full((2, 8), zero_point, dtype=torch.int32)
    scales = torch.stack(
        (
            torch.linspace(0.5, 1.2, 8),
            torch.linspace(1.5, 2.2, 8),
        )
    )

    qweight = pack_quantized_values_into_int32(
        qweight_unpacked, weight_type, packed_dim=0
    )
    qzeros = pack_quantized_values_into_int32(
        qzeros_unpacked, weight_type, packed_dim=1
    )

    weight = _dequantize_autoround_gptq_router_weight(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        num_bits=num_bits,
        group_size=4,
        sym=True,
        params_dtype=torch.float16,
    )

    expected_qzeros = qzeros_unpacked + 1
    row_groups = torch.arange(qweight_unpacked.shape[0]) // 4
    expected = (
        (qweight_unpacked - expected_qzeros[row_groups]) * scales[row_groups]
    ).t()
    torch.testing.assert_close(weight, expected.to(torch.float16))


def test_gemma4_router_keeps_dense_weight_for_quantized_checkpoint(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del default_vllm_config
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    router = Gemma4Router(
        SimpleNamespace(hidden_size=4, num_experts=3, rms_norm_eps=1e-6),
        quant_config=_FakeQuantConfig(),
        prefix="layers.0.router",
    )

    assert hasattr(router.proj, "weight")
    assert not hasattr(router.proj, "qweight")


def test_autoround_router_config_can_be_inferred_from_packed_shapes() -> None:
    qweight = torch.empty(352, 128, dtype=torch.int32)
    scales = torch.empty(22, 128, dtype=torch.float16)

    assert _infer_autoround_gptq_router_config(
        qweight=qweight,
        scales=scales,
        dense_weight_shape=torch.Size([128, 2816]),
        sym=True,
    ) == (4, 128, True)

    qweight = torch.empty(704, 128, dtype=torch.int32)
    assert _infer_autoround_gptq_router_config(
        qweight=qweight,
        scales=scales,
        dense_weight_shape=torch.Size([128, 2816]),
        sym=True,
    ) == (8, 128, True)


def test_inc_gptq_row_parallel_tail_fallback_uses_global_group_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.quantization.inc as inc
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(inc, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)

    method = INCGPTQRowParallelTailLinearMethod(
        weight_bits=4,
        group_size=16,
        sym=True,
    )
    layer = torch.nn.Module()
    layer.input_size_per_partition = 24
    method.create_weights(
        layer,
        input_size_per_partition=24,
        output_partition_sizes=[8],
        input_size=48,
        output_size=8,
        params_dtype=torch.float32,
    )

    assert layer.g_idx.tolist() == [1] * 8 + [2] * 16

    qweight_unpacked = torch.full((24, 8), 9, dtype=torch.int32)
    layer.qweight.data.copy_(
        pack_quantized_values_into_int32(
            qweight_unpacked, scalar_types.uint4b8, packed_dim=0
        )
    )
    layer.scales.data.copy_(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
    )
    method.process_weights_after_loading(layer)

    x = torch.ones(1, 24, dtype=torch.float16)
    output = method.apply(layer, x)

    # qweight 9 minus uint4 symmetric bias 8 gives dequant value 1.
    expected = 8 * layer.scales.data[1] + 16 * layer.scales.data[2]
    expected = expected.unsqueeze(0)
    torch.testing.assert_close(output, expected.to(torch.float16))


def test_moe_wna16_forwards_layer_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del w1, w2
        captured.update(kwargs)
        return torch.empty_like(hidden_states)

    monkeypatch.setattr(fused_moe, "fused_experts", fake_fused_experts)

    quant_config = MoeWNA16Config.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "lm_head": False,
        }
    )
    method = MoeWNA16Method(quant_config, moe=SimpleNamespace(disable_inplace=True))
    method.moe.disable_inplace = True

    layer = SimpleNamespace(
        activation=MoEActivation.GELU_TANH,
        w13_qweight=torch.empty(1, 4, 4, dtype=torch.uint8),
        w2_qweight=torch.empty(1, 4, 4, dtype=torch.uint8),
        apply_router_weight_on_input=False,
        global_num_experts=1,
        expert_map=None,
    )

    output = method.apply(
        layer,
        torch.ones(2, 4),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert output.shape == (2, 4)
    assert captured["activation"] == MoEActivation.GELU_TANH
