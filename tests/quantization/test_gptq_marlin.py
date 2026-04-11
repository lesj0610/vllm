from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe as fused_moe_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.quantization.gptq_marlin as gptq_marlin_mod
import vllm.model_executor.parameter as parameter_mod
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _DummyKernel:
    def __init__(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        return None

    def apply_weights(self, layer, x, bias):
        out = torch.zeros((*x.shape[:-1], layer.output_size), dtype=x.dtype)
        if bias is not None:
            out = out + bias
        return out


def test_gptq_marlin_create_weights_uses_ceil_groups_for_row_parallel(
    monkeypatch,
):
    monkeypatch.setattr(
        gptq_marlin_mod, "verify_marlin_supported", lambda **kwargs: None
    )
    monkeypatch.setattr(
        gptq_marlin_mod, "choose_mp_linear_kernel", lambda _: _DummyKernel
    )
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    config = GPTQMarlinConfig(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )
    method = GPTQMarlinLinearMethod(config)
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        input_size_per_partition=2112,
        output_partition_sizes=[2816],
        input_size=4224,
        output_size=2816,
        params_dtype=torch.float16,
        weight_loader=default_weight_loader,
    )

    assert layer.scales.shape == (17, 2816)
    assert layer.qzeros.shape == (17, 352)


def test_inc_row_parallel_overlap_group_load_uses_global_group_offsets(
    monkeypatch,
):
    monkeypatch.setattr(
        gptq_marlin_mod, "verify_marlin_supported", lambda **kwargs: None
    )
    monkeypatch.setattr(
        gptq_marlin_mod, "choose_mp_linear_kernel", lambda _: _DummyKernel
    )
    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 2)

    quant_config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="marlin",
    )
    layer = RowParallelLinear(
        2112,
        2816,
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.down_proj",
    )

    scales = torch.arange(17 * 2816, dtype=layer.scales.dtype).reshape(17, 2816)
    qzeros = torch.arange(17 * 352, dtype=layer.qzeros.dtype).reshape(17, 352)

    layer.weight_loader_v2(layer.scales, scales)
    layer.weight_loader_v2(layer.qzeros, qzeros)

    assert torch.equal(layer.scales.data, scales[8:17])
    assert torch.equal(layer.qzeros.data, qzeros[8:17])


def test_gate_linear_quantized_forward_does_not_require_unquantized_weight(
    monkeypatch,
):
    monkeypatch.setattr(
        gptq_marlin_mod, "verify_marlin_supported", lambda **kwargs: None
    )
    monkeypatch.setattr(
        gptq_marlin_mod, "choose_mp_linear_kernel", lambda _: _DummyKernel
    )
    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    quant_config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="marlin",
    )
    gate = GateLinear(
        2816,
        128,
        bias=False,
        out_dtype=torch.float32,
        quant_config=quant_config,
        prefix="model.layers.0.router.proj",
    )

    assert not hasattr(gate, "weight")
    assert hasattr(gate, "qweight")

    x = torch.randn(3, 2816, dtype=torch.float16)
    output, output_bias = gate(x)

    assert output_bias is None
    assert output.shape == (3, 128)
    assert output.dtype == torch.float32


def test_moe_wna16_apply_passes_layer_activation(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fused_experts(x, w1, w2, **kwargs):
        captured["activation"] = kwargs["activation"]
        captured["quant_config"] = kwargs["quant_config"]
        return torch.zeros_like(x)

    monkeypatch.setattr(fused_moe_mod, "fused_experts", fake_fused_experts)

    method = MoeWNA16Method(
        SimpleNamespace(),
        SimpleNamespace(disable_inplace=False),
    )
    method.moe_quant_config = object()

    layer = SimpleNamespace(
        activation=MoEActivation.GELU,
        w13_qweight=torch.empty(2, 4, 4, dtype=torch.uint8),
        w2_qweight=torch.empty(2, 4, 4, dtype=torch.uint8),
        apply_router_weight_on_input=False,
        global_num_experts=2,
        expert_map=None,
    )

    x = torch.randn(3, 8, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float32)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)

    output = method.apply(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=None,
    )

    assert captured["activation"] == MoEActivation.GELU
    assert captured["quant_config"] is method.moe_quant_config
    assert output.shape == x.shape
