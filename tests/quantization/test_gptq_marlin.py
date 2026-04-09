import torch

import vllm.model_executor.layers.quantization.gptq_marlin as gptq_marlin_mod
import vllm.model_executor.parameter as parameter_mod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _DummyKernel:
    def __init__(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        return None

    def apply_weights(self, layer, x, bias):
        raise NotImplementedError


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
