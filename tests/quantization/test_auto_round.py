# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


@pytest.mark.skipif(
    not current_platform.is_cpu()
    and not current_platform.is_xpu()
    and not current_platform.is_cuda(),
    reason="only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(model, enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
    assert output
    print(f"{output[0][1]}")


def _get_gemma4_router_config_infer():
    gemma4 = pytest.importorskip(
        "vllm.model_executor.models.gemma4", exc_type=ImportError
    )
    return gemma4._infer_autoround_gptq_router_config


@pytest.mark.parametrize(
    ("qweight_rows", "expected_bits"),
    [
        (16, 4),
        (32, 8),
    ],
)
def test_gemma4_autoround_router_config_inference(qweight_rows, expected_bits):
    infer_router_config = _get_gemma4_router_config_infer()
    num_bits, group_size, sym = infer_router_config(
        qweight=torch.empty(qweight_rows, 4, dtype=torch.int32),
        scales=torch.empty(2, 4),
        dense_weight_shape=torch.Size([4, 128]),
        sym=True,
    )
    assert num_bits == expected_bits
    assert group_size == 64
    assert sym is True


def test_gemma4_autoround_router_config_inference_rejects_bad_shape():
    infer_router_config = _get_gemma4_router_config_infer()
    with pytest.raises(ValueError, match="cannot infer AutoRound bit width"):
        infer_router_config(
            qweight=torch.empty(15, 4, dtype=torch.int32),
            scales=torch.empty(2, 4),
            dense_weight_shape=torch.Size([4, 128]),
            sym=True,
        )


def test_inc_extra_config_maps_regex_keys_for_gemma4_fused_qkv():
    from vllm.model_executor.layers.quantization.inc import INCConfig
    from vllm.model_executor.models.utils import WeightsMapper

    config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        extra_config={
            r".*model\.language_model\.layers\.\d+\.self_attn\..*": {
                "bits": 8,
            },
            "model.language_model.layers.5.self_attn.q_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.k_proj": {"bits": 8},
        },
    )
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "language_model.model.",
            }
        )
    )

    assert config.get_layer_config(
        object(), "language_model.model.layers.5.self_attn.qkv_proj"
    ) == (8, 128, True)
