# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.utils import marlin_utils


class _DummyLinear(LinearBase):

    def __init__(self):
        super().__init__(
            input_size=128,
            output_size=128,
            quant_config=None,
            disable_tp=True,
        )
        self.input_size_per_partition = 128
        self.output_size_per_partition = 128


def test_inc_gptq_marlin_shape_guard_falls_back_without_nameerror(monkeypatch):
    cfg = INCConfig(weight_bits=4, group_size=128, sym=True)
    layer = _DummyLinear()

    monkeypatch.setattr(marlin_utils, "check_marlin_supported",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(marlin_utils, "check_marlin_supports_layer",
                        lambda *args, **kwargs: False)

    method = cfg.apply_gptq_quant_layer(layer, prefix="dummy", backend="auto")

    assert method is not None
    assert method.__class__.__name__ == "GPTQLinearMethod"
