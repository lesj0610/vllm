# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    _dequantize_moe_wna16_int4_weight,
    _shuffle_gptq_qweight_for_moe,
    _unpack_uint4,
)


def test_moe_wna16_gptq_keeps_qzeros_for_symmetric_autoround():
    cfg = MoeWNA16Config.from_config({
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "lm_head": False,
    })

    assert cfg.linear_quant_method == "gptq"
    assert cfg.has_zp is True


def test_shuffle_gptq_qweight_for_moe_calls_gptq_shuffle(monkeypatch):
    called = {}

    def fake_shuffle(qweight, q_perm, bit):
        called["shape"] = tuple(qweight.shape)
        called["perm_shape"] = tuple(q_perm.shape)
        called["bit"] = bit

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.moe_wna16.ops.gptq_shuffle",
        fake_shuffle,
    )

    qweight = torch.arange(12, dtype=torch.int32).view(3, 4)
    shuffled = _shuffle_gptq_qweight_for_moe(qweight, 4)

    assert shuffled.data_ptr() == qweight.data_ptr()
    assert called == {"shape": (3, 4), "perm_shape": (0,), "bit": 4}


def test_unpack_uint4_expands_nibbles_in_order():
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    unpacked = _unpack_uint4(packed, dim=-1)
    assert torch.equal(unpacked, torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8))


def test_dequantize_moe_wna16_int4_weight_matches_expected_values():
    qweight = torch.tensor([[0x21, 0x43], [0x65, 0x87]], dtype=torch.uint8)
    scales = torch.tensor([[0.5, 2.0], [0.5, 2.0]], dtype=torch.float32)
    qzeros = torch.tensor([[0x11, 0x22]], dtype=torch.uint8)

    dequant = _dequantize_moe_wna16_int4_weight(
        qweight,
        scales,
        qzeros,
        group_size=2,
        out_dtype=torch.float32,
    )

    expected = torch.tensor(
        [[0.0, 0.5, 2.0, 4.0], [2.0, 2.5, 10.0, 12.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(dequant, expected)
