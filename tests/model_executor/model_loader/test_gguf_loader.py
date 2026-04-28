# SPDX-License-Identifier: Apache-2.0

import torch
from types import SimpleNamespace

from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.models.gemma4 import (
    _split_gemma4_gate_up_qweight,
    Gemma4ForCausalLM,
    Gemma4Model,
)


def test_build_gemma4_manual_mapping():
    state_names = {
        "model.layers.0.layer_scalar",
        "model.layers.0.router.scale",
        "model.layers.0.router.per_expert_scale",
        "model.layers.0.router.proj.weight",
        "model.layers.0.experts.gate_up_proj",
        "model.layers.0.experts.down_proj",
        "model.layers.0.post_feedforward_layernorm_1.weight",
        "model.layers.0.post_feedforward_layernorm_2.weight",
        "model.layers.0.pre_feedforward_layernorm_2.weight",
        "vision_tower.std_bias",
        "vision_tower.std_scale",
        "vision_tower.patch_embedder.input_proj.weight",
        "vision_tower.patch_embedder.position_embedding_table",
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight",
        "vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight",
        "vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight",
        "vision_tower.encoder.layers.0.self_attn.q_norm.weight",
        "vision_tower.encoder.layers.0.self_attn.k_norm.weight",
        "vision_tower.encoder.layers.0.input_layernorm.weight",
        "vision_tower.encoder.layers.0.post_attention_layernorm.weight",
        "vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight",
        "vision_tower.encoder.layers.0.post_feedforward_layernorm.weight",
        "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
        "vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
        "vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
        "embed_vision.embedding_projection.weight",
    }

    mapping, handled = GGUFModelLoader._build_gemma4_manual_mapping(
        state_names,
        num_hidden_layers=1,
        vision_num_hidden_layers=1,
    )

    assert mapping["blk.0.layer_output_scale.weight"] == "model.layers.0.layer_scalar"
    assert mapping["blk.0.ffn_gate_inp.scale"] == "model.layers.0.router.scale"
    assert mapping["blk.0.ffn_down_exps.scale"] == (
        "model.layers.0.router.per_expert_scale"
    )
    assert mapping["blk.0.ffn_gate_inp.weight"] == (
        "model.layers.0.router.proj.weight"
    )
    assert mapping["blk.0.ffn_gate_up_exps.weight"] == (
        "model.layers.0.moe.gate_up_proj.weight"
    )
    assert mapping["blk.0.ffn_down_exps.weight"] == (
        "model.layers.0.moe.down_proj.weight"
    )
    assert mapping["v.std_bias"] == "vision_tower.std_bias"
    assert mapping["v.std_scale"] == "vision_tower.std_scale"
    assert mapping["v.patch_embd.weight"] == (
        "vision_tower.patch_embedder.input_proj.weight"
    )
    assert mapping["v.position_embd.weight"] == (
        "vision_tower.patch_embedder.position_embedding_table"
    )
    assert mapping["mm.input_projection.weight"] == (
        "embed_vision.embedding_projection.weight"
    )
    assert mapping["v.blk.0.attn_q.weight"] == (
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    )
    assert mapping["v.blk.0.attn_out.weight"] == (
        "vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight"
    )
    assert mapping["v.blk.0.ln1.weight"] == (
        "vision_tower.encoder.layers.0.input_layernorm.weight"
    )
    assert mapping["v.blk.0.attn_post_norm.weight"] == (
        "vision_tower.encoder.layers.0.post_attention_layernorm.weight"
    )
    assert mapping["v.blk.0.ffn_gate.weight"] == (
        "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
    )
    assert "model.layers.0.experts.gate_up_proj" in handled
    assert "model.layers.0.experts.down_proj" in handled


def test_transform_gemma4_router_and_moe_tensors():
    patch_weight = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)
    name, transformed = GGUFModelLoader._transform_gemma4_gguf_tensor_name_and_weight(
        "vision_tower.patch_embedder.input_proj.weight",
        patch_weight,
    )
    assert name == "vision_tower.patch_embedder.input_proj.weight"
    assert transformed.shape == (2, 12)
    assert torch.equal(transformed, patch_weight.reshape(2, 12))

    assert (
        GGUFModelLoader._expand_gemma4_gguf_moe_tensor(
            "model.layers.0.moe.gate_up_proj.qweight",
            torch.randn(5, 8, 3),
        )
        is None
    )
    assert (
        GGUFModelLoader._expand_gemma4_gguf_moe_tensor(
            "model.layers.0.moe.down_proj.qweight",
            torch.randn(13, 17, 2),
        )
        is None
    )


def test_split_gemma4_gate_up_qweight():
    weight = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)

    gate_weight, up_weight = _split_gemma4_gate_up_qweight(weight)

    assert gate_weight.shape == (2, 3, 4)
    assert up_weight.shape == (2, 3, 4)
    assert torch.equal(gate_weight, weight[:, :3, :])
    assert torch.equal(up_weight, weight[:, 3:, :])


def test_gemma4_qweight_load_loops_over_all_experts(monkeypatch):
    records = []

    def fake_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id):
        records.append((shard_id, expert_id, loaded_weight.clone()))

    model = object.__new__(Gemma4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=2,
    )
    model.quant_config = None
    model.named_parameters = lambda: iter(
        [
            (
                "model.layers.0.moe.experts.w13_qweight",
                SimpleNamespace(weight_loader=fake_weight_loader),
            ),
            (
                "model.layers.0.moe.experts.w2_qweight",
                SimpleNamespace(weight_loader=fake_weight_loader),
            ),
        ]
    )
    model.named_buffers = lambda: iter([])

    monkeypatch.setattr(
        "vllm.model_executor.models.gemma4.is_pp_missing_parameter",
        lambda name, model: False,
    )

    weights = [
        (
            "model.layers.0.moe.gate_up_proj.qweight",
            torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4),
        ),
        (
            "model.layers.0.moe.down_proj.qweight",
            torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5),
        ),
    ]

    Gemma4Model.load_weights(model, weights)

    assert [(shard, eid) for shard, eid, _ in records] == [
        ("w1", 0),
        ("w3", 0),
        ("w2", 0),
    ]
    assert torch.equal(records[0][2], weights[0][1][:, :3, :])
    assert torch.equal(records[1][2], weights[0][1][:, 3:, :])
    assert torch.equal(records[2][2], weights[1][1])
