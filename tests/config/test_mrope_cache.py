# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import vllm.config.utils as config_utils
from vllm.config.model import ModelConfig


def _make_model_config(
    *,
    sequence_bounded: bool,
    max_model_len: int = 128,
    pruning_enabled: bool | None = None,
) -> ModelConfig:
    model_config = object.__new__(ModelConfig)
    object.__setattr__(
        model_config,
        "_model_info",
        SimpleNamespace(
            mrope_positions_are_sequence_bounded=sequence_bounded,
        ),
    )
    object.__setattr__(model_config, "max_model_len", max_model_len)
    multimodal_config = (
        None
        if pruning_enabled is None
        else SimpleNamespace(
            language_model_only=False,
            is_multimodal_pruning_enabled=lambda: pruning_enabled,
        )
    )
    object.__setattr__(model_config, "multimodal_config", multimodal_config)
    return model_config


def test_mrope_cache_capacity_is_fail_closed():
    unknown = _make_model_config(sequence_bounded=False)
    bounded = _make_model_config(sequence_bounded=True, max_model_len=256)

    assert unknown.mrope_cache_max_position is None
    assert bounded.mrope_cache_max_position == 256


def test_mrope_cache_capacity_is_fail_closed_for_pruning():
    unpruned = _make_model_config(sequence_bounded=True, pruning_enabled=False)
    pruned = _make_model_config(sequence_bounded=True, pruning_enabled=True)

    assert unpruned.mrope_cache_max_position == 128
    assert pruned.mrope_cache_max_position is None


def test_mrope_cache_capacity_affects_model_hash(monkeypatch):
    monkeypatch.setattr(
        config_utils,
        "get_hash_factors",
        lambda _config, _ignored_factors: {"base": "unchanged"},
    )

    unknown = _make_model_config(sequence_bounded=False)
    bounded_128 = _make_model_config(sequence_bounded=True, max_model_len=128)
    bounded_256 = _make_model_config(sequence_bounded=True, max_model_len=256)
    pruned = _make_model_config(sequence_bounded=True, pruning_enabled=True)

    assert unknown.compute_hash() != bounded_128.compute_hash()
    assert bounded_128.compute_hash() != bounded_256.compute_hash()
    assert bounded_128.compute_hash() != pruned.compute_hash()
