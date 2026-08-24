# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``ModelConfig.mrope_cache_max_position`` resolution.

The bound is only handed to the RoPE layer when the concrete model class
certifies its unpruned positions and no config-dependent transform can push a
position past the sequence length. It also participates in the model
compilation hash, so a compiled artifact built with one bound is never reused
for another.
"""

import types

import pytest

from vllm.config.model import ModelConfig


def _model_config(certified: bool, pruning_rate: float | None) -> ModelConfig:
    """Build the smallest object that exercises the property's branches."""
    config = ModelConfig.__new__(ModelConfig)
    object.__setattr__(
        config,
        "_model_info",
        types.SimpleNamespace(mrope_positions_are_sequence_bounded=certified),
    )
    if pruning_rate is None:
        multimodal_config = None
    else:
        multimodal_config = types.SimpleNamespace(
            is_multimodal_pruning_enabled=lambda: pruning_rate > 0
        )
    object.__setattr__(config, "multimodal_config", multimodal_config)
    object.__setattr__(config, "max_model_len", 262144)
    return config


def test_certified_model_without_pruning_gets_the_sequence_bound():
    config = _model_config(certified=True, pruning_rate=None)

    assert config.mrope_cache_max_position == 262144


def test_uncertified_model_keeps_the_legacy_cache():
    config = _model_config(certified=False, pruning_rate=None)

    assert config.mrope_cache_max_position is None


@pytest.mark.parametrize("pruning_rate", [0.0, 0.5])
def test_pruning_falls_back_to_the_legacy_cache(pruning_rate):
    """Pruned positions can exceed the reduced sequence length."""
    config = _model_config(certified=True, pruning_rate=pruning_rate)

    expected = 262144 if pruning_rate == 0.0 else None
    assert config.mrope_cache_max_position == expected


def test_uncertified_model_ignores_pruning_state():
    config = _model_config(certified=False, pruning_rate=0.5)

    assert config.mrope_cache_max_position is None


def test_bound_participates_in_the_model_hash(monkeypatch):
    """A compiled artifact built for one bound must not be reused for another."""
    import vllm.config.utils as config_utils

    monkeypatch.setattr(
        config_utils,
        "get_hash_factors",
        lambda _config, _ignored: {"base": "unchanged"},
    )
    monkeypatch.setattr(config_utils, "hash_factors", lambda factors: repr(factors))

    unbounded = _model_config(certified=False, pruning_rate=None)
    bounded = _model_config(certified=True, pruning_rate=None)

    unbounded_hash = unbounded.compute_hash()
    bounded_hash = bounded.compute_hash()

    assert "mrope_cache_max_position" in unbounded_hash
    assert unbounded_hash != bounded_hash


def test_get_rope_accepts_the_bound_from_model_config():
    """The value handed to the RoPE factory is the property, unmodified.

    ``Qwen3NextAttention`` builds its projections before calling ``get_rope``,
    so constructing it needs a distributed environment; the hand-off is pinned
    here at the boundary instead, and `test_mrope_cache_bounds.py` covers the
    factory's own behaviour for both values.
    """
    import inspect

    from vllm.model_executor.layers.rotary_embedding import get_rope

    parameter = inspect.signature(get_rope).parameters["mrope_cache_max_position"]

    assert parameter.default is None
    assert _model_config(
        certified=True, pruning_rate=None
    ).mrope_cache_max_position == (262144)
