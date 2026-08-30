# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical M-RoPE cos/sin cache bounds.

M-RoPE enlarges ``max_position_embeddings`` 4x so video timestamps that run
past the token sequence still land inside the cache. Models that certify their
unpruned positions stay within the sequence may lower only the number of
physical cache rows; the semantic maximum that feeds YaRN's frequency math must
not move.
"""

import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope_interleaved import (
    MRotaryEmbeddingInterleaved,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

MAX_POSITION_EMBEDDINGS = 4096
LEGACY_MULTIPLIER = 4

COMMON_KWARGS = dict(
    head_size=128,
    rotary_dim=64,
    base=10000.0,
    is_neox_style=True,
    dtype=torch.bfloat16,
    mrope_section=[16, 8, 8],
)
YARN_KWARGS = dict(scaling_factor=2.0, beta_fast=32, beta_slow=1)


def _build(cls=MRotaryEmbedding, bound=None, **extra):
    return cls(
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        mrope_cache_max_position=bound,
        **COMMON_KWARGS,
        **extra,
    )


def test_unbounded_rows_keep_legacy_multiplier(default_vllm_config):
    rope = _build()

    assert rope.cache_max_position_num == MAX_POSITION_EMBEDDINGS * LEGACY_MULTIPLIER
    assert rope.cos_sin_cache.shape[0] == rope.cache_max_position_num


def test_bounded_rows_shrink_but_keep_semantic_maximum(default_vllm_config):
    legacy = _build()
    bounded = _build(bound=MAX_POSITION_EMBEDDINGS)

    assert bounded.cos_sin_cache.shape[0] == MAX_POSITION_EMBEDDINGS
    assert legacy.cos_sin_cache.shape[0] == MAX_POSITION_EMBEDDINGS * LEGACY_MULTIPLIER
    # YaRN's correction range is derived from this, so it must not shrink.
    assert bounded.max_position_embeddings == legacy.max_position_embeddings


def test_unbounded_delegates_to_base_implementation(default_vllm_config):
    """An unbounded instance must be byte-identical to upstream's own cache."""
    rope = _build()

    assert rope._uses_bounded_cache is False
    delegated = super(MRotaryEmbedding, rope)._compute_cos_sin_cache()

    assert torch.equal(rope.cos_sin_cache, delegated.to(rope.cos_sin_cache.dtype))


def test_unbounded_yarn_delegates_to_yarn_implementation(default_vllm_config):
    rope = _build(**YARN_KWARGS)

    assert rope._uses_bounded_cache is False
    delegated = YaRNScalingRotaryEmbedding._compute_cos_sin_cache(rope)

    assert torch.equal(rope.cos_sin_cache, delegated.to(rope.cos_sin_cache.dtype))


def test_bounded_cache_is_a_bitwise_prefix(default_vllm_config):
    legacy = _build()
    bounded = _build(bound=MAX_POSITION_EMBEDDINGS)

    assert torch.equal(
        legacy.cos_sin_cache[:MAX_POSITION_EMBEDDINGS], bounded.cos_sin_cache
    )


def test_bounded_yarn_cache_is_a_bitwise_prefix(default_vllm_config):
    legacy = _build(**YARN_KWARGS)
    bounded = _build(bound=MAX_POSITION_EMBEDDINGS, **YARN_KWARGS)

    # YaRN sizes its own cache from ``max_position_embeddings * scaling_factor``.
    assert legacy.cos_sin_cache.shape[0] == int(
        MAX_POSITION_EMBEDDINGS * LEGACY_MULTIPLIER * YARN_KWARGS["scaling_factor"]
    )
    assert torch.equal(
        legacy.cos_sin_cache[:MAX_POSITION_EMBEDDINGS], bounded.cos_sin_cache
    )


def test_interleaved_cache_is_bounded_and_a_bitwise_prefix(default_vllm_config):
    """The interleaved subclass used to be re-enlarged by its parent."""
    legacy = MRotaryEmbeddingInterleaved(
        max_position_embeddings=MAX_POSITION_EMBEDDINGS, **COMMON_KWARGS
    )
    bounded = MRotaryEmbeddingInterleaved(
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        mrope_cache_max_position=MAX_POSITION_EMBEDDINGS,
        **COMMON_KWARGS,
    )

    assert legacy.cos_sin_cache.shape[0] == MAX_POSITION_EMBEDDINGS * LEGACY_MULTIPLIER
    assert bounded.cos_sin_cache.shape[0] == MAX_POSITION_EMBEDDINGS
    assert torch.equal(
        legacy.cos_sin_cache[:MAX_POSITION_EMBEDDINGS], bounded.cos_sin_cache
    )


def test_non_positive_bound_is_rejected(default_vllm_config):
    for bound in (0, -1):
        try:
            _build(bound=bound)
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:
            raise AssertionError(f"bound={bound} should have been rejected")


def test_bound_separates_rope_cache_entries(default_vllm_config, monkeypatch):
    """``_ROPE_DICT`` is module state; two bounds must not alias."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.rotary_embedding._ROPE_DICT", {}, raising=True
    )
    rope_kwargs = dict(
        head_size=COMMON_KWARGS["head_size"],
        max_position=MAX_POSITION_EMBEDDINGS,
        is_neox_style=True,
        rope_parameters={
            # ``get_rope`` selects M-RoPE from the presence of ``mrope_section``
            # under the default scaling type.
            "rope_type": "default",
            "rope_theta": COMMON_KWARGS["base"],
            "rope_dim": COMMON_KWARGS["rotary_dim"],
            "mrope_section": COMMON_KWARGS["mrope_section"],
        },
        dtype=COMMON_KWARGS["dtype"],
    )

    unbounded = get_rope(**rope_kwargs)
    bounded = get_rope(**rope_kwargs, mrope_cache_max_position=MAX_POSITION_EMBEDDINGS)
    unbounded_again = get_rope(**rope_kwargs)

    assert unbounded is not bounded
    assert unbounded is unbounded_again
    assert unbounded.cos_sin_cache.shape[0] != bounded.cos_sin_cache.shape[0]


def test_rope_dict_is_restored_after_isolation(default_vllm_config):
    """Guard against the previous test leaking module state."""
    assert isinstance(_ROPE_DICT, dict)
