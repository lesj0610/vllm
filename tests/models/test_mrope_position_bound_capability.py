# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sequence-bounded M-RoPE position capability and its registry cache.

The capability is fail-closed: only a concrete model class that declares it in
its own body is certified. It also travels through the on-disk ``_ModelInfo``
cache, so a cache file written before the field existed must not be trusted.
"""

import json

import pytest

from vllm.model_executor.models.interfaces import (
    has_sequence_bounded_mrope_positions,
)
from vllm.model_executor.models.registry import _LazyRegisteredModel, _ModelInfo


class _Base:
    pass


class _Certified(_Base):
    mrope_positions_are_sequence_bounded = True


class _DerivedFromCertified(_Certified):
    """Inherits the attribute but does not re-declare it."""


class _ExplicitlyOptedOut(_Certified):
    mrope_positions_are_sequence_bounded = False


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_Base, False),
        (_Certified, True),
        (_DerivedFromCertified, False),
        (_ExplicitlyOptedOut, False),
    ],
)
def test_capability_requires_a_concrete_declaration(model, expected):
    assert has_sequence_bounded_mrope_positions(model) is expected


def test_derived_class_still_inherits_the_attribute():
    """Guard the reason ``__dict__`` is used instead of ``getattr``."""
    assert _DerivedFromCertified.mrope_positions_are_sequence_bounded is True
    assert has_sequence_bounded_mrope_positions(_DerivedFromCertified) is False


def test_qwen3_5_moe_does_not_inherit_the_dense_opt_in():
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    )

    # The MoE entrypoint really is a subclass of the certified dense one.
    assert issubclass(
        Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration
    )
    assert has_sequence_bounded_mrope_positions(Qwen3_5ForConditionalGeneration) is True
    assert (
        has_sequence_bounded_mrope_positions(Qwen3_5MoeForConditionalGeneration)
        is False
    )


def test_pre_field_cache_file_is_reinspected_then_reused(tmp_path, monkeypatch):
    """A cache written before the field existed must not be loaded as-is."""
    model = _LazyRegisteredModel(
        module_name="vllm.model_executor.models.qwen3_5",
        class_name="Qwen3_5ForConditionalGeneration",
    )
    monkeypatch.setattr(
        _LazyRegisteredModel, "_get_cache_dir", staticmethod(lambda: tmp_path)
    )

    # First inspection populates the cache in its current shape.
    fresh = model.inspect_model_cls()
    assert fresh.mrope_positions_are_sequence_bounded is True

    cache_path = tmp_path / model._get_cache_filename()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    module_hash = cached["hash"]
    assert "mrope_positions_are_sequence_bounded" in cached["modelinfo"]

    # Rewrite it the way an older vLLM would have: field absent entirely.
    legacy = dict(cached["modelinfo"])
    legacy.pop("mrope_positions_are_sequence_bounded")
    cache_path.write_text(
        json.dumps({"hash": module_hash, "modelinfo": legacy}), encoding="utf-8"
    )
    assert model._load_modelinfo_from_cache(module_hash) is None

    # First call re-inspects and rewrites the cache.
    saves: list[str] = []
    original_save = _LazyRegisteredModel._save_modelinfo_to_cache

    def counting_save(self, mi, module_hash):
        saves.append(self.class_name)
        return original_save(self, mi, module_hash)

    monkeypatch.setattr(_LazyRegisteredModel, "_save_modelinfo_to_cache", counting_save)

    reinspected = model.inspect_model_cls()
    assert reinspected.mrope_positions_are_sequence_bounded is True
    assert len(saves) == 1, "stale cache should have been re-inspected and rewritten"

    # Second call must be a warm hit: no further inspection, no further save.
    warm = model.inspect_model_cls()
    assert warm.mrope_positions_are_sequence_bounded is True
    assert warm.architecture == reinspected.architecture
    assert len(saves) == 1, "warm hit should not re-inspect the model class"


def test_model_info_has_no_default_for_the_capability():
    """No dataclass default keeps future non-default fields unconstrained."""
    field = _ModelInfo.__dataclass_fields__["mrope_positions_are_sequence_bounded"]

    assert field.default is field.default_factory
