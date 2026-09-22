# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the GDN prefill selector answers, per request and per device."""

from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _resolve_gdn_prefill_backend,
)
from vllm.platforms.interface import DeviceCapability


def _config(requested: str | None, head_k_dim: int = 128):
    additional_config = {} if requested is None else {"gdn_prefill_backend": requested}
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_k_dim)
        ),
    )


def _platform(major: int, minor: int = 0, cuda_major: int = 13):
    return SimpleNamespace(
        is_cuda=lambda: True,
        is_cpu=lambda: False,
        is_device_capability=lambda m: (major * 10 + minor) == m,
        is_device_capability_family=lambda m: major * 10 == m,
        get_cuda_runtime_major=lambda: cuda_major,
        get_device_capability=lambda: DeviceCapability(major=major, minor=minor),
    )


@pytest.fixture
def sm80(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gdn_module, "current_platform", _platform(8, 0))


@pytest.fixture
def sm75(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gdn_module, "current_platform", _platform(7, 5))


def test_sm8x_runs_flashinfer_when_the_build_has_it(
    sm80, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which architectures the build covers is asked of the build."""
    monkeypatch.setattr(gdn_module, "has_flashinfer_gdn_prefill_sm8x", lambda: True)
    assert _resolve_gdn_prefill_backend(_config("flashinfer"))[1] == "flashinfer"


def test_sm8x_refuses_flashinfer_when_the_build_lacks_it(
    sm80, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gdn_module, "has_flashinfer_gdn_prefill_sm8x", lambda: False)
    with pytest.raises(ValueError, match="no path for compute capability"):
        _resolve_gdn_prefill_backend(_config("flashinfer"))


def test_sm8x_is_runnable_but_not_the_default(
    sm80, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runnable and preferred are different questions.

    The SM8x path is new here, so "auto" keeps the fallback until it has been
    measured against it, while a caller who names it still gets it.
    """
    monkeypatch.setattr(gdn_module, "has_flashinfer_gdn_prefill_sm8x", lambda: True)
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "triton"
    assert _resolve_gdn_prefill_backend(_config(None))[1] == "triton"
    assert _resolve_gdn_prefill_backend(_config("flashinfer"))[1] == "flashinfer"


def test_sm8x_needs_the_head_dim_the_kernel_builds(
    sm80, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gdn_module, "has_flashinfer_gdn_prefill_sm8x", lambda: True)
    with pytest.raises(ValueError, match="head_k_dim=64"):
        _resolve_gdn_prefill_backend(_config("flashinfer", head_k_dim=64))


@pytest.mark.parametrize("requested", ["flashinfer", "cutedsl"])
def test_an_unsupported_device_refuses_rather_than_substitutes(
    requested: str, sm75
) -> None:
    """Refusing is not the same as quietly running something else.

    The selector used to log and fall back to Triton, which left a caller who
    asked for a specific kernel with no way to find out they had not got it.
    """
    with pytest.raises(ValueError, match=requested):
        _resolve_gdn_prefill_backend(_config(requested))


def test_auto_falls_back_on_an_unsupported_device(sm75) -> None:
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "triton"


def test_triton_is_always_available(sm75) -> None:
    assert _resolve_gdn_prefill_backend(_config("triton"))[1] == "triton"


def test_sm90_prefers_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdn_module, "current_platform", _platform(9, 0))
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "flashinfer"


def test_sm100_offers_cutedsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdn_module, "current_platform", _platform(10, 0))
    assert _resolve_gdn_prefill_backend(_config("cutedsl"))[1] == "cutedsl"
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "flashinfer"


def test_a_non_cuda_platform_resolves_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gdn_module,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False, is_cpu=lambda: True),
    )
    assert _resolve_gdn_prefill_backend(_config("flashinfer"))[1] == "triton"
