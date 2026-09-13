# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-cache lifetime for the QSA FlashInfer runner.

A full CUDA graph bakes in the device addresses of the route and mask buffers
a plan owns, so evicting a plan a graph reached leaves that graph replaying
freed pointers -- an MMU fault, not a wrong answer. With capture warmups
enabled, capture commonly reaches a plan through the cache-hit path, since the
warmup forward populates the cache first. These tests pin down that a plan
touched under capture is pinned and survives eviction pressure.

Deliberately free of CUDA and FlashInfer: the runner is built through
``__new__`` so no workspace is allocated, ``_RoutePlan`` is replaced with a
stub, and the capture predicate is patched. ``test_qsa_backend_closure.py``
cannot host these -- it calls ``pytest.importorskip("flashinfer")`` at module
level, so the whole module is skipped wherever FlashInfer is missing.
"""

from collections import OrderedDict

import pytest

import vllm.logger
from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer


class _StubPlan:
    """Stands in for ``_RoutePlan`` without touching CUDA."""

    def __init__(self, *args):
        self.args = args
        self.pinned = False


@pytest.fixture
def runner(monkeypatch):
    """A runner whose plans are stubs and whose capture state is scriptable."""
    monkeypatch.setattr(qsa_flashinfer, "_RoutePlan", _StubPlan)

    capturing = {"value": False}
    monkeypatch.setattr(
        qsa_flashinfer.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing["value"],
    )

    # __init__ would allocate a CUDA workspace; only the cache matters here.
    obj = qsa_flashinfer.QSAFlashInferRunner.__new__(qsa_flashinfer.QSAFlashInferRunner)
    obj._plans = OrderedDict()
    obj._capturing = capturing
    return obj


@pytest.fixture
def clean_warning_cache():
    """Clear the process-wide warning_once cache around a test.

    ``vllm.logger._print_warning_once`` is an ``lru_cache``, so a message any
    earlier test emitted would suppress ours, and a message we emit would
    suppress a later test's. Clearing on the way out has to survive a failed
    assertion, hence the yield.
    """
    vllm.logger._print_warning_once.cache_clear()
    try:
        yield
    finally:
        vllm.logger._print_warning_once.cache_clear()


def _key(rows, width=4):
    """A plan key shaped like the real one; only rows/width vary here."""
    return (rows, width, 8, 1, 128, 4096, 16, "NHD", "fp4")


def test_eager_miss_leaves_plan_unpinned(runner):
    runner._capturing["value"] = False
    plan = runner._plan_for(_key(128))
    assert plan.pinned is False


def test_capture_hit_promotes_the_same_plan(runner):
    runner._capturing["value"] = False
    warmup_plan = runner._plan_for(_key(128))
    assert warmup_plan.pinned is False

    runner._capturing["value"] = True
    capture_plan = runner._plan_for(_key(128))

    # The graph must reach the very object it later replays, not a rebuild.
    assert capture_plan is warmup_plan
    assert capture_plan.pinned is True


def test_capture_miss_pins_on_creation(runner):
    runner._capturing["value"] = True
    plan = runner._plan_for(_key(128))
    assert plan.pinned is True


def test_pinning_is_monotonic(runner):
    runner._capturing["value"] = False
    runner._plan_for(_key(128))
    runner._capturing["value"] = True
    runner._plan_for(_key(128))

    runner._capturing["value"] = False
    plan = runner._plan_for(_key(128))
    # A graph still replays this plan; a later eager step must not unpin it.
    assert plan.pinned is True


def test_pinned_plan_survives_overflow(runner):
    runner._capturing["value"] = False
    runner._plan_for(_key(128))
    runner._capturing["value"] = True
    decode_plan = runner._plan_for(_key(128))
    runner._capturing["value"] = False

    # Prefill chunks arrive with their own row buckets and push past the limit.
    for i in range(qsa_flashinfer._MAX_PLANS * 2):
        runner._plan_for(_key(1024 + i * 128))

    assert _key(128) in runner._plans
    assert runner._plans[_key(128)] is decode_plan


def test_unpinned_plans_are_evicted_in_lru_order(runner):
    runner._capturing["value"] = False
    for i in range(qsa_flashinfer._MAX_PLANS + 2):
        runner._plan_for(_key(1024 + i * 128))

    assert len(runner._plans) == qsa_flashinfer._MAX_PLANS
    # The two least recent went first.
    assert _key(1024) not in runner._plans
    assert _key(1024 + 128) not in runner._plans
    assert _key(1024 + (qsa_flashinfer._MAX_PLANS + 1) * 128) in runner._plans


def test_overflow_with_nothing_evictable_warns_once(
    runner, caplog_vllm, clean_warning_cache
):
    # caplog_vllm, not caplog: vllm's loggers set propagate=False, so the
    # plain fixture sees nothing even while the warning is printed.
    runner._capturing["value"] = True
    for i in range(qsa_flashinfer._MAX_PLANS):
        runner._plan_for(_key(1024 + i * 128))
    assert all(p.pinned for p in runner._plans.values())

    runner._capturing["value"] = False
    with caplog_vllm.at_level("WARNING"):
        # Two distinct new keys: repeating one key would only move_to_end and
        # never re-enter the overflow branch.
        runner._plan_for(_key(8192))
        runner._plan_for(_key(8192 + 128))

    # Count emitted records, not calls: warning_once runs every time and drops
    # the duplicate inside its cache, so call counting proves nothing.
    overflow = [
        r
        for r in caplog_vllm.records
        if "QSA plan cache exceeded its soft limit" in r.message
    ]
    assert len(overflow) == 1
    assert len(runner._plans) > qsa_flashinfer._MAX_PLANS
