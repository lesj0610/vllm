# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent attention workspace reservation on the shared arena."""

import contextlib
import gc
import importlib
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    reset_workspace_manager,
)


@contextlib.contextmanager
def _ws(shared=True):
    """A workspace manager that exists only for the duration of one test."""
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    current_workspace_manager().shares_attention_workspace = shared
    try:
        yield current_workspace_manager()
    finally:
        reset_workspace_manager()


@contextlib.contextmanager
def _null_context(*args, **kwargs):
    yield


def _spec(non_causal=False):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        non_causal=non_causal,
    )


def _stub(cls, **attrs):
    """An instance with only the attributes the code under test reads."""
    obj = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _record(sink, item, result=None):
    sink.append(item)
    return result


def _groups(*builders):
    """One attention group per builder, each with a single ubatch."""
    return [[SimpleNamespace(metadata_builders=[b]) for b in builders]]


class _Wrapper:
    """Stand-in for a FlashInfer wrapper: a float view plus its own int buffer."""

    def __init__(self, float_buffer, int_bytes):
        self._float_workspace_buffer = float_buffer
        self._int_workspace_buffer = torch.empty(int_bytes, dtype=torch.uint8)


@pytest.fixture(scope="module")
def fi():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer

    return flashinfer


@pytest.fixture
def builder(monkeypatch, fi):
    """FlashInfer builders on CPU whose wrappers are recorded stand-ins."""

    def prefill(self, causal=True):
        self._prefill_wrapper = _Wrapper(self._get_workspace_buffer(), 64)
        return self._prefill_wrapper

    def decode(self, batch_size, use_cudagraph=False):
        wrapper = _Wrapper(self._get_workspace_buffer(), 128)
        self._decode_wrappers[batch_size, use_cudagraph] = wrapper
        return wrapper

    cls = fi.FlashInferMetadataBuilder
    monkeypatch.setattr(cls, "_get_prefill_wrapper", prefill)
    monkeypatch.setattr(cls, "_get_decode_wrapper", decode)
    touched: list = []
    monkeypatch.setattr(fi, "_get_trtllm_workspace_buffer", lambda: touched.append(1))

    def make(float_bytes=4096, prefill=False, decode=False, mm_prefix=False):
        return _stub(
            cls,
            device=torch.device("cpu"),
            touched=touched,
            _workspace_buffer=None,
            _prefill_wrapper=None,
            _decode_wrappers={},
            _default_workspace_buffer_size=lambda: float_bytes,
            _reservation_trtllm_prefill=prefill,
            use_trtllm_decode_attention=decode,
            model_config=SimpleNamespace(is_mm_prefix_lm=mm_prefix, max_model_len=16),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=3, max_num_batched_tokens=4
                )
            ),
            enable_cuda_graph=True,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[2, 8]),
            _decode_cudagraph_max_bs=4,
        )

    return make


_MIXED_SPEC = UniformTypeKVCacheSpecs(
    block_size=16, kv_cache_specs={"l0": _spec(), "l1": _spec(non_causal=True)}
)


@pytest.mark.parametrize(
    ("builder_name", "dcp", "spec", "expected"),
    [
        pytest.param("base", 1, None, None, id="base-builder-opts-out"),
        pytest.param("gdn", 1, None, False, id="gdn-is-neutral"),
        pytest.param("fi", 1, _spec(), True, id="flashinfer-requires"),
        pytest.param("fi", 2, _spec(), None, id="dcp-opts-out"),
        pytest.param("fi", 1, _spec(non_causal=True), None, id="non-causal-opts-out"),
        pytest.param("fi", 1, _MIXED_SPEC, None, id="non-causal-in-uniform-spec"),
    ],
)
def test_shipped_builders_declare_their_workspace_support(
    fi, builder_name, dcp, spec, expected
):
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    cls = {
        "base": AttentionMetadataBuilder,
        "gdn": GDNAttentionMetadataBuilder,
        "fi": fi.FlashInferMetadataBuilder,
    }[builder_name]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp)
    )
    assert cls.persistent_workspace_profiling_support(config, spec) is expected


@pytest.mark.parametrize(
    ("routes", "wrappers", "trtllm"),
    [
        pytest.param({}, {"prefill", (3, False), (2, True)}, False, id="native"),
        pytest.param({"prefill": True, "decode": True}, set(), True, id="trtllm"),
        pytest.param(
            {"prefill": True, "decode": True, "mm_prefix": True},
            {"prefill"},
            True,
            id="mm-prefix-keeps-native-prefill",
        ),
    ],
)
def test_reservation_builds_the_wrappers_its_routes_run(
    builder, routes, wrappers, trtllm
):
    b = builder(**routes)
    with _ws():
        worker_utils.reserve_attention_workspace(
            SimpleNamespace(attn_groups=_groups(b))
        )
    built = set(b._decode_wrappers) | ({"prefill"} if b._prefill_wrapper else set())
    assert built == wrappers
    assert bool(b.touched) is trtllm


@pytest.mark.parametrize("case", ["builders-differ", "scratch-grew-first", "opted-out"])
def test_wrappers_hold_the_final_arena(builder, case):
    """No wrapper keeps an allocation a later growth replaced, and a model that
    opted out keeps FlashInfer's own buffer."""
    sizes = (2048, 8192) if case == "builders-differ" else (4096,)
    builders = [builder(size) for size in sizes]
    with _ws(shared=case != "opted-out") as manager:
        if case == "scratch-grew-first":
            manager.get_simultaneous(((16384,), torch.uint8))  # profile_run's scratch
        worker_utils.reserve_attention_workspace(
            SimpleNamespace(attn_groups=_groups(*builders))
        )
        if case == "opted-out":
            assert all(b._prefill_wrapper is None for b in builders)
            assert builders[0]._get_workspace_buffer().numel() == 4096
            assert manager.workspace_sizes_bytes() == (0,)
            return
        arena = manager._current_workspaces[0]
        assert arena is not None
        assert arena.numel() == {"builders-differ": 8192}.get(case, 16384)
        floats = {
            w._float_workspace_buffer.untyped_storage().data_ptr()
            for b in builders
            for w in (b._prefill_wrapper, *b._decode_wrappers.values())
        }
        assert floats == {arena.untyped_storage().data_ptr()}


def _runner(monkeypatch, version):
    name = "gpu_model_runner" if version == "v1" else "gpu.model_runner"
    module = importlib.import_module(f"vllm.v1.worker.{name}")
    runner = _stub(module.GPUModelRunner, vllm_config=object())
    runner._attn_group_iterator = lambda: iter(runner.attn_groups[0])
    monkeypatch.setattr(worker_utils, "set_current_vllm_config", _null_context)

    def install(init_fn, cleanup_fn):
        # V1 owns the bootstrap as runner methods; V2 as module helpers.
        if version == "v1":
            runner._init_minimal_kv_cache_for_profiling = init_fn
            runner._cleanup_profiling_kv_cache = cleanup_fn
            return
        from vllm.v1.worker.gpu import cudagraph_utils as cg

        kv_init = "_init_minimal_kv_cache_for_profiling"
        monkeypatch.setattr(cg, kv_init, lambda _r: init_fn())
        monkeypatch.setattr(cg, "_teardown_profiling_state", lambda _r: cleanup_fn())

    return runner, install


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("phase", ["kept", "reserve-fails", "cleanup-fails", "grew"])
def test_profiled_workspace_is_what_the_reservation_keeps(monkeypatch, version, phase):
    """Read free memory before the minimal KV cache exists and after it is gone,
    with the builders still held; then release them. A step that fails owns the
    error, and the teardown runs once even when it is that failure."""
    runner, install = _runner(monkeypatch, version)
    events: list = []
    refs: dict[str, Any] = {}
    free = iter([10_000, 10_001 if phase == "grew" else 6_416])

    def memory_info():
        events.append(("free", "b" in refs and refs["b"]() is not None))
        return next(free), 0

    for name, fn in (
        ("synchronize", lambda: None),
        ("empty_cache", lambda: None),
        ("get_memory_info", memory_info),
    ):
        monkeypatch.setattr(torch.accelerator, name, fn)

    class Builder:
        def prepare_workspace_for_profiling(self, materialize):
            if phase == "reserve-fails":
                raise ValueError("reserve")
            if materialize:
                self.int_workspace = torch.empty(1536, dtype=torch.uint8)
            else:
                current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
            events.append(f"prepare:{materialize}")

    def init():
        events.append("init")
        b = Builder()
        refs["b"] = weakref.ref(b)
        runner.attn_groups = _groups(b)

    def cleanup():
        events.append("cleanup")
        del runner.attn_groups
        if phase == "cleanup-fails":
            raise RuntimeError("cleanup")

    install(init, cleanup)
    reserved = ["prepare:False", "prepare:True"]
    with _ws():
        if phase == "kept":
            kept = worker_utils.profile_persistent_attention_workspace(runner)
            assert kept == 3_584
            assert runner._profiled_persistent_workspace_sizes == (2048,)
        else:
            error = ValueError if phase == "reserve-fails" else RuntimeError
            with pytest.raises(error) as exc_info:
                worker_utils.profile_persistent_attention_workspace(runner)
            del exc_info  # its traceback holds the builder that raised
            reserved = [] if phase == "reserve-fails" else reserved
    closing = [("free", True)] if phase in ("kept", "grew") else []
    assert events == [("free", False), "init", *reserved, "cleanup", *closing]
    gc.collect()
    assert refs["b"]() is None


@pytest.mark.parametrize("case", ["grew-before", "grew-during"])
def test_reservation_ceiling_is_recorded_once_and_enforced(monkeypatch, case):
    runner, _ = _runner(monkeypatch, "v1")
    calls, sizes = [], iter([2048, 1024, 4096])

    def fake_reserve(_runner):
        calls.append(_runner)
        current_workspace_manager().get_simultaneous(((next(sizes),), torch.uint8))

    monkeypatch.setattr(worker_utils, "reserve_attention_workspace", fake_reserve)
    reserve = worker_utils.reserve_persistent_attention_workspace

    with _ws() as manager:
        manager.get_simultaneous(((1024,), torch.uint8))
        if case == "grew-before":
            runner._profiled_persistent_workspace_sizes = (1024,)
            manager.get_simultaneous(((2048,), torch.uint8))
            with pytest.raises(AssertionError, match="profiled size before"):
                reserve(runner)
            assert calls == []
            return
        reserve(runner)
        assert runner._profiled_persistent_workspace_sizes == (2048,)
        reserve(runner)  # a smaller request is fine
        with pytest.raises(AssertionError, match="profiled size during"):
            reserve(runner)


class _Prepare:
    """A metadata builder that only records the reservation calls it receives."""

    def __init__(self):
        self.events: list = []
        self.reserved = False

    def prepare_workspace_for_profiling(self, materialize):
        self.events.append(f"prepare:{materialize}")
        self.reserved = self.reserved or materialize


@pytest.fixture
def capture(monkeypatch):
    """Both measurements read a free-memory pair around the graphs they capture,
    so the wrappers have to exist before the first reading."""

    def setup(module, accel_extra=(), **attrs):
        builder = _Prepare()
        events = builder.events
        free = iter([(10_000, 0), (9_000, 0)])
        stubs: dict[str, Any] = {
            "synchronize": lambda *a: None,
            "empty_cache": lambda *a: None,
            "get_memory_info": lambda: _record(events, "memory_info", next(free)),
            **dict(accel_extra),
        }
        for name, fn in stubs.items():
            monkeypatch.setattr(module.torch.accelerator, name, fn)
        monkeypatch.setattr(module, "lock_workspace", lambda: events.append("lock"))
        runner = _stub(
            module.GPUModelRunner,
            device=torch.device("cpu"),
            attn_groups=_groups(builder),
            **attrs,
        )
        return events, builder, runner

    return setup


def test_v1_graph_profiling_reserves_before_its_first_sample(monkeypatch, capture):
    from vllm.v1.worker import gpu_model_runner as module

    events, _, runner = capture(
        module,
        vllm_config=object(),
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [
                (CUDAGraphMode.PIECEWISE, [SimpleNamespace(num_tokens=1)])
            ],
            cudagraph_keys={},
        ),
        lora_config=None,
        maybe_remove_all_loras=lambda _: None,
        _init_minimal_kv_cache_for_profiling=lambda: None,
        _cleanup_profiling_kv_cache=lambda: None,
        _create_encoder_cudagraph_manager=lambda: None,
        _freeze_gc=_null_context,
    )
    runner._warmup_and_capture = lambda *a, **k: events.append("capture")
    for name, fn in (
        ("graph_capture", lambda device: _null_context()),
        ("set_cudagraph_capturing_enabled", lambda _: None),
        ("set_current_vllm_config", lambda _: _null_context()),
    ):
        monkeypatch.setattr(module, name, fn)
    monkeypatch.setattr(module.current_platform, "graph_pool_handle", lambda: None)

    with _ws():
        assert runner.profile_cudagraph_memory() == 1_000
    order = ["prepare:False", "prepare:True", "memory_info", "capture", "memory_info"]
    assert events == order


def test_v2_capture_reserves_before_it_measures_and_then_locks(monkeypatch, capture):
    from vllm.v1.worker.gpu import model_runner as module

    class Manager:
        def needs_capture(self):
            return True

        def capture(self, *args, **kwargs):
            events.append("capture")
            groups = kwargs.get("attn_groups", args[5] if len(args) > 5 else None)
            assert groups is not None
            assert groups[0][0].metadata_builders[0].reserved
            return {}

    reserved, allocated = iter([1_000, 1_000, 1_128, 1_128]), iter([500, 500, 628, 628])
    events, _, runner = capture(
        module,
        accel_extra=(
            ("memory_reserved", lambda d: next(reserved)),
            ("memory_allocated", lambda d: next(allocated)),
        ),
        cudagraph_manager=Manager(),
        is_encoder_only=False,
        lora_config=None,
        maybe_setup_dummy_loras=lambda lora_config: _null_context(),
        model=object(),
        model_state=SimpleNamespace(supports_mm_inputs=False),
        input_buffers=object(),
        intermediate_tensors=None,
        block_tables=object(),
        kv_cache_config=object(),
        use_aux_hidden_state_outputs=False,
        speculator=None,
        adaptive_verification=None,
        pcp_manager=None,
        kv_connector=SimpleNamespace(reset_capture_state=lambda: None),
    )

    with _ws():
        assert runner.capture_model() == 1_000
    order = ["prepare:False", "prepare:True", "memory_info", "capture"]
    assert events == order + ["memory_info", "lock"]
