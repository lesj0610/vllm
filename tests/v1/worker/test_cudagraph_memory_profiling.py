# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent attention workspace reservation, as a set of case tables.

DECL         what each shipped backend declares, per spec and parallel config
MATERIALIZE  the routes a builder reaches, and the wrappers they build
ARENA        one arena, sized by the largest default, surviving the lock
DISPATCH     the prefill route is resolved from runtime inputs
LIFECYCLE    reserve, hold past the closing measurement, release, and fail
CEILING      the recorded ceiling is set once and enforced afterwards
CAPTURE      both generations reserve before they measure, then lock
OVERRIDE     the block override is restored even when config building raises
REGISTRY     V1 still validates the KV cache spec registry
SCALE_CACHE  teardown clears quantized scale views, with or without a kv_cache
"""

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
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker import gpu_model_runner as v1
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)


@pytest.mark.parametrize("raises", [False, True])
def test_override_is_restored_even_when_config_computation_raises(monkeypatch, raises):
    """A failure here must not leak the profiling override into real sizing."""
    cfg = SimpleNamespace(num_gpu_blocks_override=7)
    runner = SimpleNamespace(vllm_config=None, cache_config=cfg, max_num_reqs=4)
    runner.compilation_config = SimpleNamespace(max_cudagraph_capture_size=8)
    seen = []

    def from_groups(vllm_config, groups, available_memory):
        seen.append(cfg.num_gpu_blocks_override)
        if raises:
            raise RuntimeError("boom")
        return "minimal"

    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda c, s: [])
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_config_from_groups", from_groups)

    if raises:
        with pytest.raises(RuntimeError, match="boom"):
            worker_utils.build_minimal_kv_cache_config(runner, object())
    else:
        assert worker_utils.build_minimal_kv_cache_config(runner, object()) == "minimal"
    # The override was in force while the config was built, and is gone after.
    assert seen == [4]
    assert cfg.num_gpu_blocks_override == 7


def test_registry_v1_validates_the_spec_before_building_the_config(monkeypatch):
    runner = v1.GPUModelRunner.__new__(v1.GPUModelRunner)
    checked: list = []
    spec, cfg, events = object(), SimpleNamespace(num_blocks=1), []
    runner.get_kv_cache_spec = lambda: spec
    runner.cache_config = SimpleNamespace(num_gpu_blocks=None)
    runner.initialize_kv_cache = lambda config, is_profiling: events.append("init")

    registry = v1.KVCacheSpecRegistry
    monkeypatch.setattr(registry, "check_kv_cache_spec_registry", checked.append)
    monkeypatch.setattr(v1, "build_minimal_kv_cache_config", lambda r, s: cfg)

    runner._init_minimal_kv_cache_for_profiling()
    assert checked == [spec] and events == ["init"]


def _layer(with_kv_cache):
    impl = SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object())
    layer = SimpleNamespace(impl=impl)
    if with_kv_cache:
        layer.kv_cache = torch.empty(4)
    return layer


@pytest.mark.parametrize("with_kv_cache", [True, False])
def test_scale_cache_is_cleared_with_or_without_a_kv_cache(with_kv_cache):
    """Scale views can live on a layer that has no ``kv_cache`` attribute."""
    layer = _layer(with_kv_cache)
    worker_utils.clear_layer_kv_caches([layer])

    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    if with_kv_cache:
        assert isinstance(layer.kv_cache, torch.Tensor)
        assert layer.kv_cache.numel() == 0
    else:
        assert not hasattr(layer, "kv_cache")


def test_scale_cache_shutdown_and_profiling_teardown_share_one_helper(monkeypatch):
    """V1 clears layers the same way on both paths, so neither can drift."""
    layer, seen = _layer(with_kv_cache=False), []
    runner = v1.GPUModelRunner.__new__(v1.GPUModelRunner)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=4)
    runner.compilation_config = SimpleNamespace(static_forward_context={"l": layer})
    runner.kv_caches = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(v1, "clear_layer_kv_caches", lambda ls: seen.append(list(ls)))

    runner._cleanup_profiling_kv_cache()
    assert seen == [[layer]]
    assert runner.cache_config.num_gpu_blocks is None


@contextlib.contextmanager
def _ws(num_ubatches=1):
    """A workspace manager that exists only for the duration of one test."""
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"), num_ubatches=num_ubatches)
    try:
        yield
    finally:
        reset_workspace_manager()


@contextlib.contextmanager
def _null_context(*args, **kwargs):
    yield


def _spec(head_size, non_causal=False):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.float16,
        non_causal=non_causal,
    )


def _record(sink, item, result=None):
    sink.append(item)
    return result


def _stub(cls, **attrs):
    """An instance with only the attributes the code under test reads."""
    obj = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


class _Wrapper:
    """Stand-in for a FlashInfer wrapper: a float view plus its own int buffer."""

    def __init__(self, float_buffer=None, int_bytes=1):
        self._float_workspace_buffer = (
            float_buffer if float_buffer is not None else torch.empty(1, torch.uint8)
        )
        self._int_workspace_buffer = torch.empty(max(int_bytes, 1), dtype=torch.uint8)


def _builder(flashinfer, *, cls=None, float_bytes=4096, **over):
    """A FlashInfer builder wired for the reservation legs on CPU."""
    cls = cls or flashinfer.FlashInferMetadataBuilder
    b = cls.__new__(cls)
    b.device = torch.device("cpu")
    b._workspace_buffer = None
    b.use_dcp = False
    b.use_xqa = over.pop("xqa", False)
    b._reservation_trtllm_prefill = over.pop("trtllm_prefill", False)
    # Decode defaults to trtllm so a reservation stays on the native prefill
    # leg unless a test asks for the decode one.
    b.use_trtllm_decode_attention = over.pop("trtllm_decode", True)
    sizes = over.pop("capture_sizes", None)
    b.enable_cuda_graph = sizes is not None
    b.compilation_config = SimpleNamespace(cudagraph_capture_sizes=sizes)
    b._decode_cudagraph_max_bs = over.pop("decode_max_bs", 0)
    b.kv_cache_spec = _spec(128, non_causal=over.pop("non_causal", False))
    b.model_config = SimpleNamespace(
        dtype=torch.float16,
        is_mm_prefix_lm=over.pop("mm_prefix", False),
        max_model_len=over.pop("max_model_len", 1024),
    )
    b.max_num_batched_tokens = over.pop("max_num_batched_tokens", 8)
    b.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=b.max_num_batched_tokens,
            max_num_seqs=over.pop("max_num_seqs", 4),
        )
    )
    b._prefill_wrapper = b._noncausal_prefill_wrapper = None
    b._decode_wrapper = b._cascade_wrapper = None
    b._decode_wrappers_cudagraph = {}
    if float_bytes is not None:
        b._default_workspace_buffer_size = lambda: float_bytes
    for name, value in over.items():
        setattr(b, name, value)
    return b


@pytest.fixture(scope="module")
def fi():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer

    return flashinfer


_ACCEL_CALLS = ("synchronize", "empty_cache", "memory_allocated", "memory_reserved")


@pytest.fixture
def accel(monkeypatch):
    """Neutralize the accelerator calls the profiling paths make."""

    def patch(module, **extra):
        stubs: dict[str, Any] = dict.fromkeys(_ACCEL_CALLS, lambda *a: None)
        for name, fn in (stubs | extra).items():
            monkeypatch.setattr(module.torch.accelerator, name, fn)

    return patch


@pytest.fixture
def capture(monkeypatch, accel):
    """Both generations read a free-memory pair around the capture and lock the
    workspace at the end, so only the runner attributes differ."""

    def setup(module, accel_extra=(), **attrs):
        events: list = []
        free = iter([(10_000, 0), (9_000, 0)])
        accel(
            module,
            get_memory_info=lambda: _record(events, "memory_info", next(free)),
            **dict(accel_extra),
        )
        monkeypatch.setattr(module, "lock_workspace", lambda: events.append("lock"))
        return events, _stub(module.GPUModelRunner, device=torch.device("cpu"), **attrs)

    return setup


def _runner(monkeypatch, accel, version):
    name = "gpu_model_runner" if version == "v1" else "gpu.model_runner"
    module = importlib.import_module(f"vllm.v1.worker.{name}")
    runner = module.GPUModelRunner.__new__(module.GPUModelRunner)
    runner.vllm_config, runner.device = object(), torch.device("cpu")
    runner._attn_group_iterator = lambda: iter(runner.attn_groups[0])
    monkeypatch.setattr(worker_utils, "set_current_vllm_config", _null_context)
    accel(module)

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

    def group(*builders):
        runner.attn_groups = [[SimpleNamespace(metadata_builders=list(builders))]]

    return SimpleNamespace(module=module, runner=runner, install=install, group=group)


_MIXED_SPEC = UniformTypeKVCacheSpecs(
    block_size=16,
    kv_cache_specs={"l0": _spec(128), "l1": _spec(128, non_causal=True)},
)
_DECL = [
    # builder, dcp, mm_prefix, spec, expected
    ("base", 1, False, None, None, "base-builder-opts-out"),
    ("gdn", 1, False, None, False, "gdn-is-neutral"),
    ("fi", 1, False, _spec(128), True, "flashinfer-requires"),
    ("fi", 2, False, _spec(128), None, "dcp-opts-out"),
    ("fi", 1, True, _spec(128), True, "mm-prefix-supported"),
    ("fi", 1, False, _spec(128, non_causal=True), None, "non-causal-opts-out"),
    ("fi", 1, False, _MIXED_SPEC, None, "non-causal-inside-uniform-spec"),
]


@pytest.mark.parametrize(
    ("builder", "dcp", "mm_prefix", "spec", "expected"),
    [pytest.param(*c[:5], id=c[5]) for c in _DECL],
)
def test_shipped_builders_declare_their_workspace_support(
    fi, builder, dcp, mm_prefix, spec, expected
):
    """The declarations the worker gate consumes, as the backends ship them."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    classes = {
        "base": AttentionMetadataBuilder,
        "gdn": GDNAttentionMetadataBuilder,
        "fi": fi.FlashInferMetadataBuilder,
    }
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=mm_prefix),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )
    support = classes[builder].persistent_workspace_profiling_support(config, spec)
    assert support is expected


_MATERIALIZE = [
    # trtllm_prefill, trtllm_decode, non_causal, mm_prefix, xqa -> routes
    (False, False, False, False, False, (True, False, True, False), "all-native"),
    (False, True, False, False, False, (True, False, False, True), "trtllm-decode"),
    (True, False, False, False, False, (False, True, True, False), "trtllm-prefill"),
    (True, True, False, False, False, (False, True, False, True), "all-trtllm"),
    (True, True, False, True, False, (True, True, False, True), "mm-prefix-mixed"),
    (True, True, True, False, False, (True, False, False, False), "non-causal-native"),
    (True, True, True, False, True, (True, False, False, True), "non-causal-xqa"),
]


_FLAGS = ("trtllm_prefill", "trtllm_decode", "non_causal", "mm_prefix", "xqa")


@pytest.mark.parametrize(
    ("flags", "routes"),
    [pytest.param(dict(zip(_FLAGS, c)), c[5], id=c[6]) for c in _MATERIALIZE],
)
def test_materialize_builds_exactly_the_wrappers_its_routes_declare(
    monkeypatch, flags, routes
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as fi

    class Recording(fi.FlashInferMetadataBuilder):  # type: ignore[misc]
        calls: list[tuple[str, Any]]

        def _get_prefill_wrapper(self, causal=True):
            assert causal
            self.calls.append(("prefill", None))
            self._prefill_wrapper = _Wrapper(self._get_workspace_buffer(), 64)
            return self._prefill_wrapper

        def _get_decode_wrapper(self, batch_size, use_cudagraph=False):
            self.calls.append(("decode_cg" if use_cudagraph else "decode", batch_size))
            wrapper = _Wrapper(self._get_workspace_buffer(), 128)
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = wrapper
            else:
                self._decode_wrapper = wrapper
            return wrapper

    b = _builder(
        fi,
        cls=Recording,
        **flags,
        max_num_batched_tokens=4,
        max_num_seqs=3,
        max_model_len=16,
        capture_sizes=[0, 2, 4, 8],
        decode_max_bs=4,
    )
    b.calls = []
    touches: list = []
    monkeypatch.setattr(fi, "_get_trtllm_workspace_buffer", lambda: touches.append(1))
    native_prefill, trtllm_p, native_decode, trtllm_d = routes

    with _ws():
        assert b._get_workspace_routes() == routes
        b.prepare_workspace_for_profiling(False)
        b.prepare_workspace_for_profiling(True)

        expected: list = [("prefill", None)] if native_prefill else []
        if native_decode:
            expected += [("decode", 3), ("decode_cg", 2), ("decode_cg", 4)]
        assert b.calls == expected
        assert len(touches) == int(trtllm_p or trtllm_d)
        arena = current_workspace_manager().workspace_sizes_bytes()[0]
        assert (arena > 0) is (native_prefill or native_decode)
        # Every wrapper holds the final arena and owns its own int buffer.
        wrappers = [w for w in (b._prefill_wrapper, b._decode_wrapper) if w]
        wrappers += list(b._decode_wrappers_cudagraph.values())
        ptr = b._workspace_buffer.data_ptr() if wrappers else None
        assert all(w._float_workspace_buffer.data_ptr() == ptr for w in wrappers)
        ints = {w._int_workspace_buffer.data_ptr() for w in wrappers}
        assert len(ints) == len(wrappers)


@pytest.mark.parametrize(
    "case", ["small-then-large", "locked-late-wrapper", "head-footprint", "env-floor"]
)
def test_arena_settles_at_the_largest_default_and_survives_the_lock(
    monkeypatch, fi, case
):
    """Pass one sizes every arena, pass two builds wrappers on the final one."""
    built: list = []

    def prefill(self, causal=True):
        built.append(_Wrapper(self._get_workspace_buffer(), 64))
        return built[-1]

    monkeypatch.setattr(fi.FlashInferMetadataBuilder, "_get_prefill_wrapper", prefill)
    monkeypatch.setattr(fi, "_get_trtllm_workspace_buffer", lambda: None)
    if case in ("head-footprint", "env-floor"):
        # No float_bytes override, so the builder computes its own default.
        footprint = 8 * 4 * 16 * fi.FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
        configured = 1 if case == "head-footprint" else footprint + 1
        monkeypatch.setattr(fi.envs, "VLLM_BATCH_INVARIANT", False)
        monkeypatch.setattr(
            fi.envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", configured
        )
        expected = max(footprint, configured)
        builders = [_builder(fi, float_bytes=None, num_qo_heads=4, head_dim=16)]
    else:
        sizes = (2048, 8192) if case == "small-then-large" else (4096,)
        expected = max(sizes)
        builders = [_builder(fi, float_bytes=size) for size in sizes]

    with _ws():
        for b in builders:
            b.prepare_workspace_for_profiling(False)
        settled = current_workspace_manager().workspace_sizes_bytes()
        for b in builders:
            b.prepare_workspace_for_profiling(True)

        backing = current_workspace_manager()._current_workspaces[0]
        assert len(built) == len(builders)
        assert builders[-1]._default_workspace_buffer_size() == expected
        assert backing.numel() >= expected and settled == (backing.numel(),)
        # No wrapper kept a view of an allocation a later growth released, and
        # materialization did not grow the arena any further.
        ptr = backing.data_ptr()
        assert all(w._float_workspace_buffer.data_ptr() == ptr for w in built)
        assert current_workspace_manager().workspace_sizes_bytes() == settled

        # A wrapper built after the lock asks for the default with no argument
        # and finds the settled arena instead of growing a locked one.
        lock_workspace()
        late = builders[-1]._get_workspace_buffer()
        assert late.numel() >= expected and late.data_ptr() == ptr


_DISPATCH_INPUTS = dict(
    num_qo_heads=32,
    num_kv_heads=8,
    dcp_world_size=1,
    cache_dtype="auto",
    q_data_type_prefill=torch.float16,
    has_sinks=True,
    reorder_batch_threshold=2,
    max_num_batched_tokens=17,
    model_config=SimpleNamespace(max_model_len=4096),
)


@pytest.mark.parametrize(
    ("page_size", "configured", "expected_force"),
    [
        pytest.param(16, None, None, id="configured-dispatch"),
        pytest.param(128, False, True, id="large-page-forces-trtllm"),
    ],
)
def test_dispatch_prefill_route_comes_from_runtime_inputs(
    monkeypatch, fi, page_size, configured, expected_force
):
    """Only the inputs the reservation changes; the rest is the selector's own."""
    b = _stub(
        fi.FlashInferMetadataBuilder,
        page_size=page_size,
        attention_config=SimpleNamespace(use_trtllm_attention=configured),
        **_DISPATCH_INPUTS,
    )
    calls: list = []
    monkeypatch.setattr(
        fi, "use_trtllm_attention", lambda *a, **k: _record(calls, (a, k), True)
    )
    assert b._resolve_trtllm_prefill_attention()
    ((args, kwargs),) = calls
    assert (args[2], args[3]) == (17, 4096)
    assert kwargs["is_prefill"] is True
    assert kwargs["force_use_trtllm"] is expected_force


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("failure_phase", ["none", "reserve", "cleanup"])
def test_reservation_lifecycle(monkeypatch, accel, version, failure_phase):
    """The reservation survives the closing measurement and is released with the
    lease. Whichever step fails first owns the error, and the teardown runs once
    even when it is itself that failure: re-running a half-done one is not safe."""
    ctx = _runner(monkeypatch, accel, version)
    events, refs = [], {}

    class Builder:
        def __init__(self):
            refs["builder"] = weakref.ref(self)

        def prepare_workspace_for_profiling(self, materialize):
            if failure_phase == "reserve":
                raise ValueError("primary workspace error")
            if not materialize:
                current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
                events.append("reserve")
                return
            self.int_workspace = torch.empty(1536, dtype=torch.uint8)
            refs["int"] = weakref.ref(self.int_workspace)
            events.append("materialize")

    def init():
        events.append("init")
        ctx.group(Builder())

    def cleanup():
        events.append("cleanup")
        del ctx.runner.attn_groups
        gc.collect()
        if failure_phase != "none":
            raise RuntimeError("teardown error")

    ctx.install(init, cleanup)

    def reset_peak(device):
        # The profiler's closing measurement still has to see the reservation.
        events.append("reset_peak")
        assert refs["builder"]() is not None and refs["int"]() is not None
        assert current_workspace_manager().workspace_sizes_bytes() == (2048,)

    accel(ctx.module, reset_peak_memory_stats=reset_peak)
    steps = ["init", "reserve", "materialize"]

    with _ws():
        if failure_phase == "none":
            lease = worker_utils.prepare_profiling_workspace(ctx.runner)
            assert events == steps + ["cleanup", "reset_peak"]
            lease.clear()
        else:
            error = ValueError if failure_phase == "reserve" else RuntimeError
            done = steps[:1] if failure_phase == "reserve" else steps
            with pytest.raises(error) as exc_info:
                worker_utils.prepare_profiling_workspace(ctx.runner)
            assert events == done + ["cleanup"]
            if failure_phase == "reserve":
                del exc_info  # its traceback also holds the builder that raised
        gc.collect()
        # Nothing the lease held outlives it: on the cleanup path the traceback
        # above still owns the lease's frame, so only an explicit release clears.
        assert all(ref() is None for ref in refs.values())
        if failure_phase != "reserve":
            # The arena is the manager's, so it outlives the lease.
            assert current_workspace_manager().workspace_sizes_bytes() == (2048,)


@pytest.mark.parametrize("case", ["grew-before", "grew-during"])
def test_reservation_ceiling_is_recorded_once_and_enforced(monkeypatch, accel, case):
    """The ceiling is taken from the first reservation and never exceeded."""
    runner = _runner(monkeypatch, accel, "v1").runner
    calls, sizes = [], iter([2048, 1024, 4096])

    def fake_reserve(_runner):
        calls.append(_runner)
        current_workspace_manager().get_simultaneous(((next(sizes),), torch.uint8))

    monkeypatch.setattr(worker_utils, "reserve_attention_workspace", fake_reserve)
    reserve = worker_utils.reserve_persistent_attention_workspace

    with _ws():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        if case == "grew-before":
            # A ceiling recorded elsewhere, then growth before finalization.
            runner._profiled_persistent_workspace_sizes = (1024,)
            current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
            with pytest.raises(AssertionError, match="profiled size before"):
                reserve(runner)
            assert calls == []
            return

        reserve(runner)
        assert runner._profiled_persistent_workspace_sizes == (2048,)
        reserve(runner)  # a smaller request is fine
        with pytest.raises(AssertionError, match="profiled size during"):
            reserve(runner)


def test_v1_capture_reserves_the_workspace_and_then_locks(monkeypatch, capture):
    from vllm.v1.worker import gpu_model_runner as module

    builder = _Prepare([])
    events, runner = capture(
        module,
        attn_groups=[[SimpleNamespace(metadata_builders=[builder])]],
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(capture_torch_profiler=False)
        ),
        encoder_cudagraph_manager=None,
        _maybe_init_encoder_cudagraph_manager=lambda: None,
        _freeze_gc=_null_context,
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: []),
    )
    builder.events = events
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: SimpleNamespace(local_rank=1),
    )
    for name, fn in (
        ("graph_capture", lambda device: _null_context()),
        ("set_cudagraph_capturing_enabled", lambda _: None),
    ):
        monkeypatch.setattr(module, name, fn)

    assert runner.capture_model() == 1_000
    order = ["prepare:False", "prepare:True", "memory_info", "memory_info", "lock"]
    assert events == order


class _Prepare:
    """A metadata builder that only records the reservation calls it receives."""

    def __init__(self, events):
        self.events, self.reserved = events, False

    def prepare_workspace_for_profiling(self, materialize):
        self.events.append(f"prepare:{materialize}")
        self.reserved = self.reserved or materialize


def test_v2_capture_reserves_before_it_measures_and_then_locks(monkeypatch, capture):
    from vllm.v1.worker.gpu import model_runner as module

    builder = _Prepare([])

    class Manager:
        def needs_capture(self):
            return True

        def capture(self, *args, **kwargs):
            builder.events.append("capture")
            groups = kwargs.get("attn_groups", args[5] if len(args) > 5 else None)
            # The reservation has to be done before the graphs are captured.
            assert groups is not None
            assert groups[0][0].metadata_builders[0].reserved
            return {}

    reserved, allocated = iter([1_000, 1_000, 1_128, 1_128]), iter([500, 500, 628, 628])
    events, runner = capture(
        module,
        accel_extra=(
            ("memory_reserved", lambda d: next(reserved)),
            ("memory_allocated", lambda d: next(allocated)),
        ),
        cudagraph_manager=Manager(),
        attn_groups=[[SimpleNamespace(metadata_builders=[builder])]],
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
    builder.events = events

    assert runner.capture_model() == 1_000
    order = ["prepare:False", "prepare:True", "memory_info", "capture"]
    assert events == order + ["memory_info", "lock"]


def test_v1_graph_profiling_bootstraps_its_own_minimal_kv_cache(monkeypatch):
    events = []
    runner = _stub(
        v1.GPUModelRunner,
        vllm_config=object(),
        _init_minimal_kv_cache_for_profiling=lambda: events.append("init"),
        _cleanup_profiling_kv_cache=lambda: events.append("cleanup"),
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [], cudagraph_keys={}, keys_initialized=True
        ),
        _create_encoder_cudagraph_manager=lambda: None,
        lora_config=None,
        maybe_remove_all_loras=lambda _: None,
    )
    monkeypatch.setattr(v1, "set_current_vllm_config", lambda _: _null_context())

    assert runner.profile_cudagraph_memory() == 0
    assert events == ["init", "cleanup"]
