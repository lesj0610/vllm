# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import vllm.utils.jit_monitor as jit_monitor
import vllm.v1.worker.gpu_worker as gpu_worker_module
from tests.utils import create_new_process_for_each_test
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import Worker, maybe_rocm_profiling_fallback
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)
from vllm.v1.worker.utils import requires_persistent_attention_workspace_profiling
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.mark.parametrize("profile_persistent_workspace", [False, True])
def test_initialize_kv_cache_finalizes_persistent_workspace(
    monkeypatch, profile_persistent_workspace
):
    events = []
    worker = object.__new__(Worker)
    worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker.vllm_config = object()
    worker.model_config = SimpleNamespace(enable_return_routed_experts=False)

    @contextmanager
    def pool(tag):
        events.append(f"enter {tag}")
        yield
        events.append(f"exit {tag}")

    worker._maybe_get_memory_pool_context = pool
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda config, **kw: events.append("initialize_kv_cache")
    )
    # kv_cache_layout is resolved by the engine core; None keeps what we have.
    config = SimpleNamespace(
        num_blocks=8, needs_kv_cache_zeroing=False, kv_cache_layout=None
    )
    monkeypatch.setattr(gpu_worker, _RESERVE, lambda r: events.append("reserve"))
    for name, fn in (
        ("ensure_kv_transfer_initialized", lambda *a, **k: events.append("connector")),
        (_GATE, lambda config: profile_persistent_workspace),
    ):
        monkeypatch.setattr(gpu_worker_module, name, fn)

    worker.initialize_from_config(config)

    assert worker.cache_config.num_gpu_blocks == 8
    # The reservation is KV-init state, so it shares the runtime pool.
    expected = ["connector", "enter runtime", "initialize_kv_cache"]
    if profile_persistent_workspace:
        expected.append("reserve")
    assert events == [*expected, "exit runtime"]


def _record(sink, item, result=None):
    sink.append(item)
    return result


_GATE = "requires_persistent_attention_workspace_profiling"
_RESERVE = "reserve_persistent_attention_workspace"
_WARMUP_COMPILATION = SimpleNamespace(
    mode=CompilationMode.NONE,
    backend="eager",
    compilation_time=0.0,
    encoder_compilation_time=0.0,
)
_WARMUP_STATE = dict(
    vllm_config=SimpleNamespace(
        compilation_config=_WARMUP_COMPILATION,
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    ),
    compilation_config=_WARMUP_COMPILATION,
    cache_config=SimpleNamespace(kv_cache_memory_bytes=1),
    observability_config=SimpleNamespace(
        jit_monitor_mode="warn", jit_monitor_verbose=False
    ),
    scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
    device=torch.device("cpu"),
    execute_model=None,
    sample_tokens=None,
    # capture_model() runs inside a profiler context that reads worker state
    # this test has no reason to own; what is under test is the order around
    # that call, so the context is emptied rather than imitated.
    _get_cudagraph_capture_context=nullcontext,
)
_WARMUP_RUNNER = dict(
    lora_config=None,
    maybe_remove_all_loras=lambda cfg: None,
    _profiled_persistent_workspace_sizes=(1024,),
    is_pooling_model=False,
    _dummy_run=lambda **kwargs: (None, None),
)
_WARMUP_NOOPS = (
    "kernel_warmup",
    "set_random_seed",
    "freeze_gc_heap",
    "maybe_attach_gc_debug_callback",
    "enable_gpu_sync_check",
    "set_torch_threads_for_runtime",
)


def _warmup_worker(
    monkeypatch, *, events, arena, grow_to, enforce_eager, use_v2, gated
):
    """A ``Worker`` stripped to what ``compile_or_warm_up_model`` touches."""

    def note(name, grow=False):
        return lambda *a, **k: (
            events.append(name),
            grow and arena.get_simultaneous(((grow_to,), torch.uint8)),
        )

    worker = object.__new__(Worker)
    worker.__dict__.update(_WARMUP_STATE)
    worker.model_config = SimpleNamespace(enforce_eager=enforce_eager, seed=0)
    worker.use_v2_model_runner = use_v2
    # V1's eager sampler warmup is the last step that can size a workspace.
    worker.model_runner = SimpleNamespace(
        **_WARMUP_RUNNER,
        capture_model=lambda: _record(events, "capture", 0),
        _dummy_sampler_run=note("v1_sampler", grow=True),
    )
    patches = {
        "current_workspace_manager": lambda: arena,
        "warmup_kernels": note("warmup_kernels", grow=True),
        "lock_workspace": note("lock"),
        "is_workspace_manager_initialized": lambda: True,
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=True),
        _GATE: lambda config: gated,
        **dict.fromkeys(_WARMUP_NOOPS, lambda *a, **k: None),
    }
    for name, fn in patches.items():
        monkeypatch.setattr(gpu_worker_module, name, fn)
    monkeypatch.setattr(gpu_worker, _RESERVE, note("reserve"))
    monkeypatch.setattr(jit_monitor, "activate", lambda **kw: None)  # lazy import
    return worker


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("grows_during_warmup", [False, True])
@pytest.mark.parametrize("use_v2", [False, True])
def test_warmup_locks_the_workspace_when_profiling_is_enabled(
    monkeypatch, enforce_eager, gated, grows_during_warmup, use_v2
):
    """Reserve, warm up, reject growth past the ceiling, then lock.

    capture_model() locks when it captures, but it returns early when both
    capture modes are disabled and is skipped under enforce_eager, so an
    opted-in model needs the lock on the warmup path too. A model that did not
    opt in never reserved anything, so locking would turn a later lazy
    allocation into a hard failure instead of leaving it on the legacy path.
    """
    events: list[str] = []
    arena = WorkspaceManager(torch.device("cpu"))
    arena.get_simultaneous(((1024,), torch.uint8))  # grow-only, so 1024 is a no-op
    worker = _warmup_worker(
        monkeypatch,
        events=events,
        arena=arena,
        grow_to=2048 if grows_during_warmup else 1024,
        enforce_eager=enforce_eager,
        use_v2=use_v2,
        gated=gated,
    )
    grower = "warmup_kernels" if use_v2 else "v1_sampler"

    if gated and grows_during_warmup:
        with pytest.raises(AssertionError, match="before workspace lock"):
            worker.compile_or_warm_up_model()
        assert "lock" not in events
        assert events.index("reserve") < events.index(grower)
        return

    worker.compile_or_warm_up_model()
    assert grower in events
    if enforce_eager:
        assert "capture" not in events
    if gated:
        assert events.index("reserve") < events.index(grower)
        assert events.index("lock") == len(events) - 1
    else:
        assert "reserve" not in events and "lock" not in events


_SHIPPED_SPEC = FullAttentionSpec(
    block_size=16, num_kv_heads=1, head_size=128, dtype=torch.float16
)
_GATE_CASES = [
    # supports, speculative, elastic_ep, expected
    ([True], False, False, True, "single-required"),
    ([True, True], False, False, True, "all-required"),
    ([True, False], False, False, True, "required-with-neutral"),
    ([True, None], False, False, False, "unsupported-vetoes-required"),
    ([False], False, False, False, "neutral-only"),
    ([object()], False, False, False, "unknown-fails-closed"),
    ([], False, False, False, "no-builders"),
    ([True], True, False, False, "speculative-fallback"),
    ([True], False, True, False, "elastic-ep-fallback"),
    ("shipped", False, False, True, "flashinfer-with-gdn"),
]


@pytest.mark.parametrize(
    ("supports", "speculative", "elastic_ep", "expected"),
    [pytest.param(*case[:4], id=case[4]) for case in _GATE_CASES],
)
def test_persistent_workspace_gate(
    monkeypatch, supports, speculative, elastic_ep, expected
):
    """One ``None`` vetoes the model; at least one ``True`` is required."""
    spec: Any = object()
    if supports == "shipped":
        pytest.importorskip("flashinfer")
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder as F
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder as G

        supports, spec = [F, G], _SHIPPED_SPEC

    def layer(support):
        stub = SimpleNamespace(
            persistent_workspace_profiling_support=lambda cfg, spec: support
        )
        cls = support if isinstance(support, type) else stub
        return SimpleNamespace(
            get_kv_cache_spec=lambda cfg: spec,
            get_attn_backend=lambda: SimpleNamespace(get_builder_cls=lambda: cls),
        )

    monkeypatch.setattr(
        "vllm.v1.worker.utils.get_layers_from_vllm_config",
        lambda cfg, layer_type: {f"l{i}": layer(s) for i, s in enumerate(supports)},
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=False),
        speculative_config=object() if speculative else None,
        parallel_config=SimpleNamespace(
            enable_elastic_ep=elastic_ep, decode_context_parallel_size=1
        ),
    )
    assert requires_persistent_attention_workspace_profiling(config) is expected


def test_load_model_preserves_compiled_graphs_at_runtime(monkeypatch):
    """Profiling must use serving's thread count to keep Dynamo guards valid."""
    from torch._dynamo.testing import CompileCounter

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(gpu_worker, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        gpu_worker, "set_current_vllm_config", lambda config: nullcontext()
    )
    loading_threads = []
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(weight_transfer_config=None),
        model_runner=SimpleNamespace(
            load_model=lambda **kwargs: loading_threads.append(torch.get_num_threads())
        ),
        _maybe_get_memory_pool_context=lambda **kwargs: nullcontext(),
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        gpu_worker.Worker.load_model(worker)
        assert loading_threads == [2]

        counter = CompileCounter()
        compiled = torch.compile(lambda x: x + 1, backend=counter, fullgraph=True)
        x = torch.ones(2)
        compiled(x)
        gpu_worker.set_torch_threads_for_runtime()
        torch.testing.assert_close(compiled(x), x + 1)
        assert counter.frame_count == 1
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not torch.accelerator.is_available()
    or torch.cuda.memory.get_allocator_backend() != "native",
    reason="needs the native CUDA or ROCm allocator",
)
# A fresh allocator: blocks cached by earlier tests could serve the large buffer.
@create_new_process_for_each_test("spawn")
def test_scoped_max_split_keeps_freed_large_blocks_releasable():
    """A small allocation made after a large buffer is freed must not pin the
    buffer's segment: the profiling run (determine_available_memory) grows
    workspaces this way, and a pinned segment survives empty_cache() and is
    counted as consumed memory."""
    large = 512 * 1024 * 1024
    small = 2 * 1024 * 1024  # large pool, so it is served by splitting cached blocks

    def reserved_while_small_is_live(scope) -> int:
        torch.accelerator.empty_cache()
        with scope:
            buf = torch.empty(large, dtype=torch.uint8, device="cuda")
            del buf
            tensor = torch.empty(small, dtype=torch.uint8, device="cuda")
            torch.accelerator.empty_cache()
            reserved = torch.accelerator.memory_reserved()
            del tensor
        torch.accelerator.empty_cache()
        return reserved

    baseline = torch.accelerator.memory_reserved()
    # Without the limit the small tensor is split off the freed block and pins it.
    assert reserved_while_small_is_live(nullcontext()) - baseline >= large
    scoped = gpu_worker.Worker._scoped_allocator_max_split(
        SimpleNamespace(), max_split_size_mb=20
    )
    assert reserved_while_small_is_live(scoped) - baseline < large


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not torch.accelerator.is_available()
    or torch.cuda.memory.get_allocator_backend() != "native",
    reason="needs the native CUDA or ROCm allocator",
)
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("snapshot_fallback", [False, True])
@pytest.mark.parametrize("suffix", ["", ", "])
def test_scoped_max_split_preserves_allocator_settings(
    monkeypatch, fail, snapshot_fallback, suffix
):
    """Preserve allocator settings across successful and failed profiling runs."""
    if snapshot_fallback:
        monkeypatch.delattr(
            torch._C, "_accelerator_getAllocatorSettings", raising=False
        )

    def settings():
        return torch.cuda.memory._snapshot()["allocator_settings"]

    original = settings()["PYTORCH_CUDA_ALLOC_CONF"]
    configured = (
        "max_split_size_mb:128,garbage_collection_threshold:0.8,"
        "roundup_power2_divisions:[256:1,512:2,>:4],max_non_split_rounding_mb:32"
    )
    try:
        torch._C._accelerator_setAllocatorSettings(configured + suffix)
        before = settings()
        expected_error = pytest.raises(RuntimeError, match="profiling failed")
        with (
            expected_error if fail else nullcontext(),
            gpu_worker.Worker._scoped_allocator_max_split(SimpleNamespace(), 20),
        ):
            scoped = settings()
            assert scoped["max_split_size"] == 20 * 1024 * 1024
            for key in ("garbage_collection_threshold", "roundup_power2_divisions"):
                assert scoped[key] == before[key]
            if fail:
                raise RuntimeError("profiling failed")
        assert settings() == before
    finally:
        torch._C._accelerator_setAllocatorSettings(original)


def test_scoped_max_split_ignores_async_allocator(monkeypatch):
    """Async allocators ignore max_split and may not support memory snapshots."""
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync"
    )
    with (
        patch.object(torch.cuda.memory, "_snapshot") as snapshot,
        patch.object(torch._C, "_accelerator_setAllocatorSettings") as set_settings,
        gpu_worker.Worker._scoped_allocator_max_split(SimpleNamespace(), 20),
    ):
        pass
    snapshot.assert_not_called()
    set_settings.assert_not_called()


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_rejects_stale_schema(plan_env, monkeypatch):
    worker = _plan_worker()
    fingerprint = startup_plan.compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    maybe_save_startup_plan(worker, 50 * GiB_bytes)
    monkeypatch.setattr(
        startup_plan, "PLAN_SCHEMA_VERSION", startup_plan.PLAN_SCHEMA_VERSION + 1
    )

    assert startup_plan._load_plan(fingerprint) is None


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Memory accounting of the profiling run (Worker.determine_available_memory).

# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None


class _OrderedHandle:
    """Send handle that logs when it is waited."""

    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def is_completed(self) -> bool:
        return True

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def test_execute_model_waits_previous_pp_send_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Previous device handles are waited before the forward pass; the
    metadata handle is left to the GroupCoordinator's reaper."""
    import torch

    from vllm.sequence import IntermediateTensors

    log: list[str] = []
    previous_tensor_send = _OrderedHandle(log, "prev-tensor")
    metadata_handle = _OrderedHandle(log, "meta")
    tensor_handle = _OrderedHandle(log, "tensor")

    def isend_tensor_dict(tensors, all_gather_group=None, all_gather_tensors=None):
        log.append("isend")
        return [metadata_handle, tensor_handle]

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=False,
        isend_tensor_dict=isend_tensor_dict,
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace())

    def run_model(scheduler_output, intermediate_tensors):
        log.append("forward")
        return IntermediateTensors({"hidden_states": torch.zeros(1)})

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=run_model),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[previous_tensor_send],
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 4}
    )

    assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    assert log == ["wait:prev-tensor", "forward", "isend"]
    assert worker._pp_send_work == [tensor_handle]


def test_jit_monitor_activation_follows_enable_jit_warmup(
    monkeypatch: pytest.MonkeyPatch,
):
    """The post-warmup JIT monitor must stay off when JIT warmup is disabled
    (e.g. by enforce_eager): runtime compilation is then expected, and
    warning/erroring on it would be noise."""
    from vllm.utils import jit_monitor

    calls = []
    monkeypatch.setattr(jit_monitor, "activate", lambda **kwargs: calls.append(kwargs))

    def worker(enable_jit_warmup):
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup)
            ),
            observability_config=SimpleNamespace(
                jit_monitor_mode="warn", jit_monitor_verbose=False
            ),
        )

    gpu_worker.Worker._maybe_activate_jit_monitor(worker(True))
    assert calls == [{"mode": "warn", "verbose": False}]

    calls.clear()
    gpu_worker.Worker._maybe_activate_jit_monitor(worker(False))
    assert calls == []


_LEASE_STATE = dict(
    vllm_config=SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    ),
    cache_config=SimpleNamespace(kv_cache_memory_bytes=0, gpu_memory_utilization=0.9),
    model_config=SimpleNamespace(multimodal_config=None),
    parallel_config=SimpleNamespace(),
    init_snapshot=SimpleNamespace(
        free_memory=ANY_FREE_MEMORY, total_memory=ANY_FREE_MEMORY
    ),
    requested_memory=ANY_FREE_MEMORY,
    _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
)
_LEASE_NOOPS = (
    ("maybe_apply_startup_plan", lambda w: None),
    (_GATE, lambda config: True),
    ("maybe_rocm_profiling_fallback", lambda result: None),
    ("current_platform", SimpleNamespace(is_cuda_alike=lambda: True)),
    ("reserve_mm_ipc_gpu_memory", lambda *args: 0),
)


def test_profiling_lease_is_held_across_the_closing_measurement(monkeypatch):
    """`memory_profiling` measures on the way out of the block, so the lease has
    to still be held there and released before CUDA graph profiling runs."""
    held, events = [torch.empty(64, dtype=torch.uint8)], []

    @contextmanager
    def fake_memory_profiling(snapshot, weights_memory=0):
        result = _profile_result(4 * GiB_bytes)
        result.non_kv_cache_memory = 4 * GiB_bytes
        yield result
        events.append(("closing_measurement", bool(held)))

    worker = object.__new__(Worker)
    worker.__dict__.update(_LEASE_STATE)
    worker.model_runner = SimpleNamespace(
        model_memory_usage=0,
        profile_run=lambda randomize_inputs=False: None,
        profile_cudagraph_memory=lambda: _record(
            events, ("cudagraph_profiling", bool(held)), 0
        ),
    )
    for name, fn in _LEASE_NOOPS + (
        ("prepare_profiling_workspace", lambda runner: held),
        ("memory_profiling", fake_memory_profiling),
    ):
        monkeypatch.setattr(gpu_worker_module, name, fn)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    worker.determine_available_memory()
    assert events == [("closing_measurement", True), ("cudagraph_profiling", False)]
