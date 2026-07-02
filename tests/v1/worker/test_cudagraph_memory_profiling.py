# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


def _attention_spec(head_size: int, head_size_v: int | None = None):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.float16,
    )


class _FakeFlashInferWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor | None = None,
        int_workspace_bytes: int = 1,
    ) -> None:
        self._float_workspace_buffer = (
            float_workspace_buffer
            if float_workspace_buffer is not None
            else torch.empty(1, dtype=torch.uint8)
        )
        self._int_workspace_buffer = torch.empty(
            max(int_workspace_bytes, 1), dtype=torch.uint8
        )
        self._vllm_flashinfer_int_workspace_finalized = False
        self.is_cuda_graph_enabled = False
        self.reset_calls = 0

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self.reset_calls += 1


def _make_flashinfer_builder(flashinfer_backend):
    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    return builder


def test_flashinfer_separate_cudagraph_memory_profile_gate():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    assert not FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(256)
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(512)
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(256, head_size_v=512)
    )

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": _attention_spec(256),
            "layer.1": _attention_spec(512),
        },
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, uniform_spec
    )


def test_flashinfer_workspace_buffer_uses_workspace_manager():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first_builder = _make_flashinfer_builder(flashinfer_backend)
        first_state = first_builder.get_workspace_buffer_state()
        first = first_builder._get_workspace_buffer(
            first_builder._native_initial_workspace_buffer_size()
        )

        second_builder = _make_flashinfer_builder(flashinfer_backend)
        second_builder.set_workspace_buffer_state(first_state)
        second = second_builder._get_workspace_buffer(
            second_builder._native_initial_workspace_buffer_size()
        )

        assert first.device.type == "cpu"
        assert first.dtype == torch.uint8
        assert first.numel() == 1
        assert first.data_ptr() == second.data_ptr()
    finally:
        reset_workspace_manager()


def test_flashinfer_workspace_buffer_growth_resets_registered_wrappers():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        wrapper = _FakeFlashInferWrapper(
            builder._get_workspace_buffer(
                builder._native_initial_workspace_buffer_size()
            )
        )
        builder._register_workspace_wrapper(wrapper)
        builder._ensure_flashinfer_wrapper_workspace(wrapper, WorkspaceSizes(1024, 16))

        assert builder._workspace_buffer.numel() == 1024
        assert wrapper._float_workspace_buffer.data_ptr() == (
            builder._workspace_buffer.data_ptr()
        )
        assert wrapper._float_workspace_buffer.numel() == 1024
        assert wrapper._int_workspace_buffer.numel() == 16
        reset_calls = wrapper.reset_calls
        assert reset_calls >= 1

        builder._ensure_flashinfer_wrapper_workspace(wrapper, WorkspaceSizes(1024, 16))
        assert wrapper.reset_calls == reset_calls

        wrapper_ref = weakref.ref(wrapper)
        del wrapper
        gc.collect()

        builder._workspace_state.set_buffer(torch.empty(2048, dtype=torch.uint8))
        assert wrapper_ref() is None
        assert builder._workspace_state.wrappers == []
    finally:
        reset_workspace_manager()


def test_flashinfer_int_workspace_is_per_wrapper():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first = _FakeFlashInferWrapper()
        second = _FakeFlashInferWrapper()

        builder._ensure_flashinfer_wrapper_workspace(first, WorkspaceSizes(1024, 32))
        builder._ensure_flashinfer_wrapper_workspace(second, WorkspaceSizes(1024, 32))

        assert first._float_workspace_buffer.data_ptr() == (
            second._float_workspace_buffer.data_ptr()
        )
        assert first._int_workspace_buffer.data_ptr() != (
            second._int_workspace_buffer.data_ptr()
        )
        assert first._int_workspace_buffer.numel() == 32
        assert second._int_workspace_buffer.numel() == 32
    finally:
        reset_workspace_manager()


def test_flashinfer_finalized_int_workspace_cannot_grow():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8)
    wrapper._vllm_flashinfer_int_workspace_finalized = True

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        with pytest.raises(AssertionError, match="int workspace is finalized"):
            builder._ensure_flashinfer_wrapper_workspace(
                wrapper, WorkspaceSizes(1024, 16)
            )
    finally:
        reset_workspace_manager()


def test_flashinfer_non_cudagraph_int_workspace_can_grow(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8)
    warnings = []

    monkeypatch.setattr(
        flashinfer_backend.logger,
        "warning",
        lambda msg, *args: warnings.append(msg),
    )

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        builder._ensure_flashinfer_wrapper_workspace(wrapper, WorkspaceSizes(1024, 8))
        builder._ensure_flashinfer_wrapper_workspace(wrapper, WorkspaceSizes(1024, 16))

        assert wrapper._int_workspace_buffer.numel() == 16
        assert wrapper.reset_calls == 2
        assert any("Growing FlashInfer int workspace" in msg for msg in warnings)
    finally:
        reset_workspace_manager()


def test_flashinfer_reserves_prefill_tail_workspace(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.model_config = SimpleNamespace(max_model_len=1024, dtype=torch.float16)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=4,
        ),
        speculative_config=None,
    )
    builder.q_data_type = torch.float16
    builder.kv_cache_dtype = torch.uint8
    builder.page_size = 16
    builder.window_left = -1
    builder.prefill_fixed_split_size = -1
    builder.disable_split_kv = False

    class FakeWrapper:
        pass

    ensured = []
    observed_query_lens = []

    def fake_workspace_size(**kwargs):
        qo_indptr = kwargs["qo_indptr_cpu"]
        query_lens = torch.diff(qo_indptr).tolist()
        observed_query_lens.extend(query_lens)
        return WorkspaceSizes(4096, 64) if query_lens == [3] else WorkspaceSizes(0, 0)

    monkeypatch.setattr(
        builder,
        "_get_prefill_workspace_size_func",
        lambda **kwargs: ("fa2", object()),
    )
    monkeypatch.setattr(builder, "_call_prefill_workspace_size", fake_workspace_size)
    monkeypatch.setattr(
        builder, "_get_prefill_wrapper", lambda causal=True: FakeWrapper()
    )
    monkeypatch.setattr(
        builder,
        "_ensure_flashinfer_wrapper_workspace",
        lambda wrapper, size: ensured.append(size),
    )
    monkeypatch.setattr(
        builder,
        "_reserve_decode_wrapper_workspace",
        lambda **kwargs: WorkspaceSizes(0, 0),
    )
    builder.enable_cuda_graph = False

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        assert builder.reserve_workspace_for_cudagraph_capture() == 4160
    finally:
        reset_workspace_manager()

    assert ensured == [WorkspaceSizes(4096, 64)]
    assert 3 in observed_query_lens
    assert 8 in observed_query_lens


def test_flashinfer_reserves_decode_cudagraph_int_workspace(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    builder.decode_fixed_split_size = -1
    builder.disable_split_kv = False

    wrapper = _FakeFlashInferWrapper()
    wrapper.is_cuda_graph_enabled = True

    monkeypatch.setattr(builder, "_get_decode_wrapper", lambda *args: wrapper)
    monkeypatch.setattr(
        builder,
        "_get_decode_workspace_size",
        lambda **kwargs: WorkspaceSizes(128, 32),
    )

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        sizes = builder._reserve_decode_wrapper_workspace(
            batch_size=4,
            num_pages=8,
            last_page_len=16,
            use_cudagraph=True,
        )
    finally:
        reset_workspace_manager()

    assert sizes == WorkspaceSizes(128, 32)
    assert wrapper._float_workspace_buffer.numel() == 128
    assert wrapper._int_workspace_buffer.numel() == 32
    assert wrapper.reset_calls == 1
    assert wrapper._vllm_flashinfer_int_workspace_finalized


def test_flashinfer_workspace_query_len_candidates():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    candidates = (
        flashinfer_backend.FlashInferMetadataBuilder._get_workspace_query_len_candidates
    )

    assert candidates(8) == list(range(1, 9))

    large_candidates = candidates(1024)
    assert 1 in large_candidates
    assert 256 in large_candidates
    assert 512 in large_candidates
    assert 1024 in large_candidates
    assert 257 not in large_candidates


def test_flashinfer_nvfp4_slot_mapping_symbol_available():
    flashinfer = pytest.importorskip("flashinfer")
    assert hasattr(
        flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
    )


def test_separate_profile_accounts_persistent_and_graph_pool(monkeypatch):
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.gpu_model_runner import CUDAGraphMode, GPUModelRunner

    class FakeWrapper:
        _all_instances = []

        @staticmethod
        def clear_all_graphs():
            pass

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.lora_config = None
    runner.cudagraph_dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [
            (
                CUDAGraphMode.PIECEWISE,
                [
                    SimpleNamespace(
                        num_tokens=128,
                        uniform=False,
                        num_active_loras=0,
                    ),
                    SimpleNamespace(
                        num_tokens=64,
                        uniform=False,
                        num_active_loras=0,
                    ),
                    SimpleNamespace(
                        num_tokens=32,
                        uniform=False,
                        num_active_loras=0,
                    ),
                ],
            ),
            (
                CUDAGraphMode.FULL,
                [
                    SimpleNamespace(
                        num_tokens=80,
                        uniform=False,
                        num_active_loras=0,
                    ),
                    SimpleNamespace(
                        num_tokens=40,
                        uniform=False,
                        num_active_loras=0,
                    ),
                ],
            ),
        ],
        cudagraph_keys={},
        keys_initialized=True,
    )

    warmup_calls = []
    capture_calls = []
    cleanup_calls = []

    runner.max_model_len = 4096
    runner.max_num_tokens = 128
    runner._init_minimal_kv_cache_for_profiling = lambda: None
    runner._requires_separate_cudagraph_memory_profiling = lambda: True
    runner._create_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = null_context
    runner._cleanup_profiling_kv_cache = lambda: cleanup_calls.append("cleanup")
    runner.maybe_remove_all_loras = lambda lora_config: None
    runner._reserve_attention_workspace_for_cudagraph_capture = lambda: 200
    runner._warmup_before_cudagraph_capture = lambda *args, **kwargs: (
        warmup_calls.append((args[0], kwargs))
    )
    runner._warmup_and_capture = lambda *args, **kwargs: (
        capture_calls.append((args[0], kwargs))
    )

    memory_reserved_values = iter([1_000, 1_600])
    get_memory_info_values = iter(
        [
            (10_000_000, 0),
            (8_000_000, 0),
            (8_000_000, 0),
            (6_500_000, 0),
            (6_500_000, 0),
            (4_100_000, 0),
            (4_100_000, 0),
            (3_000_000, 0),
        ]
    )

    monkeypatch.setattr(gpu_model_runner, "CUDAGraphWrapper", FakeWrapper)
    monkeypatch.setattr(gpu_model_runner, "BreakableCUDAGraphWrapper", FakeWrapper)
    monkeypatch.setattr(
        gpu_model_runner,
        "set_current_vllm_config",
        lambda *args, **kwargs: null_context(),
    )
    monkeypatch.setattr(
        gpu_model_runner, "graph_capture", lambda *args, **kwargs: null_context()
    )
    monkeypatch.setattr(
        gpu_model_runner,
        "set_cudagraph_capturing_enabled",
        lambda enabled: None,
    )
    monkeypatch.setattr(
        gpu_model_runner.current_platform,
        "graph_pool_handle",
        lambda: object(),
    )
    monkeypatch.setattr(gpu_model_runner.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(gpu_model_runner.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        gpu_model_runner.torch.accelerator,
        "memory_reserved",
        lambda device: next(memory_reserved_values),
    )
    monkeypatch.setattr(
        gpu_model_runner.torch.accelerator,
        "get_memory_info",
        lambda: next(get_memory_info_values),
    )

    estimate = runner.profile_cudagraph_memory()

    assert estimate == 6_500_800
    assert runner.cudagraph_memory_persistent_estimate == 800
    assert runner.cudagraph_memory_graph_pool_estimate == 6_500_000
    assert [call[0].num_tokens for call in warmup_calls] == [128, 64, 80, 40]
    assert [call[0].num_tokens for call in capture_calls] == [128, 64, 80, 40]
    assert all(call[1]["num_warmups"] == 0 for call in capture_calls)
    assert warmup_calls[2][1]["profile_seq_lens"] == 1
    assert warmup_calls[3][1]["profile_seq_lens"] is None
    assert cleanup_calls == ["cleanup"]


def test_v2_profile_accounts_attention_workspace(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()

    events = []

    monkeypatch.setattr(
        gpu_model_runner_v2,
        "set_current_vllm_config",
        lambda *args, **kwargs: null_context(),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.current_platform,
        "is_cuda",
        lambda: True,
    )

    def reserve_attention_workspace():
        events.append("reserve")
        return 4096

    runner._init_minimal_kv_cache_for_profiling = lambda: events.append("init")
    runner._reserve_attention_workspace_for_cudagraph_capture = (
        reserve_attention_workspace
    )

    def profile_cudagraph_memory_graph_pool():
        events.append("graph_pool")
        return 2048

    runner._profile_cudagraph_memory_graph_pool = profile_cudagraph_memory_graph_pool
    runner._cleanup_profiling_kv_cache = lambda: events.append("cleanup")

    estimate = runner.profile_cudagraph_memory()

    assert estimate == 6144
    assert runner.cudagraph_memory_persistent_estimate == 4096
    assert runner.cudagraph_memory_graph_pool_estimate == 2048
    assert events == ["init", "reserve", "graph_pool", "cleanup"]


def test_v2_profile_is_noop_on_non_cuda(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)

    monkeypatch.setattr(
        gpu_model_runner_v2.current_platform,
        "is_cuda",
        lambda: False,
    )
    runner._init_minimal_kv_cache_for_profiling = pytest.fail

    assert runner.profile_cudagraph_memory() == 0
    assert runner.cudagraph_memory_persistent_estimate == 0
    assert runner.cudagraph_memory_graph_pool_estimate == 0


def test_v2_cuda_graph_pool_sample_uses_peak(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cuda:0")
    events = []

    get_memory_info_values = iter([(10_000, 0), (9_950, 0)])
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "synchronize",
        lambda: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "get_memory_info",
        lambda: next(get_memory_info_values),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_reserved",
        lambda device: 1_000,
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_allocated",
        lambda device: 500,
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "reset_peak_memory_stats",
        lambda device: events.append("reset_peak"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "max_memory_reserved",
        lambda device: 1_120,
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "max_memory_allocated",
        lambda device: 530,
    )

    assert (
        runner._measure_cuda_graph_pool_sample(lambda: events.append("capture")) == 120
    )
    assert events == [
        "sync",
        "empty_cache",
        "reset_peak",
        "capture",
        "sync",
    ]


def test_v2_graph_pool_profile_restores_capture_state(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.model_runner import CUDAGraphMode, GPUModelRunner

    piecewise_descs = [
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 128, None),
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 64, None),
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 32, None),
    ]
    full_descs = [
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 80, 4),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 40, 2),
    ]
    original_graph = object()
    original_breakable_entry = object()
    breakable_runner = SimpleNamespace(
        graph_pool="breakable-original",
        entries={"old": original_breakable_entry},
    )

    class FakeCudaGraphManager:
        def __init__(self):
            self.pool = "manager-original"
            self.graphs = {"old": original_graph}
            self._graphs_captured = False
            self.hidden_states = "hidden-original"
            self.aux_hidden_states = ["aux-original"]
            self.intermediate_tensors = "intermediate-original"
            self.use_aux_hidden_state_outputs = False
            self.use_breakable_cg = True
            self.breakable_cg_runner = breakable_runner

        def get_capture_descs(self):
            return [
                (CUDAGraphMode.PIECEWISE, piecewise_descs),
                (CUDAGraphMode.FULL, full_descs),
            ]

        def init_breakable_cg_runner(self, model):
            assert model == "model"

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = "model"
    runner.cudagraph_manager = FakeCudaGraphManager()
    sample_values = iter([1_000, 2_000_000, 800, 3_000_000])
    capture_calls = []

    monkeypatch.setattr(
        gpu_model_runner_v2.current_platform,
        "graph_pool_handle",
        lambda: "profile-pool",
    )
    empty_cache_calls = []
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "empty_cache",
        lambda: empty_cache_calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.compilation_counter,
        "num_cudagraph_captured",
        11,
    )
    monkeypatch.setattr(
        runner,
        "_measure_cuda_graph_pool_sample",
        lambda capture_fn: capture_fn() or next(sample_values),
    )

    def capture_model_cudagraphs(**kwargs):
        capture_calls.append(kwargs)
        assert kwargs["capture_speculator"] is False
        assert runner.cudagraph_manager.pool == "profile-pool"
        assert breakable_runner.graph_pool == "profile-pool"
        runner.cudagraph_manager.graphs["profile"] = object()
        runner.cudagraph_manager._graphs_captured = True
        runner.cudagraph_manager.hidden_states = "hidden-profile"
        runner.cudagraph_manager.aux_hidden_states = ["aux-profile"]
        runner.cudagraph_manager.intermediate_tensors = "intermediate-profile"
        runner.cudagraph_manager.use_aux_hidden_state_outputs = True
        breakable_runner.entries["profile"] = object()
        gpu_model_runner_v2.compilation_counter.num_cudagraph_captured = 99

    monkeypatch.setattr(
        runner,
        "_capture_model_cudagraphs",
        capture_model_cudagraphs,
    )

    estimate = runner._profile_cudagraph_memory_graph_pool()

    assert estimate == 7_001_000
    assert [call["capture_descs"] for call in capture_calls] == [
        {CUDAGraphMode.PIECEWISE: [piecewise_descs[0]]},
        {CUDAGraphMode.PIECEWISE: [piecewise_descs[1]]},
        {CUDAGraphMode.FULL: [full_descs[0]]},
        {CUDAGraphMode.FULL: [full_descs[1]]},
    ]
    assert runner.cudagraph_manager.pool == "manager-original"
    assert runner.cudagraph_manager.graphs == {"old": original_graph}
    assert runner.cudagraph_manager._graphs_captured is False
    assert runner.cudagraph_manager.hidden_states == "hidden-original"
    assert runner.cudagraph_manager.aux_hidden_states == ["aux-original"]
    assert runner.cudagraph_manager.intermediate_tensors == "intermediate-original"
    assert runner.cudagraph_manager.use_aux_hidden_state_outputs is False
    assert breakable_runner.graph_pool == "breakable-original"
    assert breakable_runner.entries == {"old": original_breakable_entry}
    assert gpu_model_runner_v2.compilation_counter.num_cudagraph_captured == 11
    assert empty_cache_calls == ["empty_cache"]


def test_v2_cleanup_profiling_kv_cache_releases_builder_refs(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    class Builder:
        pass

    builder = Builder()
    builder_ref = weakref.ref(builder)
    layer = SimpleNamespace(
        kv_cache=torch.empty(1),
        impl=SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object()),
    )

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=4)
    runner.kv_caches = [torch.empty(1)]
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    runner.kv_cache_config = object()
    runner.block_tables = object()
    runner.kernel_block_sizes = [16]
    runner.cudagraph_manager = object()
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": layer})
    del builder

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "empty_cache", lambda: None
    )

    runner._cleanup_profiling_kv_cache()
    gc.collect()

    assert runner.kv_caches == []
    assert not hasattr(runner, "attn_groups")
    assert not hasattr(runner, "kv_cache_config")
    assert not hasattr(runner, "block_tables")
    assert not hasattr(runner, "kernel_block_sizes")
    assert not hasattr(runner, "cudagraph_manager")
    assert runner.cache_config.num_gpu_blocks is None
    assert isinstance(layer.kv_cache, torch.Tensor)
    assert layer.kv_cache.numel() == 0
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    assert builder_ref() is None


def test_v2_capture_reserves_workspace_before_measurement_and_locks(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    class Builder:
        reserved = False

        def reserve_workspace_for_cudagraph_capture(self):
            events.append("builder_reserve")
            self.reserved = True

    class FakeCudaGraphManager:
        def needs_capture(self):
            return True

        def capture(
            self,
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            **kwargs,
        ):
            events.append("capture")
            assert attn_groups[0][0].metadata_builders[0].reserved
            return {}

    events = []
    builder = Builder()
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.cudagraph_manager = FakeCudaGraphManager()
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda lora_config: null_context()
    runner.model = object()
    runner.model_state = object()
    runner.input_buffers = object()
    runner.intermediate_tensors = None
    runner.block_tables = object()
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    runner.kv_cache_config = object()
    runner.use_aux_hidden_state_outputs = False
    runner.speculator = None

    memory_reserved_values = iter([1_000, 1_128])
    get_memory_info_values = iter([(10_000, 0), (9_000, 0)])

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "empty_cache", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_reserved",
        lambda device: next(memory_reserved_values),
    )

    def get_memory_info():
        events.append("memory_info")
        return next(get_memory_info_values)

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "get_memory_info",
        get_memory_info,
    )
    monkeypatch.setattr(
        gpu_model_runner_v2,
        "lock_workspace",
        lambda: events.append("lock"),
    )

    assert runner.capture_model() == 1_000
    assert events == [
        "builder_reserve",
        "memory_info",
        "capture",
        "memory_info",
        "lock",
    ]
