# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import msgspec
import pytest
import torch

import vllm.envs as envs
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.v1.ple_offload import worker as ple_offload_worker
from vllm.v1.ple_offload.connector import PleOffloadConnector
from vllm.v1.worker.gpu_worker import Worker


class _TestPleOffloadLayer(PleOffloadLayer):
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        del hidden_states, args, kwargs
        return input_ids.unsqueeze(-1)


class _WeightLoadingPleLayer(_TestPleOffloadLayer):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.bias = torch.nn.Parameter(torch.zeros(2))


class _DummyMetadataPleLayer(_TestPleOffloadLayer):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_metadata_device: torch.device | None = None

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        self.dummy_metadata_device = device


class _WeightLoadingModel(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"checkpoint.": ""})

    def __init__(self) -> None:
        super().__init__()
        self.ple = _WeightLoadingPleLayer()
        self.received_checkpoint_names: list[str] = []

    def load_weights(self, weights) -> set[str]:
        """Record filtered names and run the normal automatic loader."""
        filtered_weights = list(weights)
        self.received_checkpoint_names = [name for name, _ in filtered_weights]
        return AutoWeightsLoader(self).load_weights(
            filtered_weights,
            mapper=self.hf_to_vllm_mapper,
        )


class _TestDefaultModelLoader:
    def __init__(self, checkpoint_names: list[str]) -> None:
        self.checkpoint_names = checkpoint_names

    def get_all_weights(self, model_config, model):
        """Return a small streamed checkpoint for weight-filtering tests."""
        del model_config, model
        return ((name, torch.ones(2)) for name in self.checkpoint_names)


def _load_test_ple_weights(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_names: list[str],
) -> tuple[ple_offload_worker.PleOffloadRunner, _WeightLoadingModel]:
    """Run PLE weight discovery with a mapped synthetic checkpoint."""
    model = _WeightLoadingModel()
    loader = _TestDefaultModelLoader(checkpoint_names)
    monkeypatch.setattr(
        ple_offload_worker,
        "initialize_model",
        lambda **_: model,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "DefaultModelLoader",
        _TestDefaultModelLoader,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "get_model_loader",
        lambda _: loader,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "process_weights_after_loading",
        lambda *args: None,
    )

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        load_config=SimpleNamespace(),
    )
    runner._layers = {}
    runner._load_weights()
    return runner, model


def test_ple_offload_loads_mapped_checkpoint_names(
    monkeypatch: pytest.MonkeyPatch,
    caplog_vllm: pytest.LogCaptureFixture,
) -> None:
    checkpoint_names = [
        "checkpoint.ple.weight",
        "checkpoint.unrelated.weight",
        "checkpoint.ple.bias",
    ]

    with caplog_vllm.at_level("INFO", logger=ple_offload_worker.__name__):
        runner, model = _load_test_ple_weights(monkeypatch, checkpoint_names)

    assert model.received_checkpoint_names == [
        "checkpoint.ple.weight",
        "checkpoint.ple.bias",
    ]
    assert runner.layer_names == ["ple"]
    assert "matched 2 checkpoint tensor(s)" in caplog_vllm.text
    assert "verified 2/2 materialized parameter(s)" in caplog_vllm.text


def test_ple_offload_rejects_checkpoint_without_matching_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="filter matched no weights"):
        _load_test_ple_weights(
            monkeypatch,
            ["checkpoint.unrelated.weight"],
        )


def test_ple_offload_rejects_missing_materialized_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=r"parameters: \['ple.bias'\]"):
        _load_test_ple_weights(
            monkeypatch,
            ["checkpoint.ple.weight"],
        )


@pytest.mark.parametrize("load_format", ["dummy", "auto"])
def test_ple_connector_initializes_metadata_only_for_dummy_load(
    load_format: str,
) -> None:
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.device = torch.device("cpu")
    model = torch.nn.Module()
    model.ple = _DummyMetadataPleLayer()
    vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(load_format=load_format),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            hf_text_config=SimpleNamespace(ple_embed_dim=2),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )

    layers = connector._setup_layers(vllm_config, model)

    expected_device = connector.device if load_format == "dummy" else None
    assert model.ple.dummy_metadata_device == expected_device
    assert list(layers) == ["ple"]


def test_ple_offload_wait_only_waits_for_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_calls = []
    error = SimpleNamespace(value=0)
    stream = SimpleNamespace(cuda_stream=17)
    flag_tensor = torch.zeros(1, dtype=torch.int32)

    def fake_wait(*args: object) -> tuple[SimpleNamespace]:
        wait_calls.append(args)
        return (error,)

    monkeypatch.setattr(
        ple_offload_layer.torch.cuda,
        "current_stream",
        lambda: stream,
    )
    wait_flags = ple_offload_layer._stream_mem_ops()[1]
    fake_driver = SimpleNamespace(
        CUstream=lambda value: value,
        CUdeviceptr=lambda value: value,
        cuStreamWaitValue32=fake_wait,
        cuStreamWriteValue32=lambda *args: pytest.fail(
            f"wait unexpectedly wrote the flag: {args}"
        ),
    )
    monkeypatch.setattr(
        ple_offload_layer,
        "_stream_mem_ops",
        lambda: (fake_driver, wait_flags),
    )

    result = ple_offload_layer._ple_offload_wait_impl(
        flag_tensor,
        torch.empty(4, 2),
        torch.empty(4, 2),
    )

    assert result is None
    assert wait_calls == [
        (
            stream.cuda_stream,
            flag_tensor.data_ptr(),
            ple_offload_layer.CpuGpuSemaphore.DONE_VALUE,
            wait_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        )
    ]


def test_offloaded_forward_waits_then_releases_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_calls = []
    reset_calls = []
    flag_tensor = torch.zeros(1, dtype=torch.int32)
    output_buffer = torch.arange(12).reshape(6, 2)

    monkeypatch.setattr(
        torch.ops.vllm,
        "ple_offload_wait",
        lambda *args: wait_calls.append(args),
    )

    layer = _TestPleOffloadLayer()
    layer._is_cpu_offloaded = True
    layer._gpu_output_buffer = output_buffer
    layer._sem = SimpleNamespace(
        flag_tensor=flag_tensor,
        reset=lambda stream: reset_calls.append(stream),
    )
    hidden_states = torch.zeros(3, 2)
    input_ids = torch.arange(3)

    output = layer(hidden_states, input_ids)
    stream = object()
    layer.release_offloaded_output(stream)  # type: ignore[arg-type]

    assert wait_calls == [
        (
            flag_tensor,
            output_buffer,
            hidden_states,
        )
    ]
    assert output.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(output, output_buffer[: input_ids.shape[0]])
    assert reset_calls == [stream]


def test_ple_offload_request_msgpack_round_trip() -> None:
    request = ple_offload_worker.PleOffloadRequest(
        dp_rank=2,
        num_tokens=17,
        num_reqs=3,
    )

    decoded = ple_offload_worker._PLE_OFFLOAD_REQUEST_DECODER.decode(
        msgspec.msgpack.encode(request)
    )

    assert decoded == request


@pytest.mark.parametrize(
    ("ple_layer_ids", "expected"),
    [
        ([1], True),
        ([], False),
    ],
)
def test_ple_offload_requires_ple_layers(
    monkeypatch: pytest.MonkeyPatch,
    ple_layer_ids: list[int],
    expected: bool,
) -> None:
    worker = Worker.__new__(Worker)
    worker.model_config = SimpleNamespace(  # type: ignore[assignment]
        hf_text_config=SimpleNamespace(ple_layer_ids=ple_layer_ids)
    )
    monkeypatch.setattr(envs, "VLLM_PLE_CPU_OFFLOAD", True)

    assert worker._has_ple_layers() is expected


@pytest.mark.parametrize(
    ("architecture", "enable_expert_parallel", "unsupported_setting"),
    [
        ("Qwen4ExpForCausalLM", False, None),
        ("Qwen4ExpForConditionalGeneration", True, None),
        ("UnsupportedArchitecture", True, "architecture"),
    ],
)
def test_ple_offload_accepts_supported_configurations(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    enable_expert_parallel: bool,
    unsupported_setting: str | None,
) -> None:
    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = True
    worker.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=1,
        data_parallel_backend="mp",
        data_parallel_size_local=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=enable_expert_parallel,
        use_ubatching=False,
    )
    worker.model_config = SimpleNamespace(architecture=architecture)
    worker.vllm_config = SimpleNamespace(weight_transfer_config=None)
    monkeypatch.setattr(gpu_worker_module.current_platform, "is_cuda", lambda: True)

    if unsupported_setting is None:
        worker._validate_ple_offload_config()
    else:
        with pytest.raises(ValueError) as exc_info:
            worker._validate_ple_offload_config()
        assert f"Unsupported settings: {unsupported_setting}" in str(exc_info.value)
        assert architecture not in str(exc_info.value)


@pytest.mark.parametrize(
    ("dp_rank", "expected_calls"),
    [(0, 1), (1, 0)],
)
def test_only_dp0_tp0_spawns_shared_ple_offload_worker(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    expected_calls: int,
) -> None:
    calls = []
    worker = Worker.__new__(Worker)
    worker._ple_offload_enabled = True
    worker._ple_offload_worker_handle = None
    worker.rank = 0
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace()
    worker.parallel_config = SimpleNamespace(
        data_parallel_rank=dp_rank,
        data_parallel_size=2,
        tensor_parallel_size=2,
        _ple_offload_ipc_path="ipc:///tmp/test-ple-offload",
    )
    handle = object()

    def fake_make_process(*args: object) -> object:
        calls.append(args)
        return handle

    monkeypatch.setattr(
        ple_offload_worker.PleOffloadWorker,
        "make_process",
        fake_make_process,
    )

    worker.spawn_ple_offload()

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls == [
            (
                worker.vllm_config,
                4,
                "ipc:///tmp/test-ple-offload",
            )
        ]
        assert worker._ple_offload_worker_handle is handle


def test_offload_distributed_sets_config_only_for_model_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = VllmConfig()
    calls = []

    # The Offload subprocess may inherit DP environment variables from a GPU
    # worker, but its isolated model-parallel world must always remain DP1.
    monkeypatch.setattr(envs, "VLLM_DP_SIZE", 2)
    monkeypatch.setattr(envs, "VLLM_DP_RANK", 1)
    monkeypatch.setattr(envs, "VLLM_DP_RANK_LOCAL", 1)

    monkeypatch.setattr(
        ple_offload_worker.dist,
        "is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "init_distributed_environment",
        lambda **kwargs: calls.append(
            ("world", get_current_vllm_config_or_none(), kwargs)
        ),
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "ensure_model_parallel_initialized",
        lambda **kwargs: calls.append(
            ("model_parallel", get_current_vllm_config_or_none(), kwargs)
        ),
    )
    monkeypatch.setattr(
        ple_offload_worker.tempfile,
        "mkdtemp",
        lambda **_: "/tmp/test-ple-offload",
    )

    ple_offload_worker._init_offload_distributed()

    offload_config = calls[1][1]
    assert offload_config is not vllm_config
    assert offload_config.parallel_config.data_parallel_size == 1
    assert offload_config.parallel_config.tensor_parallel_size == 1
    assert offload_config.parallel_config.pipeline_parallel_size == 1
    assert calls == [
        (
            "world",
            None,
            {
                "world_size": 1,
                "rank": 0,
                "distributed_init_method": "file:///tmp/test-ple-offload/store",
                "local_rank": 0,
                "backend": "gloo",
            },
        ),
        (
            "model_parallel",
            offload_config,
            {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "backend": "gloo",
            },
        ),
    ]
    assert get_current_vllm_config_or_none() is None


def test_ple_offload_runner_groups_registrations_by_dp_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def __init__(self, registrations):
            self.registrations = iter(registrations)

        def recv(self):
            return next(self.registrations)

    class FakeStream:
        pass

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            hf_text_config=SimpleNamespace(ple_embed_dim=2),
        ),
    )
    runner._layers = {
        "ple": SimpleNamespace(
            get_offload_output_dtype=lambda default: default,
            get_offload_output_dim=lambda default: default + 1,
        )
    }
    runner._worker_targets = {}
    runner._pinned_bufs = {}
    runner._input_bufs = {}

    registrations = []
    for dp_rank in range(2):
        for tp_rank in range(2):
            registrations.append(
                ple_offload_worker.PleOffloadRegistration(
                    worker_id=dp_rank * 2 + tp_rank,
                    dp_rank=dp_rank,
                    tp_rank=tp_rank,
                    gpu_output_buffers={"ple": torch.empty(8, 2)},
                    sem_flag_tensors={"ple": torch.zeros(1, dtype=torch.int32)},
                    input_ids_buf=torch.full((8,), dp_rank, dtype=torch.int32),
                    query_start_loc_buf=torch.zeros(4, dtype=torch.int32),
                    ngram_context_buf=None,
                )
            )

    original_empty = torch.empty

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(ple_offload_worker.pickle, "loads", lambda item: item)
    monkeypatch.setattr(ple_offload_worker.torch, "empty", unpinned_empty)
    monkeypatch.setattr(
        ple_offload_worker.torch.cuda,
        "Stream",
        lambda **_: FakeStream(),
    )
    monkeypatch.setattr(
        ple_offload_worker.CpuGpuSemaphore,
        "from_ipc_tensor",
        lambda _: SimpleNamespace(),
    )

    runner.accept_registrations(FakeSocket(registrations), len(registrations))

    assert set(runner._worker_targets) == {0, 1}
    assert [target.tp_rank for target in runner._worker_targets[0]["ple"]] == [0, 1]
    assert [target.tp_rank for target in runner._worker_targets[1]["ple"]] == [0, 1]
    assert set(runner._input_bufs) == {0, 1}
    assert runner._input_bufs[0].input_ids_buf[0].item() == 0
    assert runner._input_bufs[1].input_ids_buf[0].item() == 1
    assert set(runner._pinned_bufs) == {0, 1}
    assert runner._pinned_bufs[0]["ple"].shape == (8, 3)
    assert runner._pinned_bufs[1]["ple"].shape == (8, 3)


def test_ple_offload_runner_routes_requests_layer_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class FakeLayer:
        def __init__(self, name: str):
            self.name = name

        def forward_impl(
            self,
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
            output_buffer,
        ):
            del hidden_states, query_start_loc, ngram_context
            events.append((self.name, int(input_ids[0].item())))
            result = input_ids.unsqueeze(-1).expand(-1, 2)
            output_buffer[: result.shape[0]].copy_(result)
            return output_buffer[: result.shape[0]]

    class FakeStream:
        def synchronize(self) -> None:
            pass

    class FakeSemaphore:
        def wait_reset(self, stream) -> None:
            del stream

        def signal(self, stream) -> None:
            del stream

    def target():
        return ple_offload_worker.PleOffloadOutputTarget(
            tp_rank=0,
            gpu_output_buffer=torch.empty(4, 2, dtype=torch.int32),
            sem=FakeSemaphore(),
            copy_stream=FakeStream(),  # type: ignore[arg-type]
        )

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner._clamp_input_ids = True
    runner._layers = {"ple0": FakeLayer("ple0"), "ple1": FakeLayer("ple1")}
    runner._worker_targets = {
        0: {"ple0": [target()], "ple1": [target()]},
        1: {"ple0": [target()], "ple1": [target()]},
    }
    runner._input_bufs = {
        0: ple_offload_worker.PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([-1, 11], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 2], dtype=torch.int32),
            ngram_context_buf=None,
        ),
        1: ple_offload_worker.PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([20], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 1], dtype=torch.int32),
            ngram_context_buf=None,
        ),
    }
    runner._pinned_bufs = {
        dp_rank: {
            layer_name: torch.empty(4, 2, dtype=torch.int32)
            for layer_name in runner._layers
        }
        for dp_rank in range(2)
    }
    monkeypatch.setattr(
        ple_offload_worker.torch.cuda,
        "stream",
        lambda _: nullcontext(),
    )

    runner._handle_requests(
        [
            ple_offload_worker.PleOffloadRequest(
                dp_rank=0,
                num_tokens=2,
                num_reqs=1,
            ),
            ple_offload_worker.PleOffloadRequest(
                dp_rank=1,
                num_tokens=1,
                num_reqs=1,
            ),
        ]
    )

    assert events == [
        ("ple0", 0),
        ("ple0", 20),
        ("ple1", 0),
        ("ple1", 20),
    ]
    torch.testing.assert_close(
        runner._worker_targets[0]["ple1"][0].gpu_output_buffer[:2],
        torch.tensor([[0, 0], [11, 11]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        runner._worker_targets[1]["ple1"][0].gpu_output_buffer[:1],
        torch.tensor([[20, 20]], dtype=torch.int32),
    )


def test_wait_for_ready_closes_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(
        ple_offload_worker.PleOffloadWorker,
        "start_watchdog",
        staticmethod(started.append),
    )
    context = ple_offload_worker.get_mp_context()
    ready_reader, ready_writer = context.Pipe(duplex=False)
    ready_writer.send(
        {
            "status": ple_offload_worker.PleOffloadWorker.READY_STR,
            "layer_names": ["layers.0.ple.ple_embedding"],
        }
    )
    ready_writer.close()
    handle = ple_offload_worker.PleOffloadWorkerHandle(
        proc=None,
        death_writer=None,
        ready_pipe_reader=ready_reader,
    )

    ple_offload_worker.PleOffloadWorker.wait_for_ready(handle)

    assert handle.ready_pipe_reader is None
    assert started == [handle], "readiness must arm the watchdog"


def test_ple_offload_watchdog_releases_waits_and_stops_the_worker(monkeypatch):
    """A dead offload process must break the GPU wait, not stall it."""
    import signal as signal_module
    import threading

    from vllm.v1.ple_offload.worker import (
        PleOffloadWorker,
        PleOffloadWorkerHandle,
    )

    exited = threading.Event()

    class DeadChild:
        exitcode = -9

        def join(self, timeout=None):
            exited.wait(5)

        def is_alive(self):
            return not exited.is_set()

    released = []
    signals = []

    def fake_release() -> int:
        released.append(1)
        return 3

    monkeypatch.setattr(ple_offload_layer, "release_all_semaphores", fake_release)
    monkeypatch.setattr(
        ple_offload_worker.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    handle = PleOffloadWorkerHandle(
        proc=DeadChild(), death_writer=None, ready_pipe_reader=None
    )
    PleOffloadWorker.start_watchdog(handle)
    exited.set()
    handle.watchdog.join(timeout=5)

    assert not handle.watchdog.is_alive()
    assert released == [1], "the watchdog must release the pending GPU waits"
    assert signals and signals[0][1] == signal_module.SIGTERM


def test_ple_offload_watchdog_stays_quiet_on_a_planned_shutdown(monkeypatch):
    """close() marks the exit as expected, so the watchdog must do nothing."""
    import threading

    from vllm.v1.ple_offload.worker import (
        PleOffloadWorker,
        PleOffloadWorkerHandle,
    )

    exited = threading.Event()

    class ClosingChild:
        exitcode = 0

        def join(self, timeout=None):
            exited.wait(5)

        def is_alive(self):
            return not exited.is_set()

    signals = []
    monkeypatch.setattr(
        ple_offload_worker.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    handle = PleOffloadWorkerHandle(
        proc=ClosingChild(), death_writer=None, ready_pipe_reader=None
    )
    PleOffloadWorker.start_watchdog(handle)
    handle.shutting_down = True
    exited.set()
    handle.watchdog.join(timeout=5)

    assert not signals, "a planned shutdown must not signal the worker"


def test_drop_checkpoint_page_cache_advises_every_shard(tmp_path, monkeypatch):
    """Shutdown must hand the checkpoint's cached pages back to the kernel."""
    from vllm.v1.ple_offload.worker import drop_checkpoint_page_cache

    sizes = {
        "model-00001-of-00002.safetensors": 4096,
        "model-00002-of-00002.safetensors": 8192,
    }
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(b"\0" * size)
    (tmp_path / "config.json").write_text("{}")

    advised = []
    real_fadvise = ple_offload_worker.os.posix_fadvise

    def record(fd, offset, length, advice):
        advised.append((offset, length, advice))
        return real_fadvise(fd, offset, length, advice)

    monkeypatch.setattr(ple_offload_worker.os, "posix_fadvise", record)
    total = drop_checkpoint_page_cache(str(tmp_path))

    assert total == sum(sizes.values())
    assert len(advised) == len(sizes), "config.json must not be touched"
    assert all(
        entry == (0, 0, ple_offload_worker.os.POSIX_FADV_DONTNEED) for entry in advised
    )


def test_drop_checkpoint_page_cache_ignores_a_missing_directory():
    """A dummy-weight run has no checkpoint directory to release."""
    from vllm.v1.ple_offload.worker import drop_checkpoint_page_cache

    assert drop_checkpoint_page_cache("/nonexistent/checkpoint/path") == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_release_all_semaphores_breaks_a_real_stream_wait():
    """The release must run even while a stream is parked in the wait."""
    import threading

    semaphore = ple_offload_layer.CpuGpuSemaphore(torch.device("cuda"))
    semaphore.reset()
    torch.accelerator.synchronize()

    waiting = torch.Stream(device="cuda")
    marker = torch.zeros(1, dtype=torch.int32, device="cuda")
    previous = torch.accelerator.current_stream()
    torch.accelerator.set_stream(waiting)
    try:
        # Park this stream until the flag reads DONE, then leave a mark so the
        # test can tell the wait actually cleared.
        semaphore_wait = torch.ops.vllm.ple_offload_wait
        semaphore_wait(semaphore.flag_tensor, marker, marker)
        marker.fill_(7)
    finally:
        torch.accelerator.set_stream(previous)

    finished = threading.Event()

    def wait_for_stream() -> None:
        waiting.synchronize()
        finished.set()

    watcher = threading.Thread(target=wait_for_stream, daemon=True)
    watcher.start()
    assert not finished.wait(0.5), "the stream should still be parked"

    # The real caller is the watchdog thread, which holds no CUDA context.
    from_thread: dict[str, int] = {}
    failure: dict[str, str] = {}

    def release_from_fresh_thread() -> None:
        try:
            from_thread["released"] = ple_offload_layer.release_all_semaphores()
        except Exception as error:  # pragma: no cover - surfaced by the assert
            failure["error"] = repr(error)

    releaser = threading.Thread(target=release_from_fresh_thread, daemon=True)
    releaser.start()
    releaser.join(timeout=30)
    assert not failure, failure.get("error")
    assert from_thread.get("released", 0) >= 1

    assert finished.wait(10), "release_all_semaphores did not break the wait"
    watcher.join(timeout=5)
    assert marker.item() == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_release_all_semaphores_uses_a_dedicated_non_blocking_stream(monkeypatch):
    """The release must not queue behind the wait it is breaking.

    Writing on the current or legacy default stream can be ordered against
    the stalled ``cuStreamWaitValue32``, so the write would never run. Pin
    the behaviour: the write goes to a stream created NON_BLOCKING, and it is
    not the caller's current stream.
    """
    import threading

    cuda_driver, _ = ple_offload_layer._stream_mem_ops()
    ple_offload_layer._recovery_stream.cache_clear()

    created_flags = []
    created_streams = []
    real_create = cuda_driver.cuStreamCreate

    def record_create(flags):
        created_flags.append(flags)
        result = real_create(flags)
        created_streams.append(int(result[1]))
        return result

    written_streams = []
    real_write = cuda_driver.cuStreamWriteValue32

    def record_write(stream, ptr, value, flags):
        written_streams.append(int(stream))
        return real_write(stream, ptr, value, flags)

    monkeypatch.setattr(cuda_driver, "cuStreamCreate", record_create)
    monkeypatch.setattr(cuda_driver, "cuStreamWriteValue32", record_write)

    semaphore = ple_offload_layer.CpuGpuSemaphore(torch.device("cuda"))

    # Run it where the watchdog runs: a thread with no current CUDA context.
    outcome: dict[str, int] = {}
    outcome_error: dict[str, str] = {}

    def release_from_fresh_thread() -> None:
        try:
            outcome["released"] = ple_offload_layer.release_all_semaphores()
        except Exception as error:  # pragma: no cover - surfaced by the assert
            outcome_error["error"] = repr(error)

    thread = threading.Thread(target=release_from_fresh_thread, daemon=True)
    thread.start()
    thread.join(timeout=30)
    assert not outcome_error, outcome_error.get("error")
    released = outcome.get("released", 0)

    assert released >= 1
    assert created_flags == [cuda_driver.CUstream_flags.CU_STREAM_NON_BLOCKING.value], (
        "the recovery stream must be created non-blocking"
    )
    assert written_streams, "no release write was issued"
    assert set(written_streams) == set(created_streams), (
        "the release must write on the stream it created, not an ambient one"
    )
    del semaphore
    ple_offload_layer._recovery_stream.cache_clear()


def test_drop_checkpoint_page_cache_resolves_a_hub_model_id(tmp_path, monkeypatch):
    """A repo id has to resolve to the hub cache, not be skipped."""
    from vllm.transformers_utils import repo_utils
    from vllm.v1.ple_offload.worker import drop_checkpoint_page_cache

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 2048)

    monkeypatch.setattr(repo_utils, "get_model_path", lambda *a, **k: str(tmp_path))
    assert drop_checkpoint_page_cache("org/model") == 2048


def test_drop_checkpoint_page_cache_skips_an_unavailable_model_id(monkeypatch):
    """No local copy means nothing to release, and no exception either."""
    from vllm.transformers_utils import repo_utils
    from vllm.v1.ple_offload.worker import drop_checkpoint_page_cache

    def explode(*args, **kwargs):
        raise OSError("not cached")

    monkeypatch.setattr(repo_utils, "get_model_path", explode)
    assert drop_checkpoint_page_cache("org/never-downloaded") == 0


def test_drop_page_cache_for_releases_the_recorded_shards(tmp_path, monkeypatch):
    """Cleanup follows the files the loader opened, not a re-derived path."""
    from vllm.v1.ple_offload.worker import drop_page_cache_for

    shards = []
    for index, size in enumerate((4096, 8192)):
        shard = tmp_path / f"shard-{index}.safetensors"
        shard.write_bytes(b"\0" * size)
        shards.append(str(shard))

    advised = []
    real_fadvise = ple_offload_worker.os.posix_fadvise

    def record(fd, offset, length, advice):
        advised.append(advice)
        return real_fadvise(fd, offset, length, advice)

    monkeypatch.setattr(ple_offload_worker.os, "posix_fadvise", record)
    assert drop_page_cache_for(shards) == 12288
    assert advised == [ple_offload_worker.os.POSIX_FADV_DONTNEED] * 2


def test_drop_page_cache_for_ignores_paths_that_are_gone(tmp_path):
    """A checkpoint deleted mid-run must not break shutdown."""
    from vllm.v1.ple_offload.worker import drop_page_cache_for

    assert drop_page_cache_for([str(tmp_path / "missing.safetensors")]) == 0


def test_drop_page_cache_for_reports_only_what_the_kernel_accepted(
    tmp_path, monkeypatch
):
    """A refused advise must not read as released, or the fallback is skipped."""
    from vllm.v1.ple_offload.worker import drop_page_cache_for

    shard = tmp_path / "shard.safetensors"
    shard.write_bytes(b"\0" * 4096)

    def refuse(fd, offset, length, advice):
        raise OSError("advice refused")

    monkeypatch.setattr(ple_offload_worker.os, "posix_fadvise", refuse)
    assert drop_page_cache_for([str(shard)]) == 0


def test_wait_for_ready_keeps_the_shards_the_child_reported(monkeypatch):
    """The parent needs the child's shard list to clean up after a kill."""
    from multiprocessing import Pipe

    from vllm.v1.ple_offload.worker import (
        PleOffloadWorker,
        PleOffloadWorkerHandle,
    )

    # Readiness arms the watchdog, which would join a process this test does
    # not have; the watchdog itself is covered separately.
    monkeypatch.setattr(
        PleOffloadWorker, "start_watchdog", staticmethod(lambda handle: None)
    )

    reader, writer = Pipe(duplex=False)
    writer.send(
        {
            "status": PleOffloadWorker.READY_STR,
            "layer_names": ["ple"],
            "checkpoint_shards": ["/models/demo/shard-0.safetensors"],
        }
    )
    handle = PleOffloadWorkerHandle(
        proc=SimpleNamespace(is_alive=lambda: True),
        death_writer=None,
        ready_pipe_reader=reader,
    )
    PleOffloadWorker.wait_for_ready(handle)

    assert handle.checkpoint_shards == ["/models/demo/shard-0.safetensors"]
    writer.close()


def test_watchdog_releases_the_reported_shards(monkeypatch):
    """The kill path must release the same files the load opened."""
    import signal as signal_module
    import threading

    from vllm.v1.ple_offload.worker import (
        PleOffloadWorker,
        PleOffloadWorkerHandle,
    )

    exited = threading.Event()

    class DeadChild:
        exitcode = -9

        def join(self, timeout=None):
            exited.wait(5)

        def is_alive(self):
            return not exited.is_set()

    dropped: list[list[str]] = []
    signals: list[int] = []
    monkeypatch.setattr(ple_offload_layer, "release_all_semaphores", lambda: 0)

    def fake_drop(paths) -> int:
        dropped.append(list(paths))
        return 4096

    monkeypatch.setattr(ple_offload_worker, "drop_page_cache_for", fake_drop)
    monkeypatch.setattr(
        ple_offload_worker.os, "kill", lambda pid, sig: signals.append(sig)
    )

    handle = PleOffloadWorkerHandle(
        proc=DeadChild(),
        death_writer=None,
        ready_pipe_reader=None,
        checkpoint_shards=["/models/demo/shard-0.safetensors"],
    )
    PleOffloadWorker.start_watchdog(handle)
    exited.set()
    handle.watchdog.join(timeout=5)

    assert dropped == [["/models/demo/shard-0.safetensors"]]
    assert signals == [signal_module.SIGTERM]


def _connector_stub(monkeypatch):
    """A connector with only the routing surface prepare_forward touches."""
    from vllm.v1.ple_offload.connector import PleOffloadConnector

    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    calls: dict[str, int] = {"sync": 0, "async": 0, "dummy": 0}
    monkeypatch.setattr(
        PleOffloadConnector,
        "_prepare_forward_sync",
        lambda self, num_reqs, num_tokens: calls.__setitem__("sync", calls["sync"] + 1),
    )
    monkeypatch.setattr(
        PleOffloadConnector,
        "_launch",
        lambda self, num_reqs, num_tokens: calls.__setitem__(
            "async", calls["async"] + 1
        ),
    )
    monkeypatch.setattr(
        PleOffloadConnector,
        "signal_dummy_outputs",
        lambda self, num_tokens: calls.__setitem__("dummy", calls["dummy"] + 1),
    )
    return connector, calls


def test_prepare_forward_takes_the_sync_path_only_for_full_graphs(monkeypatch):
    """Graph replay needs the synchronous path; everything else must not pay for it."""
    connector, calls = _connector_stub(monkeypatch)

    connector.prepare_forward(1, 8, dummy_run=False, use_cudagraph=True)
    assert (calls["sync"], calls["async"]) == (1, 0)

    connector.prepare_forward(1, 8, dummy_run=False, use_cudagraph=False)
    assert (calls["sync"], calls["async"]) == (1, 1)

    connector.prepare_forward(1, 8, dummy_run=False)
    assert (calls["sync"], calls["async"]) == (1, 2), "the default stays async"

    connector.prepare_forward(1, 8, dummy_run=True, use_cudagraph=True)
    assert calls["dummy"] == 1 and calls["sync"] == 1


def test_both_model_runners_report_full_graph_replay():
    """MRV1 and MRV2 must both tell the connector when a graph will replay."""
    import inspect

    import vllm.v1.worker.gpu.model_runner as mrv2
    import vllm.v1.worker.gpu_model_runner as mrv1

    for module in (mrv1, mrv2):
        source = inspect.getsource(module)
        index = source.find("_ple_offload_connector.prepare_forward(")
        assert index != -1, f"{module.__name__} does not call prepare_forward"
        call = source[index : index + 400]
        assert "use_cudagraph=" in call, (
            f"{module.__name__} never reports graph replay to the PLE connector"
        )
        assert "CUDAGraphMode.FULL" in call, (
            f"{module.__name__} must gate the sync path on FULL graphs"
        )


def test_poll_semaphores_gives_up_instead_of_spinning_forever(monkeypatch):
    """A stuck offload worker must fail the step, not burn a core silently."""
    from vllm.v1.ple_offload.connector import PleOffloadConnector

    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    never_done = SimpleNamespace(
        _sem=SimpleNamespace(flag_tensor=SimpleNamespace(item=lambda: 0))
    )
    connector._layers = {"layers.1.ple": never_done}
    monkeypatch.setattr(envs, "VLLM_PLE_OFFLOAD_STEP_TIMEOUT", 0.05)

    with pytest.raises(RuntimeError, match="did not answer for layers.1.ple"):
        connector._poll_semaphores()
