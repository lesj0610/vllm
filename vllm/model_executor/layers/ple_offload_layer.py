# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for PLE layers that support cross-process CPU offload.

Synchronization between the offload process and the GPU main stream is
handled by :class:`CpuGpuSemaphore`, which uses CUDA's
``cuStreamWaitValue32`` / ``cuStreamWriteValue32`` stream memory operations.

Typical decode-step timeline
----------------------------
Offload process (CPU)                  GPU main stream (forward)
-----------------------------          ------------------------------
forward_impl() on CPU                  ... other GPU ops ...
pinned -> GPU async H2D (copy_stream)   WaitValue32(flag==1)  <- in Graph
WriteValue32(flag=1, copy_stream)       return _gpu_output_buffer[:n]
                                       ... model consumes output ...
                                       WriteValue32(flag=0)  <- after forward
"""

import functools
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op


@functools.cache
def _stream_mem_ops() -> tuple[Any, Any]:
    """Return the CUDA driver bindings the semaphore's stream ops need.

    ROCm builds import this module for ``is_offload_process`` but never reach
    a stream memory operation, and ``cuda-bindings`` is absent there, so the
    import cannot sit at module scope.
    """
    from cuda.bindings import driver

    return driver, driver.CUstreamWaitValue_flags


# Module-level flag set to True inside the offload subprocess.
# Because the offload process and GPU worker processes are separate OS
# processes (spawned via multiprocessing), each has its own memory space.
# A plain module-level bool is sufficient -- no thread-local storage needed.
logger = init_logger(__name__)

_offload_worker_flag = False


def is_offload_process() -> bool:
    """Return True inside the dedicated PLE CPU-offload subprocess."""
    return _offload_worker_flag


def mark_as_offload_worker() -> None:
    """Mark the current process as the dedicated PLE offload worker."""
    global _offload_worker_flag
    _offload_worker_flag = True


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


# Semaphores the GPU worker is waiting on. A stream-memory wait can only be
# broken by writing the value it waits for, so the offload watchdog needs to
# reach every live semaphore when the offload process dies.
_LIVE_SEMAPHORES: "weakref.WeakSet[CpuGpuSemaphore]" = weakref.WeakSet()


@functools.cache
def _recovery_stream(device_index: int) -> Any:
    """Create the non-blocking stream used to break a stalled wait.

    The write that releases a semaphore cannot be queued behind the wait it
    is meant to break, and the legacy default stream can be ordered against
    that wait. A stream created with CU_STREAM_NON_BLOCKING is ordered
    against nothing, so the write runs even while another stream is parked in
    ``cuStreamWaitValue32``.
    """
    del device_index  # One CUDA context per worker process.
    cuda_driver, _ = _stream_mem_ops()
    _, stream = _cuda_check(
        cuda_driver.cuStreamCreate(
            cuda_driver.CUstream_flags.CU_STREAM_NON_BLOCKING.value
        ),
        "cuStreamCreate(recovery)",
    )
    return stream


def release_all_semaphores() -> int:
    """Write DONE to every live semaphore and return how many were released.

    Called when the offload process is gone: the results are worthless, but
    the streams waiting for them have to make progress so the worker can shut
    down instead of hanging in the driver.
    """
    cuda_driver, _ = _stream_mem_ops()
    released = 0
    streams = set()
    for semaphore in list(_LIVE_SEMAPHORES):
        try:
            flag = semaphore.flag_tensor
            stream = _recovery_stream(flag.device.index or 0)
            _cuda_check(
                cuda_driver.cuStreamWriteValue32(
                    cuda_driver.CUstream(stream),
                    cuda_driver.CUdeviceptr(flag.data_ptr()),
                    CpuGpuSemaphore.DONE_VALUE,
                    0,
                ),
                "release_all_semaphores",
            )
            streams.add(stream)
            released += 1
        except Exception:
            logger.exception("Failed to release a PLE offload semaphore")
    # Only the recovery streams are synchronized: a device-wide wait would
    # block on the very stream this is trying to free.
    for stream in streams:
        try:
            _cuda_check(
                cuda_driver.cuStreamSynchronize(cuda_driver.CUstream(stream)),
                "cuStreamSynchronize(recovery)",
            )
        except Exception:
            logger.exception("Failed to synchronize a PLE recovery stream")
    return released


class CpuGpuSemaphore:
    """Cross-process semaphore backed by a one-element int32 CUDA tensor.

    The flag is stored in a regular GPU tensor that is shared through
    PyTorch's CUDA IPC mechanism. Both processes obtain device pointers that
    map to the same physical GPU memory, so a stream-memory wait in the GPU
    worker observes the write issued by the offload process.
    """

    RESET_VALUE = 0
    DONE_VALUE = 1

    def __init__(self, device: torch.device) -> None:
        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        _LIVE_SEMAPHORES.add(self)

    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        """Construct a semaphore from a CUDA tensor received through IPC."""
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        _LIVE_SEMAPHORES.add(semaphore)
        return semaphore

    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the CUDA tensor used to share the semaphore through IPC."""
        return self._flag_tensor

    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        cuda_driver, CUstreamWaitValue_flags = _stream_mem_ops()
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                0,
            ),
            "CpuGpuSemaphore.reset",
        )

    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        cuda_driver, CUstreamWaitValue_flags = _stream_mem_ops()
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.DONE_VALUE,
                0,
            ),
            "CpuGpuSemaphore.signal",
        )

    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        cuda_driver, CUstreamWaitValue_flags = _stream_mem_ops()
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "CpuGpuSemaphore.wait_reset",
        )


# ---------------------------------------------------------------------------
# Custom op: vllm::ple_offload_wait
# ---------------------------------------------------------------------------
# Wraps cuStreamWaitValue32 as a torch custom op so that
# torch.compile(fullgraph=True) treats the CUDA Driver call as an opaque node.
# The hidden_states argument creates a dynamo data-dependency edge that prevents
# the wait from being reordered before preceding GPU work.
# ---------------------------------------------------------------------------


def _ple_offload_wait_impl(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Wait for the CPU result without releasing its output buffer."""
    cuda_driver, CUstreamWaitValue_flags = _stream_mem_ops()
    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    dev_ptr = cuda_driver.CUdeviceptr(sem_flag_tensor.data_ptr())
    _cuda_check(
        cuda_driver.cuStreamWaitValue32(
            cuda_stream,
            dev_ptr,
            CpuGpuSemaphore.DONE_VALUE,
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        ),
        "cuStreamWaitValue32(done)",
    )


def _ple_offload_wait_fake(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Represent the side-effect-only wait during dynamo tracing."""
    pass


direct_register_custom_op(
    op_name="ple_offload_wait",
    op_func=_ple_offload_wait_impl,
    mutates_args=["gpu_output_buffer"],
    fake_impl=_ple_offload_wait_fake,
)


class PleOffloadLayer(nn.Module, ABC):
    """Base class for embedding-like PLE layers that can run on CPU.

    Subclasses implement :meth:`forward_impl`, which contains the regular
    on-device computation. In a GPU worker with PLE offload enabled, subclass
    constructors are skipped so their large weights are never allocated. The
    CPU process constructs a complete copy and owns those weights instead.
    """

    _is_cpu_offloaded = False
    _gpu_output_buffer: torch.Tensor
    _sem: CpuGpuSemaphore

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Skip subclass initialization in cross-process GPU-worker mode."""
        super().__init_subclass__(**kwargs)
        original_init_obj = cls.__dict__.get("__init__")
        if original_init_obj is None or not callable(original_init_obj):
            return
        original_init = cast(Callable[..., None], original_init_obj)

        @functools.wraps(original_init)
        def guarded_init(
            self: "PleOffloadLayer", *args: object, **kwargs: object
        ) -> None:
            if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
                nn.Module.__init__(self)
                return
            original_init(self, *args, **kwargs)

        cls.__init__ = guarded_init  # type: ignore[method-assign, assignment]

    @classmethod
    def get_target_device(cls) -> torch.device:
        """Return CPU for the offload process and the active GPU otherwise."""
        if envs.VLLM_PLE_CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device("cuda", torch.accelerator.current_device_index())

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the actual embedding computation on the owning device."""
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used by cross-process output buffers."""
        return default_dtype

    def setup_cross_process_offload(
        self,
        gpu_output_buffer: torch.Tensor,
        semaphore: CpuGpuSemaphore,
    ) -> None:
        """Configure the GPU-worker placeholder with its IPC resources."""
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._sem = semaphore

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Wait for an offloaded result or delegate to ``forward_impl``."""
        if self._is_cpu_offloaded:
            torch.ops.vllm.ple_offload_wait(
                self._sem.flag_tensor,
                self._gpu_output_buffer,
                hidden_states,
            )
            return self._gpu_output_buffer[: input_ids.shape[0]]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Mark the cross-process output buffer reusable on ``stream``."""
        if self._is_cpu_offloaded:
            self._sem.reset(stream)
