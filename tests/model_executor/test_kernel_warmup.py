# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

_deep_gemm_stub = types.ModuleType(
    "vllm.model_executor.warmup.deep_gemm_warmup"
)
_deep_gemm_stub.deep_gemm_warmup = lambda *args, **kwargs: None
sys.modules.setdefault(
    "vllm.model_executor.warmup.deep_gemm_warmup", _deep_gemm_stub
)

from vllm.model_executor.warmup.kernel_warmup import kernel_warmup


def _make_worker(backend_names: list[str]):
    attn_groups = [[SimpleNamespace(backend=SimpleNamespace(
        get_name=lambda name=name: name))] for name in backend_names]

    worker = Mock()
    worker.get_model.return_value = Mock()
    worker.scheduler_config.max_num_batched_tokens = 8192
    worker.scheduler_config.max_num_seqs = 8
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False
    worker.model_runner.is_pooling_model = False
    worker.model_runner.attn_groups = attn_groups
    return worker


def test_kernel_warmup_triton_runs_prefill_and_decode():
    worker = _make_worker(["TRITON_ATTN"])

    kernel_warmup(worker)

    assert worker.model_runner._dummy_run.call_count == 2
    worker.model_runner._dummy_run.assert_any_call(
        num_tokens=16,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
    )
    worker.model_runner._dummy_run.assert_any_call(
        num_tokens=8,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        uniform_decode=True,
    )


def test_kernel_warmup_triton_skips_mixed_backends():
    worker = _make_worker(["TRITON_ATTN", "FLASHINFER"])

    kernel_warmup(worker)

    worker.model_runner._dummy_run.assert_not_called()
