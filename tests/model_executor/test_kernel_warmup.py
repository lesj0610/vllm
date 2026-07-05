# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.warmup import kernel_warmup, minimax_m3_msa_warmup


def _patch_common_warmups(monkeypatch):
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "qwen_triton_warmup", Mock())
    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", Mock())
    monkeypatch.setattr(kernel_warmup, "sparse_mla_triton_warmup_if_needed", Mock())
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        Mock(),
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        Mock(),
    )
    monkeypatch.setattr(minimax_m3_msa_warmup, "minimax_m3_msa_warmup", Mock())

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.warmup.hybrid_gdn_mamba_mrope_warmup",
        SimpleNamespace(hybrid_gdn_mamba_mrope_warmup=Mock()),
    )


def _make_worker(model_runner):
    if not hasattr(model_runner, "dtype"):
        model_runner.dtype = None
    return SimpleNamespace(
        get_model=lambda: object(),
        model_runner=model_runner,
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
            kernel_config=SimpleNamespace(
                enable_cutedsl_warmup=False,
                enable_flashinfer_autotune=False,
            ),
            model_config=SimpleNamespace(),
        ),
    )


def test_kernel_warmup_invokes_private_kv_block_zeroer(monkeypatch):
    _patch_common_warmups(monkeypatch)

    zeroer = Mock()
    model_runner = SimpleNamespace(
        _kv_block_zeroer=zeroer,
        is_pooling_model=True,
        attn_groups=[],
    )

    kernel_warmup.kernel_warmup(_make_worker(model_runner))

    zeroer.warmup.assert_called_once_with()


def test_kernel_warmup_invokes_public_kv_block_zeroer(monkeypatch):
    _patch_common_warmups(monkeypatch)

    zeroer = Mock()
    model_runner = SimpleNamespace(
        kv_block_zeroer=zeroer,
        is_pooling_model=True,
        attn_groups=[],
    )

    kernel_warmup.kernel_warmup(_make_worker(model_runner))

    zeroer.warmup.assert_called_once_with()
