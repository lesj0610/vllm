# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer staged KV dequant support."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

try:
    import flashinfer
except ImportError:
    pytest.skip(
        "flashinfer is required for FlashInfer KV dequant tests",
        allow_module_level=True,
    )

import vllm.v1.attention.backends.flashinfer as flashinfer_backend_module
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flashinfer import (
    FIPrefill,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PerLayerParameters, set_kv_cache_layout
from vllm.v1.attention.kv_dequant.flashinfer_tile import (
    staged_dequantize_paged_kv_cache,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import INT4_CODEBOOK_LEVELS
from vllm.v1.kv_cache_interface import (
    INT4_CHANNELS_PER_SCALE,
    FullAttentionSpec,
    KVQuantMode,
    get_int4_num_scale_groups,
    get_kv_cache_head_size_bytes,
)


def _encode_float32_inline_scales(scales: torch.Tensor) -> torch.Tensor:
    return (
        scales.contiguous()
        .view(torch.int8)
        .view(*scales.shape[:-1], scales.shape[-1] * 4)
    )


def _encode_float16_inline_scales(scales: torch.Tensor) -> torch.Tensor:
    return (
        scales.contiguous()
        .view(torch.uint8)
        .view(*scales.shape[:-1], scales.shape[-1] * 2)
    )


def _pack_int4(idx: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(
        *idx.shape[:-1], (idx.shape[-1] + 1) // 2, dtype=torch.uint8, device=idx.device
    )
    packed.copy_(idx[..., ::2])
    packed[..., : idx[..., 1::2].shape[-1]] |= idx[..., 1::2] << 4
    return packed


def _build_int8_inline_cache(
    q_cache: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    head_size = q_cache.shape[-1]
    raw = torch.zeros(
        *q_cache.shape[:-1],
        head_size + 4,
        dtype=torch.int8,
        device=q_cache.device,
    )
    raw[..., :head_size] = q_cache
    raw[..., head_size:] = _encode_float32_inline_scales(scales)
    return raw


def _build_int4_inline_cache(
    q_cache: torch.Tensor,
    scales: torch.Tensor,
    *,
    head_size: int,
) -> torch.Tensor:
    packed_head_size = get_kv_cache_head_size_bytes(
        head_size, torch.uint8, KVQuantMode.INT4_PER_TOKEN_HEAD
    )
    scale_pad = get_int4_num_scale_groups(head_size) * 2
    raw = torch.zeros(
        *q_cache.shape[:-1],
        packed_head_size + scale_pad,
        dtype=torch.uint8,
        device=q_cache.device,
    )
    raw[..., : q_cache.shape[-1]] = q_cache
    raw[..., packed_head_size:] = _encode_float16_inline_scales(scales)
    return raw


def _require_flashinfer_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("FlashInfer runtime smoke test requires CUDA")


def test_flashinfer_backend_get_kv_cache_shape_quantized():
    assert FlashInferBackend.get_kv_cache_shape(
        8, 16, 4, 64, cache_dtype_str="int8_per_token_head"
    ) == (8, 2, 16, 4, 68)
    assert FlashInferBackend.get_kv_cache_shape(
        8, 16, 4, 66, cache_dtype_str="int4_per_token_head"
    ) == (8, 2, 16, 4, 40)


def _mock_per_layer_parameters(head_size: int):
    return {
        "test_layer_0": PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=head_size**-0.5,
        )
    }


def _install_dummy_model_config(
    config,
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
):
    class DummyModelConfig:
        dtype = torch.float16
        max_model_len = 64

        def get_num_attention_heads(self, parallel_config):
            return num_heads

        def get_num_kv_heads(self, parallel_config):
            return num_kv_heads

        def get_head_size(self):
            return head_size

    config.model_config = DummyModelConfig()


def test_flashinfer_metadata_builder_uses_model_dtype_for_staged_quant_q(
    default_vllm_config,
    monkeypatch,
):
    head_size = 40
    _install_dummy_model_config(
        default_vllm_config, head_size=head_size, num_heads=8, num_kv_heads=8
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=head_size,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    )
    monkeypatch.setattr(
        flashinfer_backend_module,
        "can_use_trtllm_attention",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        flashinfer_backend_module,
        "get_per_layer_parameters",
        lambda *args, **kwargs: _mock_per_layer_parameters(head_size),
    )

    builder = FlashInferMetadataBuilder(
        kv_cache_spec, ["test_layer_0"], default_vllm_config, torch.device("cpu")
    )

    assert builder.use_staged_quant_kv is True
    assert builder.use_trtllm_decode_attention is False
    assert builder.kv_cache_dtype == torch.uint8
    assert builder.q_data_type == torch.float16


def test_flashinfer_cudagraph_support_drops_for_staged_quant(
    default_vllm_config,
    monkeypatch,
):
    _install_dummy_model_config(
        default_vllm_config, head_size=40, num_heads=8, num_kv_heads=8
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=40,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    monkeypatch.setattr(
        flashinfer_backend_module,
        "can_use_trtllm_attention",
        lambda *args, **kwargs: True,
    )

    support = FlashInferMetadataBuilder.get_cudagraph_support(
        default_vllm_config, kv_cache_spec
    )
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_staged_dequantize_paged_kv_cache_int8_matches_reference():
    quant = torch.tensor(
        [[[[[1, -2, 3, -4]]], [[[5, -6, 7, -8]]]]],
        dtype=torch.int8,
    )
    scales = torch.tensor([2.0, 0.5], dtype=torch.float32).view(1, 2, 1, 1, 1)
    raw = _build_int8_inline_cache(quant, scales)

    dequant = staged_dequantize_paged_kv_cache(
        raw,
        head_size=4,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        out_dtype=torch.float16,
    )

    expected = quant.to(torch.float32) * scales
    torch.testing.assert_close(dequant.float(), expected, atol=0.0, rtol=0.0)


def test_staged_dequantize_paged_kv_cache_int4_matches_reference():
    head_size = 40
    idx = torch.tensor(
        [
            [
                [
                    [
                        [
                            0,
                            1,
                            8,
                            9,
                            2,
                            10,
                            3,
                            11,
                            4,
                            12,
                            5,
                            13,
                            6,
                            14,
                            7,
                            15,
                            0,
                            1,
                            8,
                            9,
                            2,
                            10,
                            3,
                            11,
                            4,
                            12,
                            5,
                            13,
                            6,
                            14,
                            7,
                            15,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                        ]
                    ]
                ],
                [
                    [
                        [
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                        ]
                    ]
                ],
            ]
        ],
        dtype=torch.uint8,
    )
    packed = _pack_int4(idx)
    scales = torch.tensor([1.0, 0.25, 0.5, 2.0], dtype=torch.float16).view(
        1, 2, 1, 1, 2
    )
    raw = _build_int4_inline_cache(packed, scales, head_size=head_size)

    dequant = staged_dequantize_paged_kv_cache(
        raw,
        head_size=head_size,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        out_dtype=torch.float16,
    )

    codebook = torch.tensor(INT4_CODEBOOK_LEVELS, dtype=torch.float32)
    group_idx = torch.arange(head_size) // INT4_CHANNELS_PER_SCALE
    expected = codebook[idx.to(torch.long)] * scales.float().gather(
        -1,
        group_idx.view(1, 1, 1, 1, head_size).expand(*idx.shape[:-1], head_size),
    )
    torch.testing.assert_close(
        dequant.float(),
        expected.to(torch.float16).float(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("kv_cache_dtype", "kv_quant_mode", "head_size"),
    [
        ("int8_per_token_head", KVQuantMode.INT8_PER_TOKEN_HEAD, 4),
        ("int4_per_token_head", KVQuantMode.INT4_PER_TOKEN_HEAD, 40),
    ],
)
def test_flashinfer_impl_forward_uses_staged_kv_cache_without_scales(
    kv_cache_dtype: str,
    kv_quant_mode: KVQuantMode,
    head_size: int,
    default_vllm_config,
    monkeypatch,
):
    if kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
        quant = torch.tensor(
            [[[[[1, -2, 3, -4]]], [[[5, -6, 7, -8]]]]],
            dtype=torch.int8,
        )
        scales = torch.tensor([2.0, 0.5], dtype=torch.float32).view(1, 2, 1, 1, 1)
        kv_cache = _build_int8_inline_cache(quant, scales)
    else:
        idx = torch.tensor(
            [
                [
                    [
                        [
                            [
                                0,
                                1,
                                8,
                                9,
                                2,
                                10,
                                3,
                                11,
                                4,
                                12,
                                5,
                                13,
                                6,
                                14,
                                7,
                                15,
                                0,
                                1,
                                8,
                                9,
                                2,
                                10,
                                3,
                                11,
                                4,
                                12,
                                5,
                                13,
                                6,
                                14,
                                7,
                                15,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                            ]
                        ]
                    ],
                    [
                        [
                            [
                                15,
                                14,
                                13,
                                12,
                                11,
                                10,
                                9,
                                8,
                                7,
                                6,
                                5,
                                4,
                                3,
                                2,
                                1,
                                0,
                                15,
                                14,
                                13,
                                12,
                                11,
                                10,
                                9,
                                8,
                                7,
                                6,
                                5,
                                4,
                                3,
                                2,
                                1,
                                0,
                                8,
                                7,
                                6,
                                5,
                                4,
                                3,
                                2,
                                1,
                            ]
                        ]
                    ],
                ]
            ],
            dtype=torch.uint8,
        )
        packed = _pack_int4(idx)
        scales = torch.tensor([1.0, 0.25, 0.5, 2.0], dtype=torch.float16).view(
            1, 2, 1, 1, 2
        )
        kv_cache = _build_int4_inline_cache(packed, scales, head_size=head_size)

    query = torch.randn(1, 1, head_size, dtype=torch.float16)
    key = torch.randn(1, 1, head_size, dtype=torch.float16)
    value = torch.randn(1, 1, head_size, dtype=torch.float16)
    output = torch.empty_like(query)

    class DummyPrefillWrapper:
        def __init__(self):
            self._window_left = -1
            self._logits_soft_cap = 0.0
            self._sm_scale = head_size**-0.5
            self._causal = True
            self.call_args = None

        def run(self, q, paged_kv_cache, **kwargs):
            self.call_args = (q, paged_kv_cache, kwargs)
            kwargs["out"].zero_()
            return kwargs["out"]

    monkeypatch.setattr(
        flashinfer_backend_module,
        "BatchPrefillWithPagedKVCacheWrapper",
        DummyPrefillWrapper,
    )
    wrapper = DummyPrefillWrapper()

    attn_metadata = FlashInferMetadata(
        num_actual_tokens=1,
        slot_mapping=torch.tensor([0], dtype=torch.long),
        q_data_type=query.dtype,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=1,
        prefill=FIPrefill(wrapper=wrapper),
        decode=None,
        use_cascade=False,
        cascade_wrapper=None,
    )

    impl = FlashInferImpl(
        num_heads=1,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )

    set_kv_cache_layout("NHD")
    try:
        impl.forward(
            MagicMock(_k_scale_float=1.0, _v_scale_float=1.0),
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
    finally:
        set_kv_cache_layout(None)

    assert wrapper.call_args is not None
    _, staged_kv_cache, kwargs = wrapper.call_args
    assert staged_kv_cache.dtype == query.dtype
    assert staged_kv_cache.shape[-1] == head_size
    assert kwargs["k_scale"] is None
    assert kwargs["v_scale"] is None


@pytest.mark.parametrize(
    ("kv_cache_dtype", "kv_quant_mode", "head_size"),
    [
        ("int8_per_token_head", KVQuantMode.INT8_PER_TOKEN_HEAD, 128),
        ("int4_per_token_head", KVQuantMode.INT4_PER_TOKEN_HEAD, 128),
    ],
)
def test_flashinfer_impl_forward_staged_quant_runs_real_prefill_wrapper_on_cuda(
    kv_cache_dtype: str,
    kv_quant_mode: KVQuantMode,
    head_size: int,
    default_vllm_config,
):
    _require_flashinfer_cuda_runtime()

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    batch_size = 1
    qo_len = 3
    kv_len = 8
    page_size = 4
    num_pages = (kv_len + page_size - 1) // page_size
    num_qo_heads = 2
    num_kv_heads = 2

    if kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
        quant = torch.randint(
            -8,
            8,
            (num_pages, 2, page_size, num_kv_heads, head_size),
            dtype=torch.int8,
            device=device,
        )
        scales = torch.rand(
            num_pages, 2, page_size, num_kv_heads, 1, dtype=torch.float32, device=device
        )
        kv_cache = _build_int8_inline_cache(quant, scales)
    else:
        packed_width = head_size // 2
        idx = torch.randint(
            0,
            16,
            (num_pages, 2, page_size, num_kv_heads, head_size),
            dtype=torch.uint8,
            device=device,
        )
        packed = _pack_int4(idx)
        assert packed.shape[-1] == packed_width
        scales = torch.rand(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            get_int4_num_scale_groups(head_size),
            dtype=torch.float16,
            device=device,
        )
        kv_cache = _build_int4_inline_cache(packed, scales, head_size=head_size)

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    q_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [(kv_len - 1) % page_size + 1], dtype=torch.int32, device=device
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_size,
        page_size,
        causal=True,
        sm_scale=head_size**-0.5,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    query = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    key = torch.randn(
        batch_size * qo_len,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    value = torch.randn(
        batch_size * qo_len,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    output = torch.empty_like(query)

    attn_metadata = FlashInferMetadata(
        num_actual_tokens=batch_size * qo_len,
        slot_mapping=torch.arange(batch_size * qo_len, dtype=torch.long, device=device),
        q_data_type=query.dtype,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=batch_size,
        num_prefill_tokens=batch_size * qo_len,
        prefill=FIPrefill(wrapper=wrapper),
        decode=None,
        use_cascade=False,
        cascade_wrapper=None,
    )
    impl = FlashInferImpl(
        num_heads=num_qo_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )

    staged_expected_cache = staged_dequantize_paged_kv_cache(
        kv_cache,
        head_size=head_size,
        kv_quant_mode=kv_quant_mode,
        out_dtype=query.dtype,
    )
    expected = wrapper.run(query, staged_expected_cache)

    set_kv_cache_layout("NHD")
    try:
        actual = impl.forward(
            MagicMock(_q_scale_float=2.0, _k_scale_float=3.0, _v_scale_float=4.0),
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
    finally:
        set_kv_cache_layout(None)

    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
