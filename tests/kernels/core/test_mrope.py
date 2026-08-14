# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import (
    MRotaryEmbedding,
    apply_interleaved_rope,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.utils.torch_utils import set_random_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_position_embeddings: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Generate test data for given configuration."""
    set_random_seed(42)
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(
        0, max_position_embeddings // 4, (3, num_tokens), device=device
    )

    # Create query and key tensors
    query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)

    return positions, query, key


class MRoPETestInfo(NamedTuple):
    model_name: str
    is_neox_style: bool = True
    # https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L1317
    atol: float = 1e-2
    rtol: float = 1.6e-2
    marks: list[pytest.MarkDecorator] = []


MODELS_TO_TEST = [
    MRoPETestInfo(
        model_name="zai-org/GLM-4.1V-9B-Thinking",
        is_neox_style=False,
    ),
    MRoPETestInfo(model_name="Qwen/Qwen2-VL-7B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen2-VL-72B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen2.5-VL-72B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen3-VL-4B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"),
]

num_tokens_list = [11, 8192]


def test_apply_interleaved_rope():
    mrope_section = [3, 1, 1]
    x = torch.tensor(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )

    result = apply_interleaved_rope(x, mrope_section)

    expected = torch.tensor([[0, 11, 22, 3, 4], [5, 16, 27, 8, 9]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only test."
)
def test_apply_interleaved_rope_torch_compile():
    mrope_section = [24, 20, 20]
    num_tokens = 8192
    rotary_dim = sum(mrope_section) * 2
    cache = torch.randn(
        3,
        num_tokens,
        rotary_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    x = cache[..., : rotary_dim // 2]

    expected = apply_interleaved_rope(x, mrope_section)
    compiled_fn = torch.compile(
        apply_interleaved_rope,
        backend="inductor",
        fullgraph=True,
    )

    result = compiled_fn(x, mrope_section)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize(
    "model_info, model_name",
    [
        pytest.param(test_config, test_config.model_name, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_mrope(
    default_vllm_config,
    model_name: str,
    model_info: MRoPETestInfo,
    tp_size: int,
    dtype: torch.dtype,
    num_tokens: int,
):
    atol = model_info.atol
    rtol = model_info.rtol

    config = get_config(model_name, False).get_text_config()

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else config.hidden_size // total_num_heads
    )
    is_neox_style = model_info.is_neox_style

    max_position = config.max_position_embeddings

    mrope_helper_class = get_rope(
        head_size=head_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=config.rope_parameters,
        dtype=dtype,
    ).to(device=device)

    # create q k v input tensors
    # create rotary pos emb input tensors
    positions, query, key = generate_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, max_position, dtype, device
    )

    query_native, key_native = mrope_helper_class.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    query_cuda, key_cuda = mrope_helper_class.forward_cuda(
        positions,
        query.clone(),
        key.clone(),
    )

    torch.testing.assert_close(query_native, query_cuda, atol=atol, rtol=rtol)
    torch.testing.assert_close(key_native, key_cuda, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize(
    "model_info, model_name",
    [
        pytest.param(test_config, test_config.model_name, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_mrope_torch_compile_tracing(
    default_vllm_config,
    model_name: str,
    model_info: MRoPETestInfo,
    tp_size: int,
    dtype: torch.dtype,
    num_tokens: int,
):
    atol = model_info.atol
    rtol = model_info.rtol

    config = get_config(model_name, False).get_text_config()

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else config.hidden_size // total_num_heads
    )
    is_neox_style = model_info.is_neox_style
    max_position = config.max_position_embeddings

    mrope_helper_class = get_rope(
        head_size=head_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=config.rope_parameters,
        dtype=dtype,
    ).to(device=device)

    # Generate test data
    positions, query, key = generate_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, max_position, dtype, device
    )

    # Create a wrapper that makes the in-place function appear functional
    def functional_forward_cuda(pos, q, k):
        """Wrapper that converts in-place operation to functional style

        CUDA Graph does not support in-place operations.
        This wrapper creates working copies of the
        input tensors and modifies them.
        """
        q_work = q.clone()  # Create working copies
        k_work = k.clone()
        # Your in-place function modifies q_work and k_work
        mrope_helper_class.forward_cuda(pos, q_work, k_work)
        return q_work, k_work  # Return the modified tensors

    # Get reference results
    query_native, key_native = mrope_helper_class.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    try:
        compiled_forward_cuda = torch.compile(
            functional_forward_cuda,
            fullgraph=True,
            backend="inductor",
            mode="reduce-overhead",
            dynamic=False,
        )

        # Run compiled version
        query_compiled_cuda, key_compiled_cuda = compiled_forward_cuda(
            positions,
            query,
            key,
        )

        # Run original version for comparison
        query_cuda = query.clone()
        key_cuda = key.clone()
        mrope_helper_class.forward_cuda(positions, query_cuda, key_cuda)

        # Verify results
        torch.testing.assert_close(
            query_compiled_cuda, query_cuda, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(key_compiled_cuda, key_cuda, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            query_compiled_cuda, query_native, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(key_compiled_cuda, key_native, atol=atol, rtol=rtol)

        print("✓ forward_cuda successfully traced with torch.compile inductor")

    except Exception as e:
        pytest.fail(f"forward_cuda failed to trace with torch.compile inductor: {e}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scaling_factor", [None, 2.0])
def test_mrope_cache_bound_preserves_legacy_values(
    default_vllm_config,
    dtype: torch.dtype,
    scaling_factor: float | None,
):
    max_position = 16
    semantic_max_position = max_position * 4
    cache_max_position = 16
    kwargs = {
        "head_size": 8,
        "rotary_dim": 8,
        "max_position_embeddings": max_position,
        "base": 10000,
        "is_neox_style": True,
        "dtype": dtype,
        "mrope_section": [2, 1, 1],
        "scaling_factor": scaling_factor,
    }

    legacy = MRotaryEmbedding(**kwargs)
    bounded = MRotaryEmbedding(
        **kwargs,
        mrope_cache_max_position=cache_max_position,
    )

    if scaling_factor is None:
        reference = RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=semantic_max_position,
            base=10000,
            is_neox_style=True,
            dtype=dtype,
        )
        expected_legacy_cache_size = semantic_max_position
    else:
        reference = YaRNScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=semantic_max_position,
            base=10000,
            is_neox_style=True,
            scaling_factor=scaling_factor,
            dtype=dtype,
        )
        expected_legacy_cache_size = int(semantic_max_position * scaling_factor)

    assert legacy.max_position_embeddings == semantic_max_position
    assert bounded.max_position_embeddings == semantic_max_position
    assert legacy.cos_sin_cache.shape == (expected_legacy_cache_size, 8)
    assert bounded.cos_sin_cache.shape == (cache_max_position, 8)
    assert torch.equal(legacy.cos_sin_cache, reference.cos_sin_cache)
    assert torch.equal(
        bounded.cos_sin_cache,
        legacy.cos_sin_cache[:cache_max_position],
    )

    positions = torch.tensor(
        [[0, 3, cache_max_position - 1]] * 3,
        dtype=torch.long,
    )
    query = torch.randn(3, 8, dtype=dtype)
    key = torch.randn(3, 8, dtype=dtype)
    legacy_query, legacy_key = legacy.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )
    bounded_query, bounded_key = bounded.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )
    assert torch.equal(legacy_query, bounded_query)
    assert torch.equal(legacy_key, bounded_key)

    with pytest.raises(IndexError):
        bounded.forward_native(
            torch.full((3, 1), cache_max_position, dtype=torch.long),
            query[:1].clone(),
            key[:1].clone(),
        )


def test_mrope_cache_key_includes_physical_capacity(default_vllm_config):
    kwargs = {
        "head_size": 8,
        "max_position": 16,
        "rope_parameters": {"mrope_section": [2, 1, 1]},
        "dtype": torch.float32,
    }

    _ROPE_DICT.clear()
    try:
        legacy = get_rope(**kwargs)
        bounded = get_rope(**kwargs, mrope_cache_max_position=16)
        bounded_again = get_rope(**kwargs, mrope_cache_max_position=16)
        differently_bounded = get_rope(**kwargs, mrope_cache_max_position=32)

        assert legacy is not bounded
        assert bounded is bounded_again
        assert differently_bounded is not bounded
        assert legacy.cos_sin_cache.shape[0] == 64
        assert bounded.cos_sin_cache.shape[0] == 16
        assert differently_bounded.cos_sin_cache.shape[0] == 32
    finally:
        _ROPE_DICT.clear()


def test_interleaved_mrope_cache_bound(default_vllm_config):
    kwargs = {
        "head_size": 8,
        "max_position": 16,
        "rope_parameters": {
            "rope_type": "openpangu",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "dtype": torch.float32,
    }

    _ROPE_DICT.clear()
    try:
        legacy = get_rope(**kwargs)
        bounded = get_rope(**kwargs, mrope_cache_max_position=16)

        assert legacy.cos_sin_cache.shape[0] == 64
        assert bounded.cos_sin_cache.shape[0] == 16
        assert torch.equal(bounded.cos_sin_cache, legacy.cos_sin_cache[:16])
    finally:
        _ROPE_DICT.clear()


def test_yarn_mrope_cache_bound_through_factory(default_vllm_config):
    scaling_factor = 2.1
    kwargs = {
        "head_size": 8,
        "max_position": 64,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": scaling_factor,
            "original_max_position_embeddings": 16,
            "mrope_section": [2, 1, 1],
        },
        "dtype": torch.float32,
    }

    _ROPE_DICT.clear()
    try:
        legacy = get_rope(**kwargs)
        bounded = get_rope(**kwargs, mrope_cache_max_position=16)

        assert legacy.max_position_embeddings == 64
        assert bounded.max_position_embeddings == 64
        assert legacy.cos_sin_cache.shape[0] == 135
        assert bounded.cos_sin_cache.shape[0] == 16
        assert torch.equal(bounded.cos_sin_cache, legacy.cos_sin_cache[:16])
    finally:
        _ROPE_DICT.clear()


@pytest.mark.parametrize("mrope_position_delta", [0, -3])
def test_mrope_completion_positions_stay_within_cache_bound(mrope_position_delta):
    cache_max_position = 16
    positions = np.empty((3, 3), dtype=np.int64)
    MRotaryEmbedding.get_next_input_positions_tensor(
        out=positions,
        out_offset=0,
        mrope_position_delta=mrope_position_delta,
        context_len=cache_max_position - 3,
        num_new_tokens=3,
    )

    assert positions.min() >= 0
    assert positions.max() < cache_max_position
