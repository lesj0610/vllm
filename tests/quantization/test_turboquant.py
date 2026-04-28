# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for paper-faithful TurboQuant `_nc` presets.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
from vllm.model_executor.layers.quantization.turboquant.quantizer import (
    generate_qjl_projection,
    generate_random_orthogonal,
)

ALL_PRESETS = list(TQ_PRESETS.keys())
CUDA_AVAILABLE = torch.cuda.is_available()
QJL_ALPHA = 0.25 * math.sqrt(math.pi / 2.0)

PRESET_EXPECTED = {
    "turboquant_4bit_nc": dict(
        key_quant_bits=4,
        key_mse_bits=3,
        qjl_bits=1,
        value_quant_bits=4,
        n_centroids=8,
        centroid_bits=3,
        key_mse_packed_size=48,
        key_qjl_packed_size=16,
        key_packed_size=66,
        value_packed_size=68,
        slot_size=134,
        slot_size_aligned=134,
        residual_norm_quant_max=0.40,
    ),
    "turboquant_k3v4_nc": dict(
        key_quant_bits=3,
        key_mse_bits=2,
        qjl_bits=1,
        value_quant_bits=4,
        n_centroids=4,
        centroid_bits=2,
        key_mse_packed_size=32,
        key_qjl_packed_size=16,
        key_packed_size=50,
        value_packed_size=68,
        slot_size=118,
        slot_size_aligned=118,
        residual_norm_quant_max=0.55,
    ),
    "turboquant_3bit_nc": dict(
        key_quant_bits=3,
        key_mse_bits=2,
        qjl_bits=1,
        value_quant_bits=3,
        n_centroids=4,
        centroid_bits=2,
        key_mse_packed_size=32,
        key_qjl_packed_size=16,
        key_packed_size=50,
        value_packed_size=52,
        slot_size=102,
        slot_size_aligned=102,
        residual_norm_quant_max=0.50,
    ),
}


# ============================================================================
# Config tests
# ============================================================================


class TestTurboQuantConfig:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, TurboQuantConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("turboquant_k8v4", head_dim=128)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_expected_layout_values(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_quant_bits == exp["key_quant_bits"]
        assert cfg.key_mse_bits == exp["key_mse_bits"]
        assert cfg.qjl_bits == exp["qjl_bits"]
        assert cfg.value_quant_bits == exp["value_quant_bits"]
        assert cfg.n_centroids == exp["n_centroids"]
        assert cfg.centroid_bits == exp["centroid_bits"]
        assert cfg.key_mse_packed_size == exp["key_mse_packed_size"]
        assert cfg.key_qjl_packed_size == exp["key_qjl_packed_size"]
        assert cfg.key_packed_size == exp["key_packed_size"]
        assert cfg.value_packed_size == exp["value_packed_size"]
        assert cfg.slot_size == exp["slot_size"]
        assert cfg.slot_size_aligned == exp["slot_size_aligned"]
        assert cfg.residual_norm_quant_max == pytest.approx(
            exp["residual_norm_quant_max"]
        )

    @pytest.mark.parametrize(
        ("arch", "expected"),
        [
            ("Qwen3ForCausalLM", (-2.0, 9.0)),
            ("Qwen3_5ForConditionalGeneration", (-2.0, 9.0)),
            ("Qwen3_5MoeForConditionalGeneration", (-2.0, 9.0)),
            ("Gemma4ForConditionalGeneration", (-3.0, 3.0)),
            ("UnknownArch", (-2.0, 10.0)),
        ],
    )
    def test_key_norm_log_range_for_architecture(self, arch, expected):
        assert TurboQuantConfig.get_key_norm_log_range_for_arch(arch) == expected

    def test_long_prefill_backend_env_is_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import vllm.envs as envs

        envs.disable_envs_cache()
        monkeypatch.delenv("VLLM_TQ_LONG_PREFILL_BACKEND", raising=False)
        assert envs.VLLM_TQ_LONG_PREFILL_BACKEND == "stream"

        monkeypatch.setenv("VLLM_TQ_LONG_PREFILL_BACKEND", "native")
        envs.disable_envs_cache()
        assert envs.VLLM_TQ_LONG_PREFILL_BACKEND == "native"

        monkeypatch.setenv("VLLM_TQ_LONG_PREFILL_BACKEND", "invalid")
        envs.disable_envs_cache()
        with pytest.raises(ValueError, match="VLLM_TQ_LONG_PREFILL_BACKEND"):
            _ = envs.VLLM_TQ_LONG_PREFILL_BACKEND

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
    def test_slot_size_matches_components(self, preset, head_dim):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size
        assert cfg.key_packed_size == (
            cfg.key_mse_packed_size
            + cfg.key_qjl_packed_size
            + cfg.key_norm_packed_size
            + cfg.residual_norm_packed_size
        )
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0


# ============================================================================
# Centroid tests
# ============================================================================


class TestCentroids:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_centroids_shape(self, bits, expected_n):
        c = get_centroids(128, bits)
        assert c.shape == (expected_n,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        c = get_centroids(128, bits)
        assert torch.all(c[:-1] < c[1:])

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_lloyd_max_boundaries_are_midpoints(self, bits):
        centroids, boundaries = solve_lloyd_max(128, bits)
        expected = (centroids[:-1] + centroids[1:]) / 2.0
        assert torch.allclose(boundaries, expected, atol=1e-6)


# ============================================================================
# Random transform tests
# ============================================================================


class TestRandomTransforms:
    def test_random_orthogonal_is_orthonormal(self):
        pi = generate_random_orthogonal(64, seed=42)
        eye = pi @ pi.T
        assert torch.allclose(eye, torch.eye(64), atol=1e-5)

    def test_random_orthogonal_is_deterministic(self):
        p1 = generate_random_orthogonal(64, seed=42)
        p2 = generate_random_orthogonal(64, seed=42)
        assert torch.equal(p1, p2)

    def test_qjl_projection_is_deterministic(self):
        s1 = generate_qjl_projection(64, seed=42)
        s2 = generate_qjl_projection(64, seed=42)
        s3 = generate_qjl_projection(64, seed=43)
        assert torch.equal(s1, s2)
        assert not torch.equal(s1, s3)
        assert s1.shape == (64, 64)


# ============================================================================
# Store / decode tests
# ============================================================================


def _make_tq_state(
    d: int,
    preset: str,
    device: torch.device,
    architecture: str = "Qwen3ForCausalLM",
):
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=d)
    pi = generate_random_orthogonal(d, seed=42, device=str(device)).float()
    pit = pi.T.contiguous()
    s = generate_qjl_projection(d, seed=43, device=str(device)).float()
    st = s.T.contiguous()
    centroids, midpoints = solve_lloyd_max(d, cfg.centroid_bits)
    key_norm_log_min, key_norm_log_max = (
        TurboQuantConfig.get_key_norm_log_range_for_arch(architecture)
    )
    lut = torch.exp2(
        torch.linspace(
            key_norm_log_min,
            key_norm_log_max,
            256,
            device=device,
            dtype=torch.float32,
        )
    )
    return (
        cfg,
        pi,
        pit,
        s,
        st,
        centroids.to(device),
        midpoints.to(device),
        torch.tensor(key_norm_log_min, device=device, dtype=torch.float32),
        torch.tensor(key_norm_log_max, device=device, dtype=torch.float32),
        lut,
    )


def _reconstruct_quantized_key(
    key: torch.Tensor,
    pit: torch.Tensor,
    pi: torch.Tensor,
    s: torch.Tensor,
    st: torch.Tensor,
    centroids: torch.Tensor,
    midpoints: torch.Tensor,
    *,
    quantize_key_norm: bool,
    key_norm_log_min: float,
    key_norm_log_max: float,
    quantize_residual_norm: bool,
    residual_norm_cap: float,
) -> torch.Tensor:
    key_f = key.float()
    key_norm = key_f.norm(dim=-1, keepdim=True)
    key_norm_out = key_norm
    if quantize_key_norm:
        key_norm_log = torch.log2(key_norm.clamp_min(1e-8))
        key_norm_out = torch.exp2(
            key_norm_log_min
            + (
                torch.clamp(
                    torch.round(
                        (key_norm_log - key_norm_log_min)
                        * (255.0 / (key_norm_log_max - key_norm_log_min))
                    ),
                    0,
                    255,
                )
                * ((key_norm_log_max - key_norm_log_min) / 255.0)
            )
        )
    x_hat = key_f / key_norm.clamp_min(1e-8)
    y = x_hat @ pit
    idx = torch.bucketize(y, midpoints)
    y_tilde = centroids[idx]
    xhat_mse = y_tilde @ pi
    residual = x_hat - xhat_mse
    residual_norm = residual.norm(dim=-1, keepdim=True)
    if quantize_residual_norm:
        residual_norm = torch.clamp(
            torch.round(residual_norm * (255.0 / residual_norm_cap)),
            0,
            255,
        ) * (residual_norm_cap / 255.0)
    qjl_sign = ((residual @ st) >= 0).float() * 2.0 - 1.0
    xhat_qjl = (QJL_ALPHA / key.shape[-1]) * residual_norm * (qjl_sign @ s)
    return key_norm_out * (xhat_mse + xhat_qjl)


def _attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.einsum("qhd,khd->hqk", query.float(), key.float()) / math.sqrt(
        query.shape[-1]
    )


def _mean_relative_score_error(ref: torch.Tensor, other: torch.Tensor) -> float:
    denom = ref.abs().clamp_min(1e-6)
    return ((other - ref).abs() / denom).mean().item()


def _mean_attention_l1(ref_scores: torch.Tensor, other_scores: torch.Tensor) -> float:
    ref_weights = F.softmax(ref_scores, dim=-1)
    other_weights = F.softmax(other_scores, dim=-1)
    return (ref_weights - other_weights).abs().sum(dim=-1).mean().item()


def test_materialized_qjl_key_matches_decode_score_formula():
    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    d = 128
    hk = 3
    seq_len = 7
    cfg, pi, pit, s, st, centroids, midpoints, _kn_min, _kn_max, _kn_lut = (
        _make_tq_state(d, "turboquant_4bit_nc", device)
    )

    torch.manual_seed(2026)
    key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float32)
    query = torch.randn(2, hk, d, device=device, dtype=torch.float32)

    key_f = key.float()
    key_norm = key_f.norm(dim=-1, keepdim=True)
    x_hat = key_f / key_norm.clamp_min(1e-8)
    y = x_hat @ pit
    idx = torch.bucketize(y, midpoints)
    y_tilde = centroids[idx]
    xhat_mse = y_tilde @ pi
    residual = x_hat - xhat_mse
    residual_norm = residual.norm(dim=-1, keepdim=True)
    residual_norm = torch.clamp(
        torch.round(residual_norm * (255.0 / cfg.residual_norm_quant_max)),
        0,
        255,
    ) * (cfg.residual_norm_quant_max / 255.0)
    qjl_sign = ((residual @ st) >= 0).float() * 2.0 - 1.0

    k_dequant = key_norm * (xhat_mse + (QJL_ALPHA / d) * residual_norm * (qjl_sign @ s))
    score_dequant = _attention_scores(query, k_dequant)

    q_rot = query.float() @ pit
    q_qjl = query.float() @ st
    term_base = torch.einsum("qhd,khd->hqk", q_rot, y_tilde)
    term_qjl = torch.einsum("qhd,khd->hqk", q_qjl, qjl_sign)
    key_norm_h = key_norm.squeeze(-1).transpose(0, 1).unsqueeze(1)
    residual_norm_h = residual_norm.squeeze(-1).transpose(0, 1).unsqueeze(1)
    score_decode = (
        key_norm_h * (term_base + (QJL_ALPHA / d) * residual_norm_h * term_qjl)
    ) / math.sqrt(d)

    assert torch.allclose(score_dequant, score_decode, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestStoreDecodeRoundTrip:
    @pytest.mark.parametrize(
        ("preset", "max_rel_error", "max_attn_l1"),
        [
            ("turboquant_4bit_nc", 0.005, 0.01),
            ("turboquant_k3v4_nc", 0.005, 0.01),
            ("turboquant_3bit_nc", 0.005, 0.01),
        ],
    )
    def test_residual_norm_uint8_is_low_drift(
        self,
        preset,
        max_rel_error,
        max_attn_l1,
    ):
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        d = 128
        hk = 4
        seq_len = 16

        cfg, pi, pit, s, st, centroids, midpoints, _kn_min, _kn_max, _kn_lut = (
            _make_tq_state(d, preset, device)
        )

        torch.manual_seed(123)
        key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float32)
        query = torch.randn(1, hk, d, device=device, dtype=torch.float32)

        k_ref = _reconstruct_quantized_key(
            key,
            pit,
            pi,
            s,
            st,
            centroids,
            midpoints,
            quantize_residual_norm=False,
            quantize_key_norm=False,
            key_norm_log_min=float(_kn_min.item()),
            key_norm_log_max=float(_kn_max.item()),
            residual_norm_cap=cfg.residual_norm_quant_max,
        )
        k_u8 = _reconstruct_quantized_key(
            key,
            pit,
            pi,
            s,
            st,
            centroids,
            midpoints,
            quantize_residual_norm=True,
            quantize_key_norm=False,
            key_norm_log_min=float(_kn_min.item()),
            key_norm_log_max=float(_kn_max.item()),
            residual_norm_cap=cfg.residual_norm_quant_max,
        )

        scores_ref = _attention_scores(query, k_ref)
        scores_u8 = _attention_scores(query, k_u8)

        assert _mean_relative_score_error(scores_ref, scores_u8) < max_rel_error
        assert _mean_attention_l1(scores_ref, scores_u8) < max_attn_l1

    @pytest.mark.parametrize(
        ("architecture", "key_scale", "max_rel_error", "max_attn_l1"),
        [
            ("Qwen3ForCausalLM", 1.0, 0.02, 0.03),
            ("Gemma4ForConditionalGeneration", 0.18, 0.02, 0.03),
        ],
    )
    def test_key_norm_log_uint8_is_low_drift(
        self,
        architecture,
        key_scale,
        max_rel_error,
        max_attn_l1,
    ):
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        preset = "turboquant_4bit_nc"
        d = 128
        hk = 4
        seq_len = 16

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            _key_norm_lut,
        ) = _make_tq_state(d, preset, device, architecture=architecture)

        torch.manual_seed(1234)
        key = (
            torch.randn(seq_len, hk, d, device=device, dtype=torch.float32) * key_scale
        )
        query = torch.randn(1, hk, d, device=device, dtype=torch.float32)

        k_ref = _reconstruct_quantized_key(
            key,
            pit,
            pi,
            s,
            st,
            centroids,
            midpoints,
            quantize_key_norm=False,
            key_norm_log_min=float(key_norm_log_min.item()),
            key_norm_log_max=float(key_norm_log_max.item()),
            quantize_residual_norm=False,
            residual_norm_cap=cfg.residual_norm_quant_max,
        )
        k_u8 = _reconstruct_quantized_key(
            key,
            pit,
            pi,
            s,
            st,
            centroids,
            midpoints,
            quantize_key_norm=True,
            key_norm_log_min=float(key_norm_log_min.item()),
            key_norm_log_max=float(key_norm_log_max.item()),
            quantize_residual_norm=False,
            residual_norm_cap=cfg.residual_norm_quant_max,
        )

        scores_ref = _attention_scores(query, k_ref)
        scores_u8 = _attention_scores(query, k_u8)

        assert _mean_relative_score_error(scores_ref, scores_u8) < max_rel_error
        assert _mean_attention_l1(scores_ref, scores_u8) < max_attn_l1

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
    def test_decode_matches_dequant_reference(self, preset, input_dtype):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            full_dequant_turboquant_cache,
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device("cuda")
        d = 128
        hk = 4
        hq = 4
        seq_len = 4
        block_size = 16
        num_blocks = 1

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            key_norm_lut,
        ) = _make_tq_state(d, preset, device)

        torch.manual_seed(123)
        key = torch.randn(seq_len, hk, d, device=device, dtype=input_dtype)
        value = torch.randn(seq_len, hk, d, device=device, dtype=input_dtype)
        query = torch.randn(1, hq, d, device=device, dtype=input_dtype)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        out = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            key_norm_lut=key_norm_lut,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            PiT=pit,
            max_num_kv_splits=4,
        )

        k_ref, v_ref = full_dequant_turboquant_cache(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            S=s,
        )
        q = query.float().transpose(0, 1).unsqueeze(0)
        k = k_ref[0].unsqueeze(0)
        v = v_ref[0].unsqueeze(0)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(d)
        )[0].transpose(0, 1)

        diff = (out.float() - ref.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(1, -1), ref.float().reshape(1, -1)
        ).item()
        assert diff.mean().item() < 0.15
        assert diff.max().item() < 0.95
        assert cos > 0.96
        assert torch.isfinite(k_ref).all()
        assert torch.isfinite(v_ref).all()

    def test_decode_respects_sliding_window(self):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            full_dequant_turboquant_cache,
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device("cuda")
        preset = "turboquant_4bit_nc"
        d = 128
        hk = 2
        hq = 2
        seq_len = 6
        sliding_window = 3
        block_size = 16
        num_blocks = 1

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            key_norm_lut,
        ) = _make_tq_state(
            d,
            preset,
            device,
            architecture="Gemma4ForConditionalGeneration",
        )

        key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        query = torch.randn(1, hq, d, device=device, dtype=torch.float16)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        out = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            key_norm_lut=key_norm_lut,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            PiT=pit,
            max_num_kv_splits=4,
            sliding_window=sliding_window,
        )

        k_ref, v_ref = full_dequant_turboquant_cache(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            S=s,
        )
        q = query.float().transpose(0, 1).unsqueeze(0)
        k = k_ref[0, :, -sliding_window:, :].unsqueeze(0)
        v = v_ref[0, :, -sliding_window:, :].unsqueeze(0)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(d)
        )[0].transpose(0, 1)

        diff = (out.float() - ref.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(1, -1), ref.float().reshape(1, -1)
        ).item()
        assert diff.mean().item() < 0.13
        assert diff.max().item() < 0.55
        assert cos > 0.965

    def test_page_dequant_matches_full_reference(self):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            dequant_turboquant_cache_pages,
            full_dequant_turboquant_cache,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device("cuda")
        preset = "turboquant_4bit_nc"
        d = 128
        hk = 2
        seq_len = 20
        block_size = 8
        num_blocks = math.ceil(seq_len / block_size)

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            _key_norm_lut,
        ) = _make_tq_state(d, preset, device)

        torch.manual_seed(456)
        key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
            1, num_blocks
        )
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        k_ref, v_ref = full_dequant_turboquant_cache(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            S=s,
        )
        k_pages, v_pages, compact_bt = dequant_turboquant_cache_pages(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            Pi=pi,
            S=s,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            output_dtype=torch.float32,
        )

        k_flat = k_pages.reshape(-1, hk, d)[:seq_len].transpose(0, 1)
        v_flat = v_pages.reshape(-1, hk, d)[:seq_len].transpose(0, 1)
        assert torch.equal(
            compact_bt, torch.arange(num_blocks, device=device).view(1, num_blocks)
        )
        assert torch.allclose(k_flat, k_ref[0], atol=0, rtol=0)
        assert torch.allclose(v_flat, v_ref[0], atol=0, rtol=0)

    def test_transient_page_dequant_works_with_flash_attention(self):
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
            is_flash_attn_varlen_func_available,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            dequant_turboquant_cache_pages,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        if not is_flash_attn_varlen_func_available():
            pytest.skip("flash_attn_varlen_func is not available")

        device = torch.device("cuda")
        preset = "turboquant_4bit_nc"
        d = 128
        hk = 2
        q_len = 5
        seq_len = 40
        cached_len = seq_len - q_len
        block_size = 16
        num_blocks = math.ceil(seq_len / block_size)

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            _key_norm_lut,
        ) = _make_tq_state(d, preset, device)

        torch.manual_seed(789)
        key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        query = torch.randn(q_len, hk, d, device=device, dtype=torch.float16)
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
            1, num_blocks
        )
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        k_pages, v_pages, compact_bt = dequant_turboquant_cache_pages(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            Pi=pi,
            S=s,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            output_dtype=query.dtype,
        )

        out = torch.empty_like(query)
        flash_attn_varlen_func(
            q=query,
            k=k_pages,
            v=v_pages,
            out=out,
            cu_seqlens_q=torch.tensor([0, q_len], device=device, dtype=torch.int32),
            max_seqlen_q=q_len,
            seqused_k=torch.tensor([seq_len], device=device, dtype=torch.int32),
            max_seqlen_k=seq_len,
            softmax_scale=1.0 / math.sqrt(d),
            causal=True,
            block_table=compact_bt,
            fa_version=get_flash_attn_version(head_size=d),
        )

        k_flat = k_pages.reshape(-1, hk, d)[:seq_len].transpose(0, 1).unsqueeze(0)
        v_flat = v_pages.reshape(-1, hk, d)[:seq_len].transpose(0, 1).unsqueeze(0)
        q_ref = query.transpose(0, 1).unsqueeze(0)
        key_positions = torch.arange(seq_len, device=device)
        query_positions = cached_len + torch.arange(q_len, device=device)
        attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        ref = F.scaled_dot_product_attention(
            q_ref,
            k_flat,
            v_flat,
            attn_mask=attn_mask,
            scale=1.0 / math.sqrt(d),
        )[0].transpose(0, 1)

        diff = (out.float() - ref.float()).abs()
        assert diff.mean().item() < 0.03
        assert diff.max().item() < 0.3

    @pytest.mark.parametrize(
        (
            "d",
            "hk",
            "hq",
            "q_len",
            "cached_len",
            "block_size",
            "sliding_window",
            "architecture",
            "key_scale",
        ),
        [
            (256, 2, 2, 17, 5, 16, None, "Qwen3ForCausalLM", 1.0),
            (256, 2, 4, 21, 7, 8, None, "Qwen3ForCausalLM", 1.0),
            (256, 2, 8, 13, 9, 8, None, "Qwen3ForCausalLM", 1.0),
            (256, 2, 4, 19, 13, 8, 11, "Qwen3ForCausalLM", 1.0),
            (256, 2, 12, 128, 4096, 128, None, "Qwen3ForCausalLM", 1.0),
            (512, 2, 4, 7, 5, 8, None, "Gemma4ForConditionalGeneration", 0.18),
            (512, 2, 4, 19, 13, 8, 11, "Gemma4ForConditionalGeneration", 0.18),
        ],
    )
    def test_native_prefill_matches_decode_loop(
        self,
        d: int,
        hk: int,
        hq: int,
        q_len: int,
        cached_len: int,
        block_size: int,
        sliding_window: int | None,
        architecture: str,
        key_scale: float,
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            full_dequant_turboquant_cache,
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_prefill import (
            triton_turboquant_prefill_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device("cuda")
        preset = "turboquant_4bit_nc"
        seq_len = cached_len + q_len
        num_pages = math.ceil(seq_len / block_size)
        num_blocks = num_pages

        (
            cfg,
            pi,
            pit,
            s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            key_norm_lut,
        ) = _make_tq_state(d, preset, device, architecture=architecture)

        torch.manual_seed(321)
        key = (
            torch.randn(seq_len, hk, d, device=device, dtype=torch.float16) * key_scale
        )
        value = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16)
        query = torch.randn(q_len, hq, d, device=device, dtype=torch.float16)
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        physical_blocks = torch.roll(
            torch.arange(num_pages, device=device, dtype=torch.int32), shifts=1
        )
        logical_positions = torch.arange(seq_len, device=device, dtype=torch.int32)
        slot_mapping = (
            physical_blocks[logical_positions // block_size] * block_size
            + logical_positions % block_size
        )
        block_table = physical_blocks.view(1, num_pages)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        q_rot = (query.float() @ pit).contiguous()
        q_qjl = (query.float() @ st).contiguous()
        seq_lens = torch.arange(
            cached_len + 1, seq_len + 1, device=device, dtype=torch.int32
        )
        synth_bt = block_table.expand(q_len, -1)
        decode_internal = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=synth_bt,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            key_norm_lut=key_norm_lut,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            PiT=pit,
            max_num_kv_splits=4,
            sliding_window=sliding_window,
        )
        decode_preprojected = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=synth_bt,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            key_norm_lut=key_norm_lut,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            query_rot=q_rot,
            query_qjl=q_qjl,
            max_num_kv_splits=4,
            sliding_window=sliding_window,
        )
        assert torch.allclose(
            decode_internal.float(), decode_preprojected.float(), atol=0, rtol=0
        )

        native = triton_turboquant_prefill_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            cached_len=cached_len,
            PiT=pit,
            ST=st,
            key_norm_lut=key_norm_lut,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            sliding_window=sliding_window,
        )
        assert native is not None
        assert native.dtype == query.dtype
        diff = (native.float() - decode_preprojected.float()).abs()
        cos = F.cosine_similarity(
            native.float().reshape(1, -1),
            decode_preprojected.float().reshape(1, -1),
        ).item()
        assert diff.max().item() <= 1e-2
        assert cos >= 0.999

        k_ref, v_ref = full_dequant_turboquant_cache(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            Pi=pi,
            ST=st,
            centroids=centroids,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            S=s,
        )
        q_ref = query.float().transpose(0, 1).unsqueeze(0)
        kv_group_size = hq // hk
        k_ref = k_ref[0].repeat_interleave(kv_group_size, dim=0).unsqueeze(0)
        v_ref = v_ref[0].repeat_interleave(kv_group_size, dim=0).unsqueeze(0)
        key_positions = torch.arange(seq_len, device=device)
        query_positions = cached_len + torch.arange(q_len, device=device)
        attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if sliding_window is not None:
            attn_mask = attn_mask & (
                key_positions.unsqueeze(0)
                >= query_positions.unsqueeze(1) - sliding_window + 1
            )
        full_ref = F.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            attn_mask=attn_mask,
            scale=1.0 / math.sqrt(d),
        )[0].transpose(0, 1)
        full_diff = (native.float() - full_ref.float()).abs()
        full_cos = F.cosine_similarity(
            native.float().reshape(1, -1), full_ref.float().reshape(1, -1)
        ).item()
        assert full_diff.max().item() <= 5e-2
        assert full_cos >= 0.999

    @pytest.mark.parametrize(
        ("d", "hk", "hq", "architecture"),
        [
            (256, 2, 12, "Qwen3ForCausalLM"),
            (512, 2, 4, "Gemma4ForConditionalGeneration"),
        ],
    )
    def test_native_prefill_grouped_gqa_preserves_head_identity(
        self,
        d: int,
        hk: int,
        hq: int,
        architecture: str,
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_prefill import (
            triton_turboquant_prefill_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device("cuda")
        preset = "turboquant_4bit_nc"
        kv_group_size = hq // hk
        q_len = 11
        cached_len = 23
        seq_len = cached_len + q_len
        block_size = 16
        num_pages = math.ceil(seq_len / block_size)

        (
            cfg,
            pi,
            pit,
            _s,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            key_norm_lut,
        ) = _make_tq_state(d, preset, device, architecture=architecture)

        torch.manual_seed(654)
        key = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16) * 0.02
        value = torch.randn(seq_len, hk, d, device=device, dtype=torch.float16) * 0.02
        query = torch.randn(q_len, hq, d, device=device, dtype=torch.float16) * 0.02
        token_ramp = torch.linspace(-0.2, 0.2, q_len, device=device).to(torch.float16)
        for q_head in range(hq):
            kv_head = q_head // kv_group_size
            pos = q_head
            dim0 = (q_head * 17) % d
            dim1 = (dim0 + 5) % d
            key[pos, kv_head, :] = 0
            key[pos, kv_head, dim0] = 10.0
            key[pos, kv_head, dim1] = 2.0
            value[pos, kv_head, :] = 0
            value[pos, kv_head, 0] = float(q_head + 1)
            value[pos, kv_head, 1] = float(q_head + 1) * 0.25
            query[:, q_head, :] = 0
            query[:, q_head, dim0] = 10.0
            query[:, q_head, dim1] = token_ramp + float(q_head) * 0.01

        kv_cache = torch.zeros(
            num_pages,
            block_size,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        physical_blocks = torch.roll(
            torch.arange(num_pages, device=device, dtype=torch.int32), shifts=1
        )
        logical_positions = torch.arange(seq_len, device=device, dtype=torch.int32)
        slot_mapping = (
            physical_blocks[logical_positions // block_size] * block_size
            + logical_positions % block_size
        )
        block_table = physical_blocks.view(1, num_pages)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            pi,
            pit,
            st,
            centroids,
            midpoints,
            key_norm_log_min,
            key_norm_log_max,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )

        seq_lens = torch.arange(
            cached_len + 1, seq_len + 1, device=device, dtype=torch.int32
        )
        synth_bt = block_table.expand(q_len, -1)
        q_rot = (query.float() @ pit).contiguous()
        q_qjl = (query.float() @ st).contiguous()
        decode_ref = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=synth_bt,
            seq_lens=seq_lens,
            Pi=pi,
            ST=st,
            key_norm_lut=key_norm_lut,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_norm_log_min=key_norm_log_min,
            key_norm_log_max=key_norm_log_max,
            query_rot=q_rot,
            query_qjl=q_qjl,
            max_num_kv_splits=4,
        )
        native = triton_turboquant_prefill_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            cached_len=cached_len,
            PiT=pit,
            ST=st,
            key_norm_lut=key_norm_lut,
            centroids=centroids,
            scale=1.0 / math.sqrt(d),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
        )
        assert native is not None
        diff = (native.float() - decode_ref.float()).abs()
        assert diff.max().item() <= 1e-2

        native_by_head = native.float().transpose(0, 1).reshape(hq, -1)
        decode_by_head = decode_ref.float().transpose(0, 1).reshape(hq, -1)
        distances = torch.cdist(native_by_head, decode_by_head)
        nearest = distances.argmin(dim=1)
        expected = torch.arange(hq, device=device)
        assert torch.equal(nearest, expected), (
            "GQA head mapping changed; native head nearest decode head = "
            f"{nearest.detach().cpu().tolist()}"
        )
        offdiag = distances.masked_fill(
            torch.eye(hq, device=device, dtype=torch.bool), float("inf")
        )
        margin = offdiag.min(dim=1).values - distances.diag()
        assert torch.all(margin > 1e-2), (
            "Head signatures are not separable enough to catch swaps: "
            f"min_margin={margin.min().item()}"
        )

    def test_native_prefill_projection_guard_returns_none(self):
        from vllm.v1.attention.ops.triton_turboquant_prefill import (
            triton_turboquant_prefill_attention,
        )

        device = torch.device("cuda")
        d = 256
        hk = 1
        hq = 1
        q_len = 2
        cfg, _pi, pit, _s, st, centroids, *_unused = _make_tq_state(
            d, "turboquant_4bit_nc", device
        )
        query = torch.empty(q_len, hq, d, device=device, dtype=torch.float16)
        kv_cache = torch.empty(
            1,
            1,
            hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
        q_projection_bytes_without_q_float = q_len * hq * d * 2 * 2

        assert (
            triton_turboquant_prefill_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_len=q_len,
                cached_len=0,
                PiT=pit,
                ST=st,
                key_norm_lut=None,
                centroids=centroids,
                scale=1.0 / math.sqrt(d),
                mse_bits=cfg.key_mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.effective_value_quant_bits,
                max_q_projection_bytes=q_projection_bytes_without_q_float,
            )
            is None
        )

    def test_native_prefill_gqa_group_tile_selector(self):
        from vllm.v1.attention.ops.triton_turboquant_prefill import (
            _select_gqa_group_tile,
        )

        assert _select_gqa_group_tile(head_dim=128, kv_group_size=2) == 1
        assert _select_gqa_group_tile(head_dim=256, kv_group_size=1) == 1
        assert _select_gqa_group_tile(head_dim=256, kv_group_size=2) == 2
        assert _select_gqa_group_tile(head_dim=256, kv_group_size=4) == 2
        assert _select_gqa_group_tile(head_dim=256, kv_group_size=6) == 2
        assert _select_gqa_group_tile(head_dim=256, kv_group_size=3) == 1
        assert _select_gqa_group_tile(head_dim=512, kv_group_size=1) == 1
        assert _select_gqa_group_tile(head_dim=512, kv_group_size=2) == 2
        assert _select_gqa_group_tile(head_dim=512, kv_group_size=8) == 2
        assert _select_gqa_group_tile(head_dim=512, kv_group_size=3) == 1


class TestTurboQuantWorkspaceReservation:
    def test_attention_init_registers_paper_faithful_buffers(self, default_vllm_config):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.attention import Attention

        vllm_config = default_vllm_config
        vllm_config.cache_config.cache_dtype = "turboquant_4bit_nc"

        with set_current_vllm_config(vllm_config):
            attn = Attention(
                num_heads=8,
                head_size=512,
                scale=1.0,
                cache_config=vllm_config.cache_config,
                prefix="layers.0.self_attn",
            )

        buffer_names = {name for name, _ in attn.named_buffers()}
        assert "_tq_Pi" in buffer_names
        assert "_tq_PiT" in buffer_names
        assert "_tq_S" in buffer_names
        assert "_tq_ST" in buffer_names
        assert "_tq_centroids" in buffer_names
        assert "_tq_key_norm_log_min" in buffer_names
        assert "_tq_key_norm_log_max" in buffer_names
        assert "_tq_mid_o_buf" not in buffer_names
        assert "_tq_output_buf" not in buffer_names
        assert "_tq_lse_buf" not in buffer_names

    def test_runtime_split_policy_uses_smaller_eager_splits_for_long_context(
        self, default_vllm_config
    ):
        from vllm.config import set_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import (
            create_forward_context,
            override_forward_context,
        )
        from vllm.model_executor.layers.attention import Attention

        vllm_config = default_vllm_config
        vllm_config.cache_config.cache_dtype = "turboquant_4bit_nc"

        with set_current_vllm_config(vllm_config):
            attn = Attention(
                num_heads=8,
                head_size=256,
                scale=1.0,
                cache_config=vllm_config.cache_config,
                prefix="layers.0.self_attn",
            )

        impl = attn.impl
        eager_ctx = create_forward_context(
            attn_metadata={},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        with override_forward_context(eager_ctx):
            assert impl._select_num_kv_splits(8192) == 32
            assert impl._select_num_kv_splits(16384) == 16
            assert impl._select_num_kv_splits(32768) == 8
            assert impl._select_num_kv_splits(65536) == 8
            assert impl._select_long_prefill_chunk_size(65536) == 1024

        full_ctx = create_forward_context(
            attn_metadata={},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )
        with override_forward_context(full_ctx):
            assert impl._select_num_kv_splits(65536) == 32
            assert impl._select_long_prefill_chunk_size(65536) == 128

    def test_long_prefill_flash_is_gated_to_long_context(self, default_vllm_config):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.turboquant_attn import (
            _LONG_PREFILL_FA_MIN_SEQ_LEN,
        )

        vllm_config = default_vllm_config
        vllm_config.cache_config.cache_dtype = "turboquant_4bit_nc"

        with set_current_vllm_config(vllm_config):
            attn = Attention(
                num_heads=8,
                head_size=256,
                scale=1.0,
                cache_config=vllm_config.cache_config,
                prefix="layers.0.self_attn",
            )

        sentinel = cast(torch.Tensor, object())
        assert not attn.impl._can_use_long_prefill_flash(
            query=sentinel,
            kv_cache=sentinel,
            block_table=sentinel,
            seq_len=_LONG_PREFILL_FA_MIN_SEQ_LEN - 1,
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_workspace_reservation_uses_shared_max_buffer(
        self, default_vllm_config, workspace_init
    ):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.worker.workspace import current_workspace_manager

        del workspace_init

        vllm_config = default_vllm_config
        vllm_config.cache_config.cache_dtype = "turboquant_4bit_nc"
        vllm_config.scheduler_config.max_num_seqs = 16

        with set_current_vllm_config(vllm_config):
            Attention(
                num_heads=8,
                head_size=256,
                scale=1.0,
                cache_config=vllm_config.cache_config,
                prefix="layers.0.self_attn",
            )
            workspace = current_workspace_manager()._current_workspaces[0]
            assert workspace is not None
            workspace_bytes_256 = workspace.numel()

            Attention(
                num_heads=8,
                head_size=512,
                scale=1.0,
                cache_config=vllm_config.cache_config,
                prefix="layers.1.self_attn",
            )
            workspace = current_workspace_manager()._current_workspaces[0]
            assert workspace is not None
            workspace_bytes_512 = workspace.numel()

        batch_size = 128
        heads = 8
        splits = vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        raw_bytes_256 = (
            batch_size * heads * splits * (256 + 1) * 4
            + batch_size * heads * 256 * 4
            + batch_size * heads * 4
        )
        raw_bytes_512 = (
            batch_size * heads * splits * (512 + 1) * 4
            + batch_size * heads * 512 * 4
            + batch_size * heads * 4
        )

        assert workspace_bytes_256 >= raw_bytes_256
        assert workspace_bytes_256 < raw_bytes_256 + 1024
        assert workspace_bytes_512 >= raw_bytes_512
        assert workspace_bytes_512 < raw_bytes_512 + 1024
        assert workspace_bytes_512 > workspace_bytes_256
        assert workspace_bytes_512 < raw_bytes_256 + raw_bytes_512
