# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the weight-only NVFP4 MoE backend moe_backend="rs2_experimental"."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rs2_experimental_moe import (
    Rs2ExperimentalExperts,
    rs2_row_tile,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import set_random_seed

requires_rs2 = pytest.mark.skipif(
    not Rs2ExperimentalExperts._supports_current_device(),
    reason="rs2_experimental needs compute capability 8.0 to 9.x",
)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize(
    "capability,supported",
    [
        ((7, 5), False),
        ((8, 0), True),
        ((8, 6), True),
        ((8, 7), True),
        ((8, 9), True),
        ((9, 0), True),
        ((10, 0), False),
        ((12, 0), False),
    ],
)
def test_rs2_device_range(capability: tuple[int, int], supported: bool):
    with patch.object(
        type(current_platform),
        "get_device_capability",
        return_value=DeviceCapability(*capability),
    ):
        assert Rs2ExperimentalExperts._supports_current_device() == supported


@requires_rs2
def test_rs2_kernels_built():
    # A build for a supported device must carry the kernels.
    current_platform.import_kernels()
    for op in ("rs2_moe_prep", "rs2_moe_gemm", "rs2_moe_finalize"):
        overload = getattr(torch.ops._moe_C, op).default
        assert overload.has_kernel_for_dispatch_key("CUDA"), op


def _dequantize(w: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor):
    return torch.stack(
        [
            dequantize_nvfp4_to_dtype(
                w[e],
                scale[e],
                1.0 / scale_2[e],
                torch.float32,
                w.device,
                is_sf_linear_layout=True,
            )
            for e in range(w.size(0))
        ]
    )


def _nvfp4_weights(E: int, N: int, K: int):
    """Every E2M1 code (-0 included), E4M3 block scales from 0 to 4 (zero
    blocks included) in the linear layout of a checkpoint, and a global scale
    per expert that brings the GEMM outputs near unit scale."""
    w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(0, 0x49, (E, N, K // 16), dtype=torch.uint8, device="cuda")
    scale = scale.view(torch.float8_e4m3fn)
    ones = torch.ones(E, device="cuda")
    rms = _dequantize(w, scale, ones).square().mean(dim=(1, 2)).sqrt()
    scale_2 = (0.5 + torch.rand(E, device="cuda")) / (K**0.5 * rms)
    return (w, scale, scale_2), _dequantize(w, scale, scale_2)


def _routing(m: int, E: int, top_k: int):
    """Expert e < E - 1 drawn with weight 2^-e and the last one never: the
    first experts fill several row tiles, the others leave partial ones."""
    p = 0.5 ** torch.arange(E, dtype=torch.float32)
    p[-1] = 0
    topk_ids = torch.multinomial(p.expand(m, E), top_k)
    topk_weights = torch.rand(m, top_k) + 0.1
    return topk_weights.cuda(), topk_ids.to(torch.int32).cuda()


def _reference(x, w1, w2, topk_weights, topk_ids, activation):
    """fp32 GEMMs on the dequantized weights, rounded to bf16 where the
    kernels store: the intermediate and each routed row."""
    m, top_k = topk_ids.shape
    rows = x.float().repeat_interleave(top_k, dim=0)
    ids = topk_ids.flatten()
    y = torch.zeros(m * top_k, w2.size(1), device=x.device)
    for e in ids.unique().tolist():
        sel = ids == e
        h = rows[sel] @ w1[e].t()
        if activation == MoEActivation.SILU:
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.relu(h).square()
        y[sel] = h.bfloat16().float() @ w2[e].t()
    y = y.bfloat16().float().view(m, top_k, -1)
    return (y * topk_weights[..., None]).sum(dim=1).bfloat16()


@requires_rs2
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
# (320, 208): K tails of 64 and 80 and padded record columns in both GEMMs.
@pytest.mark.parametrize("hidden,inter", [(256, 128), (320, 208)])
@torch.inference_mode()
def test_rs2_moe(
    hidden: int,
    inter: int,
    activation: MoEActivation,
    default_vllm_config,
    workspace_init,
):
    set_random_seed(0)
    E, top_k = 8, 2
    q1, w1 = _nvfp4_weights(E, 2 * inter if activation.is_gated else inter, hidden)
    q2, w2 = _nvfp4_weights(E, hidden, inter)

    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=inter,
        activation=activation,
    )
    moe_config.moe_backend = "rs2_experimental"
    backend, experts_cls = select_nvfp4_moe_backend(
        moe_config, weight_key=kNvfp4Static, activation_key=None
    )
    assert backend == NvFp4MoeBackend.RS2_EXPERIMENTAL
    assert experts_cls is Rs2ExperimentalExperts
    w13, w13_scale, w13_scale_2, a13_scale, w2_rec, w2_scale, w2_scale_2, a2_scale = (
        convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=backend,
            layer=SimpleNamespace(),
            w13=q1[0],
            w13_scale=q1[1],
            w13_scale_2=q1[2],
            a13_scale=None,
            w2=q2[0],
            w2_scale=q2[1],
            w2_scale_2=q2[2],
            a2_scale=None,
            is_act_and_mul=activation.is_gated,
            use_a16=True,
        )
    )
    quant_config = make_nvfp4_moe_quant_config(
        backend=backend,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
        use_a16=True,
    )
    kernel = make_nvfp4_moe_kernel(quant_config, moe_config, experts_cls, backend)

    calls = []
    # m = 1 leaves six experts empty.
    for m, row_tile in ((1, 8), (13, 8), (45, 16), (83, 32), (300, 64)):
        assert rs2_row_tile(m * top_k, E) == row_tile
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = _routing(m, E, top_k)
        args = dict(
            hidden_states=x,
            w1=w13,
            w2=w2_rec,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=E,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        out = kernel.apply(**args)
        ref = _reference(x, w1, w2, topk_weights, topk_ids, activation)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        calls.append((args, out))
    # Each row tile again, on buffers the other tiles have used: same bits.
    for args, out in calls:
        assert torch.equal(kernel.apply(**args), out)
