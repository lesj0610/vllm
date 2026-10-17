# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HC kernels at the edges of the range, not just on randn.

The CUDA kernels reach for `__expf`/`__fdividef`/`rsqrtf` instead of the
IEEE-correct forms, and each of those has its own behaviour at saturation and
on non-finite input. `test_hc_ops` only ever feeds them `randn`, which lands
inside +/-5 and so never asks. These do: the Triton fallback is the reference,
since the two have to be interchangeable, and both are then held against a
float64 reference where one exists.
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="HC kernels require CUDA and Triton",
)

HC = 4
HIDDEN = 2560  # not a power of two, so the general path runs
DIM = HC * HIDDEN
EPS = 1e-6

# `__expf(-v)` overflows around v = -88, and `__fdividef` flushes to zero once
# its denominator passes 2^126 (v around -87.3), so the saturation edge is
# sampled on both sides rather than only past it. The rest is the usual
# non-finite and signed-zero set plus the ends of bfloat16's range.
EXTREMES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1e-38,  # bfloat16 subnormal
    -1e-38,
    3.3895e38,  # bfloat16 max
    -3.3895e38,
    80.0,
    -80.0,
    87.0,
    -87.0,
    87.3,
    -87.3,
    88.0,
    -88.0,
    88.7,
    -88.7,
    89.0,
    -89.0,
    100.0,
    -100.0,
    1e30,
    -1e30,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def _tiled(rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
    """A tensor whose every element is an extreme, cycling so rows differ."""
    flat = torch.tensor(EXTREMES, dtype=torch.float32, device="cuda") * scale
    n = rows * cols
    out = flat.repeat((n + flat.numel() - 1) // flat.numel())[:n]
    return out.view(rows, cols).to(torch.bfloat16)


def _both(fn, *args, **kwargs):
    """Run an op down the CUDA path and down the Triton fallback."""
    real = hc_ops._has_cuda_hc
    try:
        hc_ops._has_cuda_hc = lambda: True
        cuda = fn(*args, **kwargs)
        hc_ops._has_cuda_hc = lambda: False
        triton = fn(*args, **kwargs)
    finally:
        hc_ops._has_cuda_hc = real
    return cuda, triton


def _assert_agree(cuda, triton, what: str) -> None:
    if isinstance(cuda, tuple):
        for i, (c, t) in enumerate(zip(cuda, triton)):
            _assert_agree(c, t, f"{what}[{i}]")
        return
    # bfloat16 carries 8 mantissa bits, so one ulp is ~0.8%; the fast-math
    # forms are good to a couple of fp32 ulp, far under that. atol covers the
    # flush-to-zero tail, where the true value is below bfloat16's smallest
    # subnormal times a few.
    torch.testing.assert_close(
        cuda,
        triton,
        rtol=0.016,
        atol=1e-36,
        equal_nan=True,
        msg=lambda m: f"{what}: {m}",
    )


def test_silu_extremes() -> None:
    # The op scales by 1/HC first, so the input is scaled up to put the
    # saturation edge back where the fast-math functions see it.
    x = _tiled(3, DIM, scale=float(HC))
    cuda, triton = _both(hc_ops.hc_silu, x, HC)
    _assert_agree(cuda, triton, "silu")

    f = x.double() / HC
    expected = f * torch.sigmoid(f)
    torch.testing.assert_close(
        cuda.double(), expected, rtol=0.016, atol=1e-36, equal_nan=True
    )


def test_gate_mix_extremes() -> None:
    x = _tiled(3, DIM)
    gate = _tiled(3, DIM).flip(-1)  # so a given column pairs unlike values
    cuda, triton = _both(hc_ops.hc_gate_mix, x, gate, HC)
    _assert_agree(cuda, triton, "gate_mix")

    expected = (
        torch.sigmoid(gate.double().unflatten(-1, (HC, HIDDEN)))
        * x.double().unflatten(-1, (HC, HIDDEN))
    ).mean(-2)
    torch.testing.assert_close(
        cuda.double(), expected, rtol=0.016, atol=1e-36, equal_nan=True
    )


def test_combine_extremes() -> None:
    residual = _tiled(3, DIM)
    block = _tiled(3, HIDDEN)
    # The injection is divided by HC before the sigmoid, so it too is scaled up.
    injection = _tiled(3, HC, scale=float(HC))
    cuda, triton = _both(hc_ops.hc_combine, residual, block, injection, HC)
    _assert_agree(cuda, triton, "combine")


def test_combine_norm_extremes() -> None:
    residual = _tiled(3, DIM)
    block = _tiled(3, HIDDEN)
    injection = _tiled(3, HC, scale=float(HC))
    weight = _tiled(1, DIM).squeeze(0)
    cuda, triton = _both(
        hc_ops.hc_combine_norm, residual, block, injection, weight, EPS, HC
    )
    _assert_agree(cuda, triton, "combine_norm")


@pytest.mark.parametrize(
    "fill",
    [
        pytest.param(0.0, id="all-zero"),
        pytest.param(3.3895e38, id="squares-overflow-fp32"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_rmsnorm_degenerate_rows(fill: float) -> None:
    """`rsqrtf` of a sum that is zero, overflowed, or not a number."""
    x = torch.full((2, DIM), fill, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(DIM, dtype=torch.bfloat16, device="cuda")
    cuda, triton = _both(hc_ops.grouped_gemma_rmsnorm, x, weight, EPS, HC)
    _assert_agree(cuda, triton, f"rmsnorm({fill})")


def test_rmsnorm_extreme_weights() -> None:
    """The weight is used as `1 + w`, so the extremes land on that sum."""
    x = torch.randn(3, DIM, dtype=torch.bfloat16, device="cuda")
    weight = _tiled(1, DIM).squeeze(0)
    cuda, triton = _both(hc_ops.grouped_gemma_rmsnorm, x, weight, EPS, HC)
    _assert_agree(cuda, triton, "rmsnorm-weights")
