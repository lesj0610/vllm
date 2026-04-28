# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant quantizer utilities for paper-faithful rotations/sketches."""

import torch

_CPU = torch.device("cpu")


def _cpu_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def generate_random_orthogonal(d: int,
                               seed: int,
                               device: torch.device = _CPU) -> torch.Tensor:
    """Generate a deterministic Haar-like orthogonal matrix via QR."""
    gen = _cpu_generator(seed)
    g = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


def generate_qjl_projection(d: int,
                            seed: int,
                            device: torch.device = _CPU) -> torch.Tensor:
    """Generate the dense Gaussian projection used by QJL."""
    gen = _cpu_generator(seed)
    s = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    return s.to(device)
