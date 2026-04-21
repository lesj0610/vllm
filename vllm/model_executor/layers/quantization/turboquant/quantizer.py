# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant quantizer utilities.

Triton kernels handle all quantization, packing, and dequantization on GPU.
"""

import torch

_CPU = torch.device("cpu")


def generate_wht_signs(d: int, seed: int, device: torch.device = _CPU) -> torch.Tensor:
    """Generate deterministic random +/-1 signs for WHT rotation."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=gen, device="cpu")
    signs = bits.float() * 2 - 1
    return signs.to(device)
