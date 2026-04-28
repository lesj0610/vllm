# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper-faithful TurboQuant configuration.

TurboQuant proper is the `_nc` preset family only. Each key uses a total
bit-width of ``key_quant_bits`` split into:
  * ``key_mse_bits = key_quant_bits - 1`` for TurboQuant_mse indices
  * ``1`` residual QJL bit per coordinate for TurboQuant_prod

Values remain uniformly quantized as a separate packed payload.
"""

import math
from dataclasses import dataclass

TQ_PRESETS: dict[str, dict] = {
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
    },
}

_RESIDUAL_NORM_QUANT_MAX_BY_LAYOUT: dict[tuple[int, int], float] = {
    # key_mse_bits, value_quant_bits -> observed-max-safe residual norm cap
    (3, 4): 0.40,  # turboquant_4bit_nc
    (2, 4): 0.55,  # turboquant_k3v4_nc
    (2, 3): 0.50,  # turboquant_3bit_nc
}

_KEY_NORM_LOG_RANGE_BY_ARCH: dict[str, tuple[float, float]] = {
    "Qwen3ForCausalLM": (-2.0, 9.0),
    "Qwen3_5ForConditionalGeneration": (-2.0, 9.0),
    "Qwen3_5MoeForConditionalGeneration": (-2.0, 9.0),
    "Gemma4ForConditionalGeneration": (-3.0, 3.0),
}

_DEFAULT_KEY_NORM_LOG_RANGE = (-2.0, 10.0)


@dataclass
class TurboQuantConfig:
    """Configuration for paper-faithful TurboQuant KV-cache quantization.

    The key path follows TurboQuant_prod:
      1. TurboQuant_mse with ``key_quant_bits - 1`` bits per coordinate.
      2. QJL sign sketch on the residual with 1 bit per coordinate.
      3. Store original key norm as log-uint8 and residual norm as uint8.

    The value path remains a packed uniform quantizer.
    """

    head_dim: int = 128
    key_quant_bits: int = 4  # total key bits: MSE bits + 1 QJL bit
    value_quant_bits: int = 4
    seed: int = 42

    @property
    def key_mse_bits(self) -> int:
        """TurboQuant_mse bit-width used before the residual QJL stage."""
        return self.key_quant_bits - 1

    @property
    def qjl_bits(self) -> int:
        return 1

    @property
    def centroid_bits(self) -> int:
        return self.key_mse_bits

    @property
    def n_centroids(self) -> int:
        return 2**self.key_mse_bits

    @property
    def key_mse_packed_size(self) -> int:
        return math.ceil(self.head_dim * self.key_mse_bits / 8)

    @property
    def key_qjl_packed_size(self) -> int:
        return math.ceil(self.head_dim * self.qjl_bits / 8)

    @property
    def key_norm_packed_size(self) -> int:
        return 1

    @property
    def residual_norm_packed_size(self) -> int:
        return 1

    @property
    def residual_norm_quant_max(self) -> float:
        return self.get_residual_norm_quant_max(
            self.key_mse_bits,
            self.value_quant_bits,
        )

    @staticmethod
    def get_key_norm_log_range_for_arch(
        architecture: str | None,
    ) -> tuple[float, float]:
        if not architecture:
            return _DEFAULT_KEY_NORM_LOG_RANGE
        if architecture in _KEY_NORM_LOG_RANGE_BY_ARCH:
            return _KEY_NORM_LOG_RANGE_BY_ARCH[architecture]
        if architecture.startswith(("Qwen3", "Qwen3_5")):
            return (-2.0, 9.0)
        if architecture.startswith("Gemma4"):
            return (-3.0, 3.0)
        return _DEFAULT_KEY_NORM_LOG_RANGE

    @property
    def key_norm_offset(self) -> int:
        return self.key_mse_packed_size + self.key_qjl_packed_size

    @property
    def residual_norm_offset(self) -> int:
        return self.key_norm_offset + self.key_norm_packed_size

    @property
    def key_packed_size(self) -> int:
        return (
            self.key_mse_packed_size
            + self.key_qjl_packed_size
            + self.key_norm_packed_size
            + self.residual_norm_packed_size
        )

    @property
    def effective_value_quant_bits(self) -> int:
        return self.value_quant_bits

    @property
    def value_packed_size(self) -> int:
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # scale fp16 + zero fp16

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)

    @staticmethod
    def get_residual_norm_quant_max(mse_bits: int, value_quant_bits: int) -> float:
        key = (mse_bits, value_quant_bits)
        if key not in _RESIDUAL_NORM_QUANT_MAX_BY_LAYOUT:
            raise ValueError(
                "Unknown TurboQuant residual-norm quantization layout: "
                f"mse_bits={mse_bits}, value_quant_bits={value_quant_bits}"
            )
        return _RESIDUAL_NORM_QUANT_MAX_BY_LAYOUT[key]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "TurboQuantConfig":
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(
                f"Unknown TurboQuant cache dtype: {cache_dtype!r}. "
                f"Valid presets: {valid}"
            )
        preset = TQ_PRESETS[cache_dtype]
        return TurboQuantConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
        )
