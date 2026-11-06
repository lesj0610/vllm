# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only NVFP4 MoE for sm_80 to sm_90 (BF16 activations) behind
moe_backend="rs2_experimental".

The weights are converted once into records (rs2_records) that the _moe_C
rs2_moe_* kernels read. One global decode scale per expert and GEMM (the
gate's for gate and up), as every NVFP4 MoE backend here keeps.
"""

import functools
import math
from fractions import Fraction

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Row tiles: 8, 16 and 32 rows run the 1D kernel, 64 the 2D kernel after the
# multi-CTA prep.
ROW_TILES = (8, 16, 32, 64)
# The kernels take each expert's global scale times 2^7: the products carry
# e2m1 x s x 2^-7.
ALPHA_FOLD = 128.0
# (m0, e0, e1, sign) bit positions per output register in a 16-bit half
# (hi half = +16).
_POS = ((0, 1, 2, 9), (3, 4, 5, 12), (6, 7, 8, 15), (13, 10, 11, 14))
_RECORD_BYTES = 9216  # 128 x 128: 8192 B of codes, 1024 B of scales
_TAIL_RECORD_BYTES = 5120  # a K tail of at most 64 k: half the codes
_PREP_CHUNK = 2048  # routed entries per CTA of the multi-CTA prep (R 64)


def rs2_row_tile(M: int, E: int) -> int:
    """Row tile for M = tokens x top_k routed rows over E experts: 64 from
    28 rows per expert on average, 32 above 16, 16 from 8, else 8."""
    if M >= 28 * E:
        return 64
    if M > 16 * E:
        return 32
    return 16 if M >= 8 * E else 8


def _rup(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def rs2_padded(N: int, K: int, gated: bool) -> tuple[int, int, bool]:
    """Record geometry: Np record columns (a multiple of 128; gated: gate and
    up each padded to 64), S k steps of 128, and whether the last step holds
    at most 64 k (a 5120 B record)."""
    Np = 2 * _rup(N // 2, 64) if gated else _rup(N, 128)
    S = (K + 127) // 128
    return Np, S, K % 128 != 0 and K - 128 * (S - 1) <= 64


@functools.cache
def _u5_values() -> tuple[int, ...]:
    """E4M3 magnitude byte (0..127) -> U5 byte: f(5) m(3), f = floor(log2 s)
    + 22. 0 -> 0 (the block's codes are zeroed), 0x7F (NaN) -> 0xFF (BF16
    NaN). The kernels decode U5 u as the BF16 bits 0x7000 | u << 4, s x
    2^119."""
    t = [0] * 128
    for b in range(1, 127):
        e, m = b >> 3, b & 7
        if e == 0:
            v = Fraction(m, 8) * Fraction(2) ** -6
        else:
            v = (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
        k = math.floor(math.log2(v))
        while Fraction(2) ** k > v:
            k -= 1
        while Fraction(2) ** (k + 1) <= v:
            k += 1
        mm = (v / Fraction(2) ** k - 1) * 8
        assert mm.denominator == 1 and 0 <= k + 22 <= 30
        t[b] = ((k + 22) << 3) | int(mm)
        bits = 0x7000 | (t[b] << 4)  # BF16 bits
        ex, man = (bits >> 7) & 0xFF, bits & 0x7F
        dec = (1 + Fraction(man, 128)) * Fraction(2) ** (ex - 127)
        assert dec == v * Fraction(2) ** 119, b
    t[127] = 0xFF
    return tuple(t)


def _row_slots(N: int, Np: int, gated: bool, device: torch.device) -> torch.Tensor:
    """Record row of each weight row: gated up rows start at Np / 2."""
    r = torch.arange(N, device=device)
    return torch.where(r < N // 2, r, r - N // 2 + Np // 2) if gated else r


def _column_order(Np: int, gated: bool, device: torch.device) -> torch.Tensor:
    """Record row read by each record column: gated records take 8 gate
    rows, then the matching 8 up rows, per warp and chunk."""
    if not gated:
        return torch.arange(Np, device=device)
    half = Np // 2
    c = torch.arange(Np // 128, device=device).view(-1, 1, 1)
    w = torch.arange(8, device=device).view(1, -1, 1)
    j = torch.arange(16, device=device).view(1, 1, -1)
    return torch.where(
        j < 8, 64 * c + 8 * w + j, half + 64 * c + 8 * w + j - 8
    ).flatten()


def rs2_records(
    w: torch.Tensor, scale: torch.Tensor, gated: bool, chunk: int = 16
) -> torch.Tensor:
    """Packed NVFP4 [E, N, K / 2] (low nibble = even k) and E4M3 block scales
    [E, N, K / 16] -> records [E, Np / 128 chunks x S k steps x record], chunk
    experts at a time. A record is 9216 B: codes 8192 B ([half 2][warp 8]
    [lane 32][4 words], 3 + 1 bits per code), then scales 1024 B ([warp 8]
    [g 8][16 B] U5 bytes, byte 4 j2 + 2 hi + jl); a K tail of at most 64 k is
    5120 B (codes half 0, then the scales). Padding is zero codes."""
    E, N, Kh = w.shape
    K = 2 * Kh
    assert w.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
    assert scale.shape == (E, N, K // 16), (scale.shape, w.shape)
    dev = w.device
    Np, S, half = rs2_padded(N, K, gated)
    C, Kf = Np // 128, 128 * S
    perm = _column_order(Np, gated, dev)
    rows = _row_slots(N, Np, gated, dev)
    u5_table = torch.tensor(_u5_values(), dtype=torch.int32, device=dev)
    cb = (S - 1) * _RECORD_BYTES + (_TAIL_RECORD_BYTES if half else _RECORD_BYTES)
    out = torch.empty(E, C * cb, dtype=torch.uint8, device=dev)
    for e0 in range(0, E, chunk):
        q = w[e0 : e0 + chunk]
        n = q.shape[0]
        codes = torch.stack((q & 0xF, q >> 4), dim=-1).view(n, N, K).to(torch.int32)
        sb = scale[e0 : e0 + chunk].contiguous().view(torch.uint8).to(torch.int32)
        neg = (sb >> 7).repeat_interleave(16, dim=-1)
        mag = sb & 0x7F
        zero = (mag == 0).repeat_interleave(16, dim=-1)
        # The scale's sign moves into the codes; a zero scale zeroes the
        # magnitudes and keeps the signs.
        codes = codes ^ (neg << 3)
        codes = torch.where(zero, codes & 8, codes)
        cp = torch.zeros(n, Np, Kf, dtype=torch.int32, device=dev)
        cp[:, rows, :K] = codes
        mp = torch.zeros(n, Np, Kf // 16, dtype=torch.int32, device=dev)
        mp[:, rows, : K // 16] = mag
        del codes, sb, neg, mag, zero
        # (n, C, S, Jhi, W, g, t, Jlo, kh, hi, kl); value i = 4 kh + 2 hi + kl
        # -> register i >> 1, half i & 1
        X = cp[:, perm].view(n, C, 8, 2, 8, S, 2, 4, 2, 4, 2)
        X = X.permute(0, 1, 5, 6, 2, 4, 9, 7, 8, 3, 10).contiguous()
        X = X.view(n, C, S, 2048, 8)
        word = torch.zeros(n, C, S, 2048, dtype=torch.int32, device=dev)
        for i in range(8):
            c = X[..., i]
            sh = 16 * (i & 1)
            for bit, pos in enumerate(_POS[i >> 1]):
                word |= ((c >> bit) & 1) << (pos + sh)
        wbytes = torch.stack([(word >> (8 * k)) & 0xFF for k in range(4)], dim=-1)
        wbytes = wbytes.to(torch.uint8)
        del X, word, cp
        # (n, C, W, hi, g, S, Jhi, Jlo) -> (n, C, S, W, g, Jhi, hi, Jlo)
        u5 = u5_table[mp][:, perm].view(n, C, 8, 2, 8, S, 4, 2)
        sbytes = u5.permute(0, 1, 5, 2, 4, 6, 3, 7).contiguous().to(torch.uint8)
        r = torch.cat([wbytes.view(n, C, S, 8192), sbytes.view(n, C, S, 1024)], dim=-1)
        if half:
            tail = torch.cat([r[:, :, S - 1, :4096], r[:, :, S - 1, 8192:]], dim=-1)
            r = torch.cat(
                [r[:, :, : S - 1].reshape(n, C, (S - 1) * _RECORD_BYTES), tail],
                dim=-1,
            )
        out[e0 : e0 + n] = r.reshape(n, -1)
        del mp, wbytes, u5, sbytes, r
    return out


class Rs2ExperimentalExperts(mk.FusedMoEExpertsModular):
    """prep (deterministic row placement) -> GEMM1 (+ act) -> GEMM2 -> top-k
    weighted sum. Row tile from rs2_row_tile; every tile reads the same
    records."""

    # Routing buffers shared by every layer of one geometry (layers run in turn
    # on one stream) and by every row tile, sized on first use for the
    # scheduler's token budget. Grown only outside CUDA graph capture, and a
    # grown-out set is kept: captured graphs hold its addresses.
    _bufs: dict[tuple, dict] = {}
    _retired: list[dict] = []
    _seen_tiles: set[int] = set()

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_nvfp4_w4a16, "rs2_experimental: NVFP4 W4A16 only"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.gated = moe_config.is_act_and_mul
        self.I = moe_config.intermediate_size_per_partition
        self.H = moe_config.hidden_dim
        # The quant config is built after the weights are converted, so its
        # global scales are final here (both the ModelOpt and the
        # compressed-tensors paths); not every path calls
        # process_weights_after_loading on the experts.
        assert self.g1_alphas is not None and self.g2_alphas is not None
        self.a1 = (self.g1_alphas.float() * ALPHA_FOLD).contiguous()
        self.a2 = (self.g2_alphas.float() * ALPHA_FOLD).contiguous()
        N1 = 2 * self.I if self.gated else self.I
        self.Np1, _, _ = rs2_padded(N1, self.H, self.gated)
        self.Np2, _, _ = rs2_padded(self.H, self.I, False)
        self.act = 1 if self.gated else 2
        try:
            from vllm.config import get_current_vllm_config

            sched = get_current_vllm_config().scheduler_config
            self.max_tokens = int(sched.max_num_batched_tokens or 0)
        except Exception:
            self.max_tokens = 0

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.has_device_capability((8, 0))
            and not p.has_device_capability((10, 0))
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Weight-only, as Marlin: W4A4 checkpoints run with 16-bit activations.
        return weight_key == kNvfp4Static

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep and moe_parallel_config.dp_size == 1

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 16 == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert a1.dim() == 2 and topk_ids.dim() == 2
        return w1.size(0), a1.size(0), self.I, a1.size(-1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (1,), (1,), (M, K)

    @staticmethod
    def _caps(M: int, E: int) -> tuple[int, int]:
        """Routing blocks and row slots M rows can take under any row tile."""
        nb = (M + 7) // 8 + min(E, M)
        rows = max(((M + R - 1) // R + min(E, M)) * R for R in ROW_TILES)
        return nb, rows

    def _buffers(self, dev: torch.device, M: int, top_k: int, E: int) -> dict:
        key = (dev, E, self.H, self.I, top_k, max(self.Np1, self.Np2))
        b = self._bufs.get(key)
        if b is None or b["M"] < M:
            assert not torch.cuda.is_current_stream_capturing(), (
                "rs2_experimental: routing buffers must reach their size before "
                f"CUDA graph capture (M={M})"
            )
            Mc = max(M, self.max_tokens * top_k, b["M"] if b else 0)
            nb, rows = self._caps(Mc, E)
            if b is not None:
                self._retired.append(b)
            b = {
                "M": Mc,
                "meta": torch.zeros(1 + 2 * nb, dtype=torch.int32, device=dev),
                "rowmap": torch.full((rows,), -1, dtype=torch.int32, device=dev),
                "inter": torch.zeros(rows, self.I, dtype=torch.bfloat16, device=dev),
                "y2": torch.zeros(Mc, self.H, dtype=torch.bfloat16, device=dev),
                # multi-CTA prep: [ceil(M / chunk) + 1][E]
                "pw": torch.zeros(
                    ((Mc + _PREP_CHUNK - 1) // _PREP_CHUNK + 1) * E,
                    dtype=torch.int32,
                    device=dev,
                ),
            }
            self._bufs[key] = b
        return b

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert expert_map is None and not apply_router_weight_on_input
        assert hidden_states.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
        m, H = hidden_states.shape
        assert H == self.H, (H, self.H)
        top_k = topk_ids.size(1)
        E = w1.size(0)
        M = m * top_k
        R = rs2_row_tile(M, E)
        max_nb = (M + R - 1) // R + min(E, M)
        b = self._buffers(hidden_states.device, M, top_k, E)
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        x = hidden_states.contiguous()
        meta, rowmap, inter, y2 = b["meta"], b["rowmap"], b["inter"], b["y2"]
        if R not in self._seen_tiles:
            self._seen_tiles.add(R)
            logger.info(
                "rs2_experimental: row tile %d first used at M=%d E=%d", R, M, E
            )
        ops = torch.ops._moe_C
        ops.rs2_moe_prep(ids, meta, rowmap, b["pw"], E, R, max_nb)
        # GEMM1: hidden -> blocked intermediate (+ activation); GEMM2:
        # intermediate -> [tokens x top_k][H] (activation 0).
        ops.rs2_moe_gemm(
            w1,
            self.a1,
            x,
            meta,
            rowmap,
            inter,
            R,
            self.act,
            top_k,
            self.Np1,
            H,
            self.I,
            max_nb,
        )
        ops.rs2_moe_gemm(
            w2,
            self.a2,
            inter,
            meta,
            rowmap,
            y2,
            R,
            0,
            top_k,
            self.Np2,
            self.I,
            H,
            max_nb,
        )
        ops.rs2_moe_finalize(y2, tw, output, top_k)
