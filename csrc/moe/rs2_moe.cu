// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Weight-only NVFP4 MoE: BF16 activations, NVFP4 weights converted
// once at load time into records (rs2_experimental_moe.py). A record covers one
// expert, 128 output columns and 128 k: 8192 B of E2M1 codes in MMA fragment
// order, then 1024 B of block scales, each the E4M3 scale's magnitude as a
// 5-bit exponent and a 3-bit mantissa (its sign moves into the codes). A K tail
// of at most 64 k takes a 5120 B record. The per-expert global scale (x 2^7)
// scales the fp32 accumulator.
//
// A forward is prep (routing -> blocks of R rows), GEMM1 (+ activation), GEMM2
// and finalize (the top-k weighted sum). prep places entry f = token * top_k +
// slot at the rank of f among its expert's entries, so the placement, and with
// it every fp32 sum order, repeats bit for bit between calls. R 8, 16 and 32
// use 8 warps, each covering 16 columns of all R rows; R 64 uses 4 warps,
// each covering 32 columns, with swizzled activation rows, after the
// multi-CTA prep. R 96 and the two-record R 64 variants (rs2_moe_gemm_w64s,
// rs2_moe_gemm_g2w64x) cover two 128-column records per CTA: 8 warps of 32
// columns and 4 warps of 64 columns. rs2_moe_gemm_g2w64v is rs2_moe_gemm_g2w64x
// with one 16 B scale load per k128. rs2_moe_gemm_x96 and rs2_moe_gemm_q12 are
// rs2_moe_gemm where R 96 runs at 2 stages: the R 64 two-record epilogue, a
// hoisted activation address and shift-only dequant; q12 also splits the rows
// in 12 parts instead of 6. One GEMM template covers every warp layout; the
// host launches one whole output tile per CTA.
//
// The kernels need cp.async, ldmatrix and the bf16 mma.sync, so they build
// for sm_80 and sm_90; sm_86, sm_87 and sm_89 run the sm_80 code. Every config
// but R 96 at 3 stages fits the 99 KB per-block shared memory limit of sm_86
// and sm_89 (at most 89088 B); R 96 takes 3 stages (133632 B) only where a
// block can have that much and 2 stages otherwise.
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cutlass/fast_math.h"

namespace vllm::rs2_moe {

constexpr int kTileN = 128, kTileK = 128;
constexpr int kWBytes = kTileN * kTileK / 2;   // 8192
constexpr int kSBytes = kTileN * kTileK / 16;  // 1024
constexpr int kRec = kWBytes + kSBytes;        // 9216
constexpr int kHalfW = kWBytes / 2;  // 4096: k 0..63 of a record, all warps
constexpr int kTailRec = kHalfW + kSBytes;  // 5120: a K tail of at most 64
constexpr int kActPitch = 272;  // 256 B row + 16 B against ldmatrix conflicts
// Smallest per-block shared memory limit from sm_80 to sm_90 (sm_86, sm_89).
constexpr int kSmemLimit = 99 * 1024;

template <int R, int NC = 1>
struct Cfg {
  static constexpr int kABytes = R * kActPitch;
  static constexpr int kStageBytes = NC * kRec + kABytes;
};

__device__ __forceinline__ void cp16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp16z(uint32_t s, const void* g, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(g),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void lds64(uint32_t* d, uint32_t a) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
__device__ __forceinline__ void lds32(uint32_t* d, uint32_t a) {
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(d[0]) : "r"(a));
}
// Volatile shared loads: ptxas does not merge them into wider loads.
__device__ __forceinline__ void lds32v(uint32_t* d, uint32_t a) {
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(d[0]) : "r"(a));
}
// gt / d for d = 1 + __expf(-gt), bit-identical to the IEEE division (checked
// over all 2^32 fp32 gt on sm_80): the division's own fast path (MUFU.RCP + 4
// FFMA) with the cases its FCHK sends to the slow path written out as selects,
// no branch.
__device__ __forceinline__ float rcp_approx(float d) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
  return r;
}
__device__ __forceinline__ float div_fast(float n, float d) {
  float r = rcp_approx(d);
  const float t = fmaf(-d, r, 1.f);
  r = fmaf(r, t, r);
  const float q0 = fmaf(n, r, 0.f);
  const float e = fmaf(-d, q0, n);
  return fmaf(r, e, q0);
}
__device__ __forceinline__ float silu_div(float n, float d) {
  const float inf = __int_as_float(0x7f800000);
  const float s = d >= 0x1p126f ? 0x1p-24f : 1.f;
  float q = div_fast(n, d * s) * s;
  q = d == inf ? n * 0.f : q;
  q = (n == 0.f || n == inf) ? n : q;
  return q;
}
__device__ __forceinline__ void lds64v(uint32_t* d, uint32_t a) {
  asm volatile("ld.volatile.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
__device__ __forceinline__ void lds128(uint32_t* d, uint32_t a) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* d, uint32_t a) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* d, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t const* a,
                                         uint32_t const* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// a x {lo(s2), lo(s2)} and a x {hi(s2), hi(s2)}, exact; the broadcast is built
// inside the asm so ptxas can fold it into HFMA2's operand swizzle.
__device__ __forceinline__ uint32_t mul_lo(uint32_t a, uint32_t s2) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\n.reg .b32 b;\nmov.b32 {l, h}, %2;\nmov.b32 b, {l, "
      "l};\n"
      "fma.rn.bf16x2 %0, %1, b, %3;\n}\n"
      : "=r"(r)
      : "r"(a), "r"(s2), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ uint32_t mul_hi(uint32_t a, uint32_t s2) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\n.reg .b32 b;\nmov.b32 {l, h}, %2;\nmov.b32 b, {h, "
      "h};\n"
      "fma.rn.bf16x2 %0, %1, b, %3;\n}\n"
      : "=r"(r)
      : "r"(a), "r"(s2), "r"(0x80008000u));
  return r;
}

// 3+1 layout -> four bf16x2 registers of e2m1 x 2^-126.
template <bool SHF = false>
__device__ __forceinline__ void cvt31(uint32_t w, uint32_t* d) {
  d[0] = (w << 6) & 0x81C081C0u;
  d[1] = (w << 3) & 0x81C081C0u;
  d[2] = w & 0x81C081C0u;
  // m = w & 0x2C002C00u has bits only in {10, 11, 13, 26, 27, 29} and
  // 0x22000000 = 2^29 + 2^25, so __umulhi(m, 0x22000000) = (m >> 3) +
  // (m >> 7), two bit-disjoint shifts. SHF computes the same bits with shifts
  // instead of the IMAD.HI; sm_90 keeps the multiply.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t x = __umulhi(w & 0x2C002C00u, 0x22000000u);
#else
  uint32_t x;
  if constexpr (SHF) {
    const uint32_t m = w & 0x2C002C00u;
    x = (m >> 3) | (m >> 7);
  } else {
    x = __umulhi(w & 0x2C002C00u, 0x22000000u);
  }
#endif
  d[3] = (x & 0x01C001C0u) | ((w & 0x40004000u) << 1);
}
// Two U5 scale pairs of one word: even j (bytes 0, 2) and odd j (bytes 1, 3),
// as s x 2^119.
__device__ __forceinline__ uint32_t u5_even(uint32_t q) {
  return ((q << 4) & 0x0FF00FF0u) | 0x70007000u;
}
__device__ __forceinline__ uint32_t u5_odd(uint32_t q) {
  return ((q >> 4) & 0x0FF00FF0u) | 0x70007000u;
}

struct GemmParams {
  const uint8_t* w;    // prepared records [E][chunks][S]: 9216 B each, the last
                       // 5120 B for kr <= 64
  const float* alpha;  // [E] per-expert global dequant scale x 2^7
  const __nv_bfloat16*
      act;  // GEMM1: hidden [tokens][K]; GEMM2: blocked intermediate [nb][R][K]
  const int* nblocks;       // device scalar
  const int* block_expert;  // [nb]
  const int* block_rows;    // [nb]
  const int* rowmap;        // [nb][R]: token * top_k + slot, or -1
  __nv_bfloat16* out;       // GEMM1: blocked intermediate [nb][R][Nout]; GEMM2:
                            // [tokens*top_k][Nout]
  float* ws;  // unused: one whole tile per CTA, no split-tile partials
  int* cnt;   // unused, as ws
  int top_k, N,
      K;     // N: record columns, a multiple of 128 (gated: 2 x I padded to 64)
  int Nout;  // output columns (gated: I)
  int units_per_cta;            // unused, as ws
  cutlass::FastDivmod topk_dm;  // f / top_k via host-built magic (G1
                                // only); appended to preserve layout
};

// bf16 pair (lo in the low 16 bits) and an 8 x 8 b16 transpose across the warp
// (movmatrix, sm_75+): lane l reads row l / 4, columns 2 (l % 4) and + 1 of the
// piece and gets the same positions of the transposed piece.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t movm_trans(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
               : "=r"(y)
               : "r"(x));
  return y;
}

// ACT: 0 none (GEMM2), 1 silu(gate) * up (GEMM1, gated), 2 relu(x)^2 (GEMM1,
// not gated) KT: K % 128 != 0 (a tail k-step per chunk). G1E, Q12F: the R 96
// 2-stage paths of rs2_moe_gemm_x96 and rs2_moe_gemm_q12. VS: the scale load of
// rs2_moe_gemm_g2w64v.
template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1,
          bool G1E = false, bool Q12F = false, bool VS = false>
__device__ __forceinline__ void gemm_body(GemmParams const& p) {
  using C = Cfg<R, NC>;
  constexpr int NT = NW * 32;
  constexpr int SC = 8 * NC / WC, WT = NW / WC, TW = R / WT, G = TW / 8;
  constexpr int RP =
      NT / 16;  // activation rows per pass: 16 threads x 16 B per 256 B row
  constexpr int WP = kWBytes / (NT * 16);  // weight cp16 per thread and record
  static_assert((WC == 4 || WC == 8) && NW % WC == 0 &&
                    (R % RP == 0 || RP % R == 0) && TW % 8 == 0,
                "warp tile");
  static_assert(
      kWBytes % (NT * 16) == 0 && kHalfW % (NT * 16) == 0 && kSBytes / 16 <= NT,
      "issue");
  // At R 64 with 4 warps the activation rows sit at a 256 B pitch, the 16 B
  // chunk q of row r at chunk q ^ (r & 7) (a permutation inside the chunk's 128
  // B half), in place of the 272 B padded pitch. The cp.async writes and the
  // ldmatrix reads take the same mapping; the stage keeps its R x 272 B.
  constexpr bool SWZ = R >= 64;
  constexpr int AP = SWZ ? 256 : kActPitch;
  static_assert(!SWZ || (RP % 8 == 0 && TW % 8 == 0),
                "swizzle: r & 7 is fixed per thread and per lane");
  constexpr uint32_t kStage = C::kStageBytes;
  constexpr uint32_t kRing = STAGES * kStage;
  extern __shared__ __align__(128) uint8_t smem[];
  const int nb = *p.nblocks;
  const int K = p.K;
  const int record_chunks = p.N / kTileN;
  const int chunks = (record_chunks + NC - 1) / NC;
  const int S = KT ? (K + kTileK - 1) / kTileK : K / kTileK;
  const int kr = K - (S - 1) * kTileK;
  const bool half = KT && kr <= kTileK / 2;
  const long cbytes =
      static_cast<long>(S - 1) * kRec + (half ? kTailRec : kRec);
  if (blockIdx.x >= nb * chunks) return;
  const int upb = record_chunks * S;
  const int u0 = 0, u1 = S;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2,
            tq = lane & 3;
  const int wc = warp % WC, wt = warp / WC;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  const int ib = blockIdx.x / chunks;
  int iw = (blockIdx.x % chunks) * NC * S;
  int iks = 0;
  uint32_t islot = 0;
  const uint8_t* gws[NC];
  bool valid_record[NC];
  constexpr int RJ = R > RP ? R / RP : 1;
  // Linear GEMM2 (ACT 0) at R 64, 2 stages: the RJ activation-row source
  // pointers differ only by a constant RP * K, so keep one and rebuild the
  // rest from it to free registers. Other ACT/R/STAGES keep gact[RJ].
  constexpr bool kLinSingle = ACT == 0 && R == 64 && STAGES == 2;
  constexpr int GJ = kLinSingle ? 1 : RJ;
  const __nv_bfloat16* gact[GJ];
  bool avalid[RJ];
  // SC 4: the rows as indices (-1: absent), the source formed at each issue
  constexpr bool kRowIndex = SC == 4 && ACT != 0;
  int arow[kRowIndex ? RJ : 1];
#pragma unroll
  for (int j = 0; j < RJ; ++j) avalid[j] = true;
  const int ar = tid >> 4, aq = tid & 15;
  const bool has_act = R >= RP || tid < R * 16;
  const bool has_s = tid < kSBytes / 16;
  const uint32_t sw_i = sbase + tid * 16;
  // rows ar + RP j share ar & 7 (RP % 8 == 0), so the swizzled chunk is one
  // constant per thread
  const uint32_t sa_i = sbase + NC * kRec + (has_act ? ar : 0) * AP +
                        (SWZ ? (aq ^ (ar & 7)) : aq) * 16;
  auto load_block = [&]() {
    const long e = p.block_expert[ib];
    const int brows = ACT == 0 ? p.block_rows[ib] : R;
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) {
      const int c = iw / S + rec;
      valid_record[rec] = NC == 1 || c < record_chunks;
      if constexpr (KT) {
        gws[rec] = p.w + (e * record_chunks + c) * cbytes + tid * 16;
      } else {
        gws[rec] = p.w + (e * upb + iw + rec * S) * kRec + tid * 16;
      }
    }
#pragma unroll
    for (int j = 0; j < RJ; ++j) {
      const int r = ar + RP * j;
      long row = 0;
      if (has_act) {
        if constexpr (ACT != 0) {
          const int f = p.rowmap[ib * R + r];
          avalid[j] = f >= 0;
          row = avalid[j] ? p.topk_dm.div(f) : 0;
        } else {
          row = static_cast<long>(ib) * R + r;
          avalid[j] = r < brows;  // a padding row of the intermediate:
                                  // zero-filled, not read
        }
      }
      if constexpr (kLinSingle) {
        if (j == 0) gact[0] = p.act + row * K + aq * 8 + iks * kTileK;
      } else {
        gact[j] = p.act + row * K + aq * 8 + iks * kTileK;
      }
      if constexpr (kRowIndex) arow[j] = avalid[j] ? static_cast<int>(row) : -1;
    }
  };
  load_block();
  auto issue = [&]() {
    bool tl = false, hf = false;
    if constexpr (KT) {
      tl = iks == S - 1;
      hf = tl && half;
    }
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) {
      const uint32_t dst = sw_i + islot + rec * kRec;
      const uint8_t* src = valid_record[rec] ? gws[rec] : p.w;
#pragma unroll
      for (int q = 0; q < WP; ++q) {
        const int o = q * NT * 16;
        if (!hf || o < kHalfW) cp16z(dst + o, src + o, valid_record[rec]);
      }
      if (has_s)
        cp16z(dst + kWBytes, src + (hf ? kHalfW : kWBytes), valid_record[rec]);
    }
#pragma unroll
    for (int j = 0; j < RJ; ++j) {
      if (!has_act) continue;
      const uint32_t sa = sa_i + j * RP * AP + islot;
      if constexpr (kRowIndex) {
        const bool av = arow[j] >= 0;
        const __nv_bfloat16* ga = p.act +
                                  static_cast<long>(av ? arow[j] : 0) * K +
                                  aq * 8 + iks * kTileK;
        if constexpr (KT) {
          const bool in = !tl || aq * 8 < kr;
          if (!hf || aq < 8) cp16z(sa, in ? ga : p.act, av && in);
        } else {
          cp16z(sa, ga, av);
        }
      } else if constexpr (kLinSingle) {
        const __nv_bfloat16* gj;
        if constexpr (kLinSingle)
          gj = gact[0] + static_cast<long>(j) * RP * K;
        else
          gj = gact[j];
        if constexpr (KT) {
          const bool in = !tl || aq * 8 < kr;
          if (!hf || aq < 8) cp16z(sa, in ? gj : p.act, avalid[j] && in);
        } else {
          cp16z(sa, gj, avalid[j]);
        }
      } else if constexpr (KT) {
        const bool in = !tl || aq * 8 < kr;
        if (!hf || aq < 8) cp16z(sa, in ? gact[j] : p.act, avalid[j] && in);
      } else {
        cp16z(sa, gact[j], avalid[j]);
      }
    }
    islot = islot + kStage == kRing ? 0u : islot + kStage;
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) gws[rec] += hf ? kTailRec : kRec;
    if constexpr (kLinSingle) {
      gact[0] += kTileK;
    } else {
#pragma unroll
      for (int j = 0; j < RJ; ++j) gact[j] += kTileK;
    }
    ++iks;
  };

  // Writes one tile from per-lane fp32 values v[SC][G][4] (summed over k);
  // gemm's mapping with record warp slot ws = wc SC + s in place of the warp
  // and rows offset by wt TW.
  auto write_tile = [&](int b, int c, float (*v)[G][4]) {
    const float a = p.alpha[p.block_expert[b]];
    const int Nout = p.Nout;
    if constexpr (ACT == 0 &&
                  (SC == 4 || (G1E && R == 96 && STAGES == 2 && !KT))) {
      // v[s][gi][2 hh + h] is column ws 16 + 8 hh + g8, row gi 8 + 2 tq + h:
      // per (s, gi, hh) an 8 x 8 piece, columns x rows, one bf16 pair (rows 2
      // tq, 2 tq + 1) per lane. Transposed, lane (g8, tq) holds row gi 8 + g8,
      // columns 8 hh + 2 tq and + 1: one 4 B store (Nout even, out 4 B
      // aligned).
      if ((Nout & 1) == 0 && (reinterpret_cast<uintptr_t>(p.out) & 3) == 0) {
        const int rows = p.block_rows[b];
        int drow[G];
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
          drow[gi] = p.rowmap[b * R + wt * TW + gi * 8 + g8];
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
          if (wt * TW + gi * 8 + g8 >= rows) drow[gi] = -1;
        const int n0 = c * NC * kTileN + wc * SC * 16 + 2 * tq;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
          uint32_t t[SC][2];
#pragma unroll
          for (int s = 0; s < SC; ++s)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
              t[s][hh] = movm_trans(
                  pack_bf16x2(v[s][gi][2 * hh] * a, v[s][gi][2 * hh + 1] * a));
          if (drow[gi] >= 0) {
            __nv_bfloat16* o = p.out + static_cast<long>(drow[gi]) * Nout + n0;
#pragma unroll
            for (int s = 0; s < SC; ++s)
#pragma unroll
              for (int hh = 0; hh < 2; ++hh)
                if (n0 + s * 16 + hh * 8 < Nout)
                  *reinterpret_cast<uint32_t*>(o + s * 16 + hh * 8) = t[s][hh];
          }
        }
        return;
      }
    }
    // GEMM2 scatters rows: every destination row is loaded first (independent
    // loads within the block), so no load waits behind the stores before it; -1
    // marks a padding row.
    int dst[G][2];
    if constexpr (ACT == 0) {
      const int rows = p.block_rows[b];
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          dst[gi][h] = p.rowmap[b * R + wt * TW + gi * 8 + 2 * tq + h];
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (wt * TW + gi * 8 + 2 * tq + h >= rows) dst[gi][h] = -1;
    }
#pragma unroll
    for (int s = 0; s < SC; ++s) {
      const int ws = wc * SC + s;
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int r0 = wt * TW + gi * 8 + 2 * tq;
        if constexpr (ACT == 1) {
          const int i = c * NC * 64 + ws * 8 + g8;
          if (i < Nout) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const float gt = v[s][gi][h] * a, up = v[s][gi][2 + h] * a;
              float y;
              if constexpr (SC == 4 || (G1E && R == 96 && STAGES == 2 && !KT))
                y = silu_div(gt, 1.f + __expf(-gt)) * up;
              else
                y = gt / (1.f + __expf(-gt)) * up;
              p.out[(static_cast<long>(b) * R + r0 + h) * Nout + i] =
                  __float2bfloat16_rn(y);
            }
          }
        } else if constexpr (ACT == 2) {
          const int n = c * NC * kTileN + ws * 16 + g8;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int nn = n + (q >> 1) * 8;
            if (nn < Nout) {
              const float x = fmaxf(v[s][gi][q] * a, 0.f);
              p.out[(static_cast<long>(b) * R + r0 + (q & 1)) * Nout + nn] =
                  __float2bfloat16_rn(x * x);
            }
          }
        } else {
          const int n = c * NC * kTileN + ws * 16 + g8;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (dst[gi][h] >= 0) {
              const long f = dst[gi][h];
              if (n < Nout)
                p.out[f * Nout + n] = __float2bfloat16_rn(v[s][gi][h] * a);
              if (n + 8 < Nout)
                p.out[f * Nout + n + 8] =
                    __float2bfloat16_rn(v[s][gi][2 + h] * a);
            }
          }
        }
      }
    }
  };

  const uint32_t sw_c =
      sbase + (wc * SC / 8) * kRec + (wc * SC % 8) * 512 + lane * 16;
  const uint32_t ss_c =
      sbase + (wc * SC / 8) * kRec + kWBytes + (wc * SC % 8) * 128 + g8 * 16;
  const uint32_t sa_c =
      sbase + NC * kRec + (wt * TW + (lane & 7)) * kActPitch + (lane >> 3) * 16;
  // Swizzled: a lane's ldmatrix rows wt TW + gi 8 + (lane & 7) share r & 7 =
  // lane & 7, and step j2 reads logical chunk (lane >> 3) + 4 j2: j2 even sits
  // at sa_s0 + (j2 >> 1) 128 and j2 odd (chunk bit 2 set, which the XOR flips)
  // at sa_s1 + (j2 >> 1) 128
  const uint32_t sa_s0 = sbase + NC * kRec + (wt * TW + (lane & 7)) * AP +
                         (((lane >> 3)) ^ (lane & 7)) * 16;
  const uint32_t sa_s1 = sbase + NC * kRec + (wt * TW + (lane & 7)) * AP +
                         (((lane >> 3) + 4) ^ (lane & 7)) * 16;
  // R 96 at 2 stages with G1E: the row part of the half-fragment address
  const uint32_t sa_base_hf = sbase + NC * kRec + (wt * TW + (lane & 7)) * AP;
  float acc[SC][G][4];
#pragma unroll
  for (int s = 0; s < SC; ++s)
#pragma unroll
    for (int gi = 0; gi < G; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[s][gi][q] = 0.f;

  // Rows past the block's count are padding: zero activations whose results are
  // dropped. A warp runs the ldmatrix and MMA of its 8-row groups only up to
  // its last real row, in quarters of its G groups, so the group count is a
  // constant in each copy of the step and acc stays in registers.
  constexpr int Q =
      (Q12F && R == 96 && STAGES == 2 && !KT && (ACT == 0 || ACT == 1)) ? 12
      : R == 96                                                         ? 6
                : (SC > 1 && G % 4 == 0 ? 4 : 1);
  auto step = [&](auto ngc, uint32_t sb, bool hf) {
    constexpr int NG = decltype(ngc)::value;
    if constexpr (SC == 4) {
      // Four 16-column slots per warp (128 fp32 accumulators): a slot's weights
      // (8 B) and scales (4 B) of one k32 are read one k32 ahead, in place of
      // the whole k128 up front. Same operands, same MMA order.
      static_assert(
          G <= 8 && R == 64 && STAGES == 2,
          "SC 4 step: the k16 activation fragments of R 64, 2 stages");
      // VS (GEMM2 without a K tail, below sm_90): a slot's four scale words
      // of the k128 are read once with one 16 B load, in place of one 4 B
      // load per k32.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      constexpr bool kVS = false;
#else
      constexpr bool kVS = ACT == 0 && !KT && VS;
#endif
      uint32_t wn[SC][2], sn[SC];
      uint32_t sc[SC][4];
      if constexpr (kVS) {
#pragma unroll
        for (int s = 0; s < SC; ++s) lds128(sc[s], ss_c + sb + s * 128);
      }
      auto load_ws = [&](int j2) {
#pragma unroll
        for (int s = 0; s < SC; ++s) {
          lds64v(wn[s],
                 sw_c + sb + s * 512 + (j2 >> 1) * kHalfW + (j2 & 1) * 8);
          if constexpr (!kVS) lds32v(sn + s, ss_c + sb + s * 128 + j2 * 4);
        }
      };
      load_ws(0);
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (KT && j2 == 2 && hf) break;  // a 5120 B tail record: k 0..63 only
        uint32_t wq[SC][2], sq[SC];
#pragma unroll
        for (int s = 0; s < SC; ++s) {
          wq[s][0] = wn[s][0];
          wq[s][1] = wn[s][1];
          if constexpr (!kVS) sq[s] = sn[s];
        }
        if (j2 < 3 && !(KT && j2 == 1 && hf)) load_ws(j2 + 1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr bool kV3 = ACT != 0;  // sm_90 G1: v3 code (x4 spills there)
#else
        constexpr bool kV3 = false;
#endif
        if constexpr (kV3) {
          uint32_t bf[NG][2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int gi = 0; gi < NG; ++gi) {
              const uint32_t address =
                  sbase + NC * kRec + (wt * TW + gi * 8 + (lane & 7)) * AP +
                  (((((lane >> 3) & 1) + 2 * h + 4 * j2) ^ (lane & 7)) * 16);
              ldsm_x2(bf[gi], address + sb);
            }
#pragma unroll
            for (int s = 0; s < SC; ++s) {
              const uint32_t s2 = h ? u5_odd(sq[s]) : u5_even(sq[s]);
              uint32_t d[4], a[4];
              cvt31(wq[s][h], d);
              a[0] = mul_lo(d[0], s2);
              a[1] = mul_hi(d[1], s2);
              a[2] = mul_lo(d[2], s2);
              a[3] = mul_hi(d[3], s2);
#pragma unroll
              for (int gi = 0; gi < NG; ++gi) mma_bf16(acc[s][gi], a, bf[gi]);
            }
          }
        } else {
          // One ldmatrix x4 per 8-row group and k32: chunks 4 j2 .. 4 j2 + 3 of
          // the row (both k16 halves; bf[gi] + 2 h is half h), in place of one
          // x2 per half. Same fragments and MMA order, half the ldmatrix
          // instructions.
          uint32_t bf[NG][4];
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
            ldsm_x4(bf[gi], ((j2 & 1) ? sa_s1 : sa_s0) + sb + gi * 8 * AP +
                                (j2 >> 1) * 128);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int s = 0; s < SC; ++s) {
              const uint32_t sw = kVS ? sc[s][j2] : sq[s];
              const uint32_t s2 = h ? u5_odd(sw) : u5_even(sw);
              uint32_t d[4], a[4];
              cvt31(wq[s][h], d);
              a[0] = mul_lo(d[0], s2);
              a[1] = mul_hi(d[1], s2);
              a[2] = mul_lo(d[2], s2);
              a[3] = mul_hi(d[3], s2);
#pragma unroll
              for (int gi = 0; gi < NG; ++gi)
                mma_bf16(acc[s][gi], a, bf[gi] + 2 * h);
            }
          }
        }
      }
    } else {
      uint32_t wq[SC][8], sq[SC][4];
#pragma unroll
      for (int s = 0; s < SC; ++s) {
        lds128(wq[s], sw_c + sb + s * 512);
        lds128(wq[s] + 4, sw_c + sb + s * 512 + kHalfW);
        lds128(sq[s], ss_c + sb + s * 128);
      }
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (KT && j2 == 2 && hf) break;  // a 5120 B tail record: k 0..63 only
        constexpr bool half_fragment = G > 8 || (R == 64 && STAGES == 2);
        uint32_t bf[NG][half_fragment ? 2 : 4];
        auto load_activation = [&](int h) {
#pragma unroll
          for (int gi = 0; gi < NG; ++gi) {
            if constexpr (half_fragment) {
              const uint32_t address =
                  (R == 96 && STAGES == 2 && G1E
                       ? sa_base_hf + gi * 8 * AP
                       : sbase + NC * kRec +
                             (wt * TW + gi * 8 + (lane & 7)) * AP) +
                  (((((lane >> 3) & 1) + 2 * h + 4 * j2) ^ (lane & 7)) * 16);
              ldsm_x2(bf[gi], address + sb);
            } else if constexpr (SWZ) {
              ldsm_x4(bf[gi], ((j2 & 1) ? sa_s1 : sa_s0) + sb + gi * 8 * AP +
                                  (j2 >> 1) * 128);
            } else {
              ldsm_x4(bf[gi], sa_c + sb + gi * 8 * kActPitch + j2 * 64);
            }
          }
        };
        if constexpr (!half_fragment) load_activation(0);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (half_fragment) load_activation(h);
#pragma unroll
          for (int s = 0; s < SC; ++s) {
            const uint32_t s2 = h ? u5_odd(sq[s][j2]) : u5_even(sq[s][j2]);
            uint32_t d[4], a[4];
            constexpr bool kShfCvt = R == 96 && STAGES == 2 && G1E;
            cvt31<kShfCvt>(wq[s][j2 * 2 + h], d);
            a[0] = mul_lo(d[0], s2);
            a[1] = mul_hi(d[1], s2);
            a[2] = mul_lo(d[2], s2);
            a[3] = mul_hi(d[3], s2);
#pragma unroll
            for (int gi = 0; gi < NG; ++gi)
              mma_bf16(acc[s][gi], a, bf[gi] + (half_fragment ? 0 : 2 * h));
          }
        }
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (u0 + i < u1) issue();
    commit();
  }
  uint32_t cslot = 0;
  int u = u0;
  const int tile = blockIdx.x;
  {
    const int tstart = 0;
    const int u_end = S;
    // quarters of this warp's TW rows that hold a real row of the tile's block
    const int nq =
        SC == 1
            ? 1
            : min(Q,
                  (max(0, p.block_rows[tile / chunks] - wt * TW) * Q + TW - 1) /
                      TW);
    for (; u < u_end; ++u) {
      wait_group<STAGES - 2>();
      __syncthreads();
      if (u + STAGES - 1 < u1) issue();
      commit();
      const uint32_t sb = cslot;
      cslot = cslot + kStage == kRing ? 0u : cslot + kStage;
      bool hf = false;
      if constexpr (KT) hf = half && u - tstart == S - 1;
      if constexpr (Q == 4) {
        switch (nq) {
          case 0:
            break;
          case 1:
            step(std::integral_constant<int, G / 4>{}, sb, hf);
            break;
          case 2:
            step(std::integral_constant<int, G / 2>{}, sb, hf);
            break;
          case 3:
            step(std::integral_constant<int, 3 * G / 4>{}, sb, hf);
            break;
          default:
            step(std::integral_constant<int, G>{}, sb, hf);
            break;
        }
      } else if constexpr (Q == 6) {
        switch (nq) {
          case 0:
            break;
          case 1:
            step(std::integral_constant<int, G / 6>{}, sb, hf);
            break;
          case 2:
            step(std::integral_constant<int, 2 * G / 6>{}, sb, hf);
            break;
          case 3:
            step(std::integral_constant<int, 3 * G / 6>{}, sb, hf);
            break;
          case 4:
            step(std::integral_constant<int, 4 * G / 6>{}, sb, hf);
            break;
          case 5:
            step(std::integral_constant<int, 5 * G / 6>{}, sb, hf);
            break;
          default:
            step(std::integral_constant<int, G>{}, sb, hf);
            break;
        }
      } else if constexpr (Q == 12) {
        switch (nq) {
          case 0:
            break;
          case 1:
            step(std::integral_constant<int, 1 * G / 12>{}, sb, hf);
            break;
          case 2:
            step(std::integral_constant<int, 2 * G / 12>{}, sb, hf);
            break;
          case 3:
            step(std::integral_constant<int, 3 * G / 12>{}, sb, hf);
            break;
          case 4:
            step(std::integral_constant<int, 4 * G / 12>{}, sb, hf);
            break;
          case 5:
            step(std::integral_constant<int, 5 * G / 12>{}, sb, hf);
            break;
          case 6:
            step(std::integral_constant<int, 6 * G / 12>{}, sb, hf);
            break;
          case 7:
            step(std::integral_constant<int, 7 * G / 12>{}, sb, hf);
            break;
          case 8:
            step(std::integral_constant<int, 8 * G / 12>{}, sb, hf);
            break;
          case 9:
            step(std::integral_constant<int, 9 * G / 12>{}, sb, hf);
            break;
          case 10:
            step(std::integral_constant<int, 10 * G / 12>{}, sb, hf);
            break;
          case 11:
            step(std::integral_constant<int, 11 * G / 12>{}, sb, hf);
            break;
          default:
            step(std::integral_constant<int, G>{}, sb, hf);
            break;
        }
      } else if constexpr (Q == 3) {
        switch (nq) {
          case 0:
            break;
          case 1:
            step(std::integral_constant<int, G / 3>{}, sb, hf);
            break;
          case 2:
            step(std::integral_constant<int, 2 * G / 3>{}, sb, hf);
            break;
          default:
            step(std::integral_constant<int, G>{}, sb, hf);
            break;
        }
      } else if (nq > 0) {
        step(std::integral_constant<int, G>{}, sb, hf);
      }
    }
    write_tile(tile / chunks, tile % chunks, acc);
  }
}

template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1,
          bool G1E = false, bool Q12F = false, bool VS = false>
__global__ void __launch_bounds__(NW * 32) gemm(GemmParams p) {
  gemm_body<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS>(p);
}

// Routing -> blocks: one CTA, deterministic. Entry f = token * top_k + slot of
// topk_ids takes the rank of f among its expert's entries; experts get
// ceil(count / R) blocks in expert order. Warp w < W owns entries [w Q, (w + 1)
// Q) (Q a multiple of 32) and counts them per expert in its own table row, so
// ranks need no atomics: lower warps' counts, then lanes below in the same 32.
template <int R>
__global__ void __launch_bounds__(1024)
    prep(const int* __restrict__ ids, int M, int E, int W, int Q, int* nblocks,
         int* block_expert, int* block_rows, int* rowmap) {
  extern __shared__ int sh[];
  int* tab = sh;  // [W][E]: warp w's entries of expert e, then its next rank
  int* cnt = sh + W * E;  // [E]
  int* off = cnt + E;     // [E] first block of each expert
  __shared__ int wsum[32];
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5,
            lane = tid & 31;
  for (int i = tid; i < W * E; i += nt) tab[i] = 0;
  __syncthreads();
  const int f0 = warp * Q, f1 = min(f0 + Q, M);
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && lane == __ffs(peers) - 1)
        tab[warp * E + e] += __popc(peers);
      __syncwarp();  // the next 32 may give this expert another leader lane
    }
  }
  __syncthreads();
  // This thread's experts [e0, e1): counts, warp prefixes, blocks; contiguous
  // runs keep expert order.
  const int per = (E + nt - 1) / nt;
  const int e0 = min(tid * per, E), e1 = min(e0 + per, E);
  int v = 0;
  for (int e = e0; e < e1; ++e) {
    int c = 0;
    for (int w = 0; w < W; ++w) {
      const int t = tab[w * E + e];
      tab[w * E + e] = c;
      c += t;
    }
    cnt[e] = c;
    v += (c + R - 1) / R;
  }
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (tid < 32) {
    int s = tid < static_cast<int>(nt >> 5) ? wsum[tid] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, o);
      if (tid >= o) s += y;
    }
    wsum[tid] = s;
  }
  __syncthreads();
  const int excl = x - v + (warp > 0 ? wsum[warp - 1] : 0);
  int b = excl;
  for (int e = e0; e < e1; ++e) {
    const int c = cnt[e], n = (c + R - 1) / R;
    off[e] = b;
    for (int bb = 0; bb < n; ++bb) {
      const int rows = min(R, c - bb * R);
      block_expert[b + bb] = e;
      block_rows[b + bb] = rows;
      for (int r = rows; r < R; ++r) rowmap[(b + bb) * R + r] = -1;
    }
    b += n;
  }
  if (tid == nt - 1) *nblocks = excl + v;
  __syncthreads();
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0) {
        const int q = tab[warp * E + e] + __popc(peers & ((1u << lane) - 1u));
        rowmap[(off[e] + q / R) * R + q % R] = f;
      }
      __syncwarp();
      if (e >= 0 && lane == __ffs(peers) - 1)
        tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
}

// Multi-CTA prep, the same blocks and rowmap as prep (entries of an expert in f
// order): CTA c of nC owns entries [c CH, (c + 1) CH). prepx_count: per-chunk
// expert counts pc[c][E]. prepx_scan (one CTA): pc becomes each chunk's first
// rank within its expert, blocks as in prep, off[e] = the expert's first block.
// prepx_place: prep's two passes over the chunk (warp w owns [f0 + w Q, f0 + (w
// + 1) Q)), ranks offset by the chunk's base.
__global__ void __launch_bounds__(1024)
    prepx_count(const int* __restrict__ ids, int M, int E, int CH,
                int* __restrict__ pc) {
  extern __shared__ int hist[];
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int i = tid; i < E; i += nt) hist[i] = 0;
  __syncthreads();
  const int f0 = blockIdx.x * CH, f1 = min(f0 + CH, M);
  for (int f = f0 + tid; f < f1; f += nt) {
    const int e = ids[f];
    if (e >= 0) atomicAdd(hist + e, 1);  // as prep: a negative id has no slot
  }
  __syncthreads();
  for (int i = tid; i < E; i += nt) pc[blockIdx.x * E + i] = hist[i];
}

template <int R>
__global__ void __launch_bounds__(1024)
    prepx_scan(int E, int nC, int* __restrict__ pc, int* __restrict__ off,
               int* nblocks, int* block_expert, int* block_rows, int* rowmap) {
  extern __shared__ int cnt[];  // [E]
  __shared__ int wsum[32];
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5,
            lane = tid & 31;
  const int per = (E + nt - 1) / nt;
  const int e0 = min(tid * per, E), e1 = min(e0 + per, E);
  int v = 0;
  for (int e = e0; e < e1; ++e) {
    int c = 0;
    for (int k0 = 0; k0 < nC;
         k0 += 8) {  // 8 chunk counts loaded before their ranks are stored
      int t[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) t[j] = k0 + j < nC ? pc[(k0 + j) * E + e] : 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (k0 + j < nC) {
          pc[(k0 + j) * E + e] = c;
          c += t[j];
        }
    }
    cnt[e] = c;
    v += (c + R - 1) / R;
  }
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (tid < 32) {
    int s = tid < static_cast<int>(nt >> 5) ? wsum[tid] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, o);
      if (tid >= o) s += y;
    }
    wsum[tid] = s;
  }
  __syncthreads();
  const int excl = x - v + (warp > 0 ? wsum[warp - 1] : 0);
  int b = excl;
  for (int e = e0; e < e1; ++e) {
    const int c = cnt[e], n = (c + R - 1) / R;
    off[e] = b;
    for (int bb = 0; bb < n; ++bb) {
      const int rows = min(R, c - bb * R);
      block_expert[b + bb] = e;
      block_rows[b + bb] = rows;
      for (int r = rows; r < R; ++r) rowmap[(b + bb) * R + r] = -1;
    }
    b += n;
  }
  if (tid == nt - 1) *nblocks = excl + v;
}

template <int R>
__global__ void __launch_bounds__(1024)
    prepx_place(const int* __restrict__ ids, int M, int E, int CH, int W, int Q,
                const int* __restrict__ pc, const int* __restrict__ off,
                int* __restrict__ rowmap) {
  extern __shared__ int sh[];
  int* tab = sh;  // [W][E]: warp w's entries of expert e, then its next rank
  int* base = sh + W * E;  // [E]: the expert's first row (off R) plus this
                           // chunk's first rank
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5,
            lane = tid & 31;
  for (int i = tid; i < W * E; i += nt) tab[i] = 0;
  __syncthreads();
  const int c0 = blockIdx.x * CH, c1 = min(c0 + CH, M);
  const int f0 = c0 + warp * Q, f1 = min(f0 + Q, c1);
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && lane == __ffs(peers) - 1)
        tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
  __syncthreads();
  for (int e = tid; e < E; e += nt) {
    int c = 0;
    for (int w = 0; w < W; ++w) {
      const int t = tab[w * E + e];
      tab[w * E + e] = c;
      c += t;
    }
    base[e] = off[e] * R + pc[blockIdx.x * E + e];
  }
  __syncthreads();
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0)
        rowmap[base[e] + tab[warp * E + e] +
               __popc(peers & ((1u << lane) - 1u))] = f;
      __syncwarp();
      if (e >= 0 && lane == __ffs(peers) - 1)
        tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
}

// out[t][h] = sum_j w[t][j] * y[t * top_k + j][h], fp32 in slot order. 8
// columns per thread.
__global__ void finalize(const __nv_bfloat16* __restrict__ y,
                         const float* __restrict__ w,
                         __nv_bfloat16* __restrict__ out, int tokens, int top_k,
                         int H) {
  const int h8 = H / 8;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tokens * h8) return;
  const int t = idx / h8, c = idx % h8;
  float a[8] = {};
  for (int j = 0; j < top_k; ++j) {
    const float wj = w[t * top_k + j];
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(
        y + (static_cast<long>(t) * top_k + j) * H + c * 8));
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(b2[i]);
      a[2 * i] += wj * f.x;
      a[2 * i + 1] += wj * f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o2[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
  *reinterpret_cast<uint4*>(out + static_cast<long>(t) * H + c * 8) = o;
}

template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1,
          bool G1E = false, bool Q12F = false, bool VS = false>
void launch_gemm(GemmParams const& p, int grid, cudaStream_t st) {
  static_assert(NC > 1 || STAGES * Cfg<R, NC>::kStageBytes <= kSmemLimit);
  const int smem = STAGES * Cfg<R, NC>::kStageBytes;
  STD_TORCH_CHECK(smem <= get_device_prop()->sharedMemPerBlockOptin,
                  "rs2_moe_gemm: tile exceeds device shared memory");
  cudaFuncSetAttribute(gemm<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS>
      <<<grid, NW * 32, smem, st>>>(p);
}

// R 8: 8 warps at 4 stages; R 16: 8 warps at 2 stages; R 32: 8 warps at 4
// stages, 2 where an SM has at most 100 KiB of shared memory. R 96: two
// records per CTA at 3 stages where the device allows it, else 2. R 64: one
// row group at 3 stages (two CTAs per SM by shared memory); GEMM1 of
// rs2_moe_gemm_stage2 runs it at 2 stages.
template <int ACT, bool KT, bool G1S2 = false, bool G1E = false,
          bool Q12F = false>
void launch_tile(int R, GemmParams const& p, int grid, cudaStream_t st) {
  if (R == 8)
    launch_gemm<8, 8, 8, ACT, 4, KT>(p, grid, st);
  else if (R == 16)
    launch_gemm<16, 8, 8, ACT, 2, KT>(p, grid, st);
  else if (R == 32) {
    if (get_device_prop()->sharedMemPerMultiprocessor <= 100 * 1024)
      launch_gemm<32, 8, 8, ACT, 2, KT>(p, grid, st);
    else
      launch_gemm<32, 8, 8, ACT, 4, KT>(p, grid, st);
  } else if (R == 96) {
    if (get_device_prop()->sharedMemPerBlockOptin >=
        3 * Cfg<96, 2>::kStageBytes)
      launch_gemm<96, 8, 8, ACT, 3, KT, 2, G1E, Q12F>(p, grid, st);
    else
      launch_gemm<96, 8, 8, ACT, 2, KT, 2, G1E, Q12F>(p, grid, st);
  } else
    launch_gemm<64, 4, 4, ACT, G1S2 && ACT != 0 ? 2 : 3, KT>(p, grid, st);
}

template <int R>
int prep_static_smem() {  // prep's own __shared__ bytes (wsum)
  static const int s = [] {
    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, prep<R>);
    return static_cast<int>(a.sharedSizeBytes);
  }();
  return s;
}

constexpr int kPrepChunk = 2048;  // multi-CTA prep: routed entries per CTA

}  // namespace vllm::rs2_moe

// topk_ids [tokens, top_k] int32 -> meta: [0] block count, [1, 1 + max_blocks)
// each block's expert, [1 + max_blocks, 1 + 2 max_blocks) its row count; rowmap
// [max_blocks x row_tile]: each block row's entry (token * top_k + slot) or -1.
// Row tile 64 takes the multi-CTA prep, whose per-chunk expert counts and bases
// live in workspace (at least (ceil(M / 2048) + 1) x num_experts int32).
void rs2_moe_prep(torch::stable::Tensor const& topk_ids,
                  torch::stable::Tensor& meta, torch::stable::Tensor& rowmap,
                  torch::stable::Tensor& workspace, int64_t num_experts,
                  int64_t row_tile, int64_t max_blocks) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int &&
                      meta.scalar_type() == ScalarType::Int &&
                      rowmap.scalar_type() == ScalarType::Int &&
                      workspace.scalar_type() == ScalarType::Int,
                  "rs2_moe_prep: int32 tensors");
  STD_TORCH_CHECK(topk_ids.is_contiguous(),
                  "rs2_moe_prep: topk_ids must be contiguous");
  STD_TORCH_CHECK(row_tile == 8 || row_tile == 16 || row_tile == 32 ||
                      row_tile == 64 || row_tile == 96,
                  "rs2_moe_prep: row_tile");
  const int R = static_cast<int>(row_tile);
  const int M = static_cast<int>(topk_ids.numel());
  const int e = static_cast<int>(num_experts);
  const int nb = static_cast<int>(max_blocks);
  STD_TORCH_CHECK(e >= 1, "rs2_moe_prep: num_experts");
  STD_TORCH_CHECK(nb >= (M + R - 1) / R + std::min(e, M),
                  "rs2_moe_prep: max_blocks below the routing bound");
  STD_TORCH_CHECK(meta.numel() >= 1 + 2 * max_blocks &&
                      rowmap.numel() >= max_blocks * row_tile,
                  "rs2_moe_prep: meta or rowmap too small");
  const torch::stable::accelerator::DeviceGuard device_guard(
      topk_ids.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(topk_ids.get_device_index());
  const int max_smem =
      static_cast<int>(get_device_prop()->sharedMemPerBlockOptin);
  const int* ids = static_cast<const int*>(topk_ids.const_data_ptr());
  int* m = static_cast<int*>(meta.mutable_data_ptr());
  int* rm = static_cast<int*>(rowmap.mutable_data_ptr());

  if (R < 64) {
    const int stat = R == 8    ? prep_static_smem<8>()
                     : R == 16 ? prep_static_smem<16>()
                               : prep_static_smem<32>();
    const int tab_ints =
        (max_smem - stat) / static_cast<int>(sizeof(int));  // [W][E] + 2 x [E]
    STD_TORCH_CHECK(3 * e <= tab_ints,
                    "rs2_moe_prep: num_experts too large for the shared table");
    const int W = std::max(1, std::min({32, tab_ints / e - 2, (M + 31) / 32}));
    const int Q = ((M + W - 1) / W + 31) / 32 * 32;
    const int smem = (W + 2) * e * static_cast<int>(sizeof(int));
    auto run = [&](auto kern) {
      // the 48 KB default bounds dynamic plus static shared memory
      if (smem + stat > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem);
      kern<<<1, 1024, smem, st>>>(ids, M, e, W, Q, m, m + 1, m + 1 + nb, rm);
    };
    if (R == 8)
      run(prep<8>);
    else if (R == 16)
      run(prep<16>);
    else
      run(prep<32>);
    return;
  }

  const int ch = kPrepChunk;
  const int nC = std::max(1, (M + ch - 1) / ch);
  STD_TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(nC + 1) * e,
                  "rs2_moe_prep: workspace too small");
  const int ints = max_smem / static_cast<int>(sizeof(int)) -
                   64;  // [W][E] + [E], static wsum aside
  STD_TORCH_CHECK(2 * e <= ints,
                  "rs2_moe_prep: num_experts too large for the shared table");
  const int W = std::max(1, std::min({32, ints / e - 1, (ch + 31) / 32}));
  const int Q = ((ch + W - 1) / W + 31) / 32 * 32;
  int* pc = static_cast<int*>(workspace.mutable_data_ptr());
  int* off = pc + static_cast<long>(nC) * e;
  const int sm_e = e * static_cast<int>(sizeof(int)), sm_p = (W + 1) * sm_e;
  auto big = [&](auto kern, int smem) {
    if (smem > 48 * 1024 - 256)
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem);
  };
  big(prepx_count, sm_e);
  auto route = [&](auto tile) {
    constexpr int RT = decltype(tile)::value;
    big(prepx_scan<RT>, sm_e);
    big(prepx_place<RT>, sm_p);
    prepx_count<<<nC, 1024, sm_e, st>>>(ids, M, e, ch, pc);
    prepx_scan<RT>
        <<<1, 1024, sm_e, st>>>(e, nC, pc, off, m, m + 1, m + 1 + nb, rm);
    prepx_place<RT><<<nC, 1024, sm_p, st>>>(ids, M, e, ch, W, Q, pc, off, rm);
  };
  if (R == 96)
    route(std::integral_constant<int, 96>{});
  else
    route(std::integral_constant<int, 64>{});
}

// One GEMM over the routed blocks. GEMM1 (activation 1: silu(gate) * up, 2:
// relu(x)^2) reads act [tokens][K] through rowmap and writes the blocked
// intermediate [max_blocks x row_tile][Nout]; GEMM2 (activation 0) reads that
// and writes [tokens * top_k][Nout]. records: [E][N / 128 chunks][k steps]; N
// is the record column count, a multiple of 128 (gated: gate and up each padded
// to 64).
template <bool G1S2 = false, bool G1E = false, bool Q12F = false>
void rs2_moe_gemm(torch::stable::Tensor const& records,
                  torch::stable::Tensor const& alpha,
                  torch::stable::Tensor const& act,
                  torch::stable::Tensor const& meta,
                  torch::stable::Tensor const& rowmap,
                  torch::stable::Tensor& out, int64_t row_tile,
                  int64_t activation, int64_t top_k, int64_t N, int64_t K,
                  int64_t Nout, int64_t max_blocks) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(records.scalar_type() == ScalarType::Byte &&
                      alpha.scalar_type() == ScalarType::Float &&
                      act.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "rs2_moe_gemm: dtypes");
  STD_TORCH_CHECK(
      records.is_contiguous() && act.is_contiguous() && out.is_contiguous(),
      "rs2_moe_gemm: records, act and out must be contiguous");
  STD_TORCH_CHECK(N % kTileN == 0 && K % 16 == 0 && Nout % 8 == 0 && Nout <= N,
                  "rs2_moe_gemm: shape");
  STD_TORCH_CHECK(row_tile == 8 || row_tile == 16 || row_tile == 32 ||
                      row_tile == 64 || row_tile == 96,
                  "rs2_moe_gemm: row_tile");
  STD_TORCH_CHECK(activation >= 0 && activation <= 2,
                  "rs2_moe_gemm: activation");
  const torch::stable::accelerator::DeviceGuard device_guard(
      act.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(act.get_device_index());
  const int* m = static_cast<const int*>(meta.const_data_ptr());
  GemmParams p;
  p.w = static_cast<const uint8_t*>(records.const_data_ptr());
  p.alpha = static_cast<const float*>(alpha.const_data_ptr());
  p.act = static_cast<const __nv_bfloat16*>(act.const_data_ptr());
  p.nblocks = m;
  p.block_expert = m + 1;
  p.block_rows = m + 1 + max_blocks;
  p.rowmap = static_cast<const int*>(rowmap.const_data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  p.ws = nullptr;  // one whole tile per CTA: no split-tile partials
  p.cnt = nullptr;
  if (activation != 0)
    p.topk_dm = cutlass::FastDivmod(
        static_cast<int>(top_k));  // G1 only; G2 keeps identity
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.Nout = static_cast<int>(Nout);
  p.units_per_cta =
      static_cast<int>((K + kTileK - 1) / kTileK);  // all k steps of one tile
  const int columns = static_cast<int>(N / kTileN);
  const int grid = static_cast<int>(max_blocks) *
                   (row_tile == 96 ? (columns + 1) / 2 : columns);
  const int R = static_cast<int>(row_tile);
  const bool kt = K % kTileK != 0;
  if (activation == 0)
    kt ? launch_tile<0, true, G1S2, G1E, Q12F>(R, p, grid, st)
       : launch_tile<0, false, G1S2, G1E, Q12F>(R, p, grid, st);
  else if (activation == 1)
    kt ? launch_tile<1, true, G1S2, G1E, Q12F>(R, p, grid, st)
       : launch_tile<1, false, G1S2, G1E, Q12F>(R, p, grid, st);
  else
    kt ? launch_tile<2, true, G1S2, G1E, Q12F>(R, p, grid, st)
       : launch_tile<2, false, G1S2, G1E, Q12F>(R, p, grid, st);
}

// out[t][h] = sum_j topk_weights[t][j] * y[t * top_k + j][h], fp32 in slot
// order.
void rs2_moe_finalize(torch::stable::Tensor const& y,
                      torch::stable::Tensor const& topk_weights,
                      torch::stable::Tensor& out, int64_t top_k) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(y.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16 &&
                      topk_weights.scalar_type() == ScalarType::Float,
                  "rs2_moe_finalize: dtypes");
  STD_TORCH_CHECK(y.is_contiguous() && topk_weights.is_contiguous() &&
                      out.is_contiguous() && out.dim() == 2,
                  "rs2_moe_finalize: contiguous 2D out");
  const int tokens = static_cast<int>(out.size(0)),
            H = static_cast<int>(out.size(1));
  STD_TORCH_CHECK(H % 8 == 0, "rs2_moe_finalize: hidden size");
  const torch::stable::accelerator::DeviceGuard device_guard(
      out.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(out.get_device_index());
  const int n = tokens * (H / 8);
  finalize<<<(n + 255) / 256, 256, 0, st>>>(
      static_cast<const __nv_bfloat16*>(y.const_data_ptr()),
      static_cast<const float*>(topk_weights.const_data_ptr()),
      static_cast<__nv_bfloat16*>(out.mutable_data_ptr()), tokens,
      static_cast<int>(top_k), H);
}

// GEMM2 (activation 0) at row tile 64 with 2 stages instead of 3, on the same
// gemm_body. Valid only for row_tile 64, activation 0.
void rs2_moe_gemm_g2lin(torch::stable::Tensor const& records,
                        torch::stable::Tensor const& alpha,
                        torch::stable::Tensor const& act,
                        torch::stable::Tensor const& meta,
                        torch::stable::Tensor const& rowmap,
                        torch::stable::Tensor& out, int64_t row_tile,
                        int64_t activation, int64_t top_k, int64_t N, int64_t K,
                        int64_t Nout, int64_t max_blocks) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(records.scalar_type() == ScalarType::Byte &&
                      alpha.scalar_type() == ScalarType::Float &&
                      act.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "rs2_moe_gemm_g2lin: dtypes");
  STD_TORCH_CHECK(
      records.is_contiguous() && act.is_contiguous() && out.is_contiguous(),
      "rs2_moe_gemm_g2lin: records, act and out must be contiguous");
  STD_TORCH_CHECK(N % kTileN == 0 && K % 16 == 0 && Nout % 8 == 0 && Nout <= N,
                  "rs2_moe_gemm_g2lin: shape");
  STD_TORCH_CHECK(row_tile == 64 && activation == 0,
                  "rs2_moe_gemm_g2lin: only row_tile 64, activation 0");
  const torch::stable::accelerator::DeviceGuard device_guard(
      act.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(act.get_device_index());
  const int* m = static_cast<const int*>(meta.const_data_ptr());
  GemmParams p;
  p.w = static_cast<const uint8_t*>(records.const_data_ptr());
  p.alpha = static_cast<const float*>(alpha.const_data_ptr());
  p.act = static_cast<const __nv_bfloat16*>(act.const_data_ptr());
  p.nblocks = m;
  p.block_expert = m + 1;
  p.block_rows = m + 1 + max_blocks;
  p.rowmap = static_cast<const int*>(rowmap.const_data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  p.ws = nullptr;
  p.cnt = nullptr;
  // G2 (activation 0): topk_dm unused on device; keep default identity (no host
  // divmod).
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.Nout = static_cast<int>(Nout);
  p.units_per_cta = static_cast<int>((K + kTileK - 1) / kTileK);
  const int columns = static_cast<int>(N / kTileN);
  const int grid = static_cast<int>(max_blocks) * columns;  // R 64 is not 96
  const bool kt = K % kTileK != 0;
  kt ? launch_gemm<64, 4, 4, 0, 2, true>(p, grid, st)
     : launch_gemm<64, 4, 4, 0, 2, false>(p, grid, st);
}

// GEMM1 at row tile 64 with two 128-column records per CTA: 4 warps, each 64
// rows x 64 columns (SC 4), two stages. Same R 64 blocks, rowmap and blocked
// intermediate as rs2_moe_gemm_stage2.
void rs2_moe_gemm_w64s(torch::stable::Tensor const& records,
                       torch::stable::Tensor const& alpha,
                       torch::stable::Tensor const& act,
                       torch::stable::Tensor const& meta,
                       torch::stable::Tensor const& rowmap,
                       torch::stable::Tensor& out, int64_t row_tile,
                       int64_t activation, int64_t top_k, int64_t N, int64_t K,
                       int64_t Nout, int64_t max_blocks) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(records.scalar_type() == ScalarType::Byte &&
                      alpha.scalar_type() == ScalarType::Float &&
                      act.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "rs2_moe_gemm_w64s: dtypes");
  STD_TORCH_CHECK(
      records.is_contiguous() && act.is_contiguous() && out.is_contiguous(),
      "rs2_moe_gemm_w64s: records, act and out must be contiguous");
  STD_TORCH_CHECK(N % kTileN == 0 && K % 16 == 0 && Nout % 8 == 0 && Nout <= N,
                  "rs2_moe_gemm_w64s: shape");
  STD_TORCH_CHECK(row_tile == 64 && (activation == 1 || activation == 2),
                  "rs2_moe_gemm_w64s: only row_tile 64, activation 1 or 2");
  const torch::stable::accelerator::DeviceGuard device_guard(
      act.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(act.get_device_index());
  const int* m = static_cast<const int*>(meta.const_data_ptr());
  GemmParams p;
  p.w = static_cast<const uint8_t*>(records.const_data_ptr());
  p.alpha = static_cast<const float*>(alpha.const_data_ptr());
  p.act = static_cast<const __nv_bfloat16*>(act.const_data_ptr());
  p.nblocks = m;
  p.block_expert = m + 1;
  p.block_rows = m + 1 + max_blocks;
  p.rowmap = static_cast<const int*>(rowmap.const_data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  p.ws = nullptr;
  p.cnt = nullptr;
  p.topk_dm = cutlass::FastDivmod(static_cast<int>(top_k));
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.Nout = static_cast<int>(Nout);
  p.units_per_cta = static_cast<int>((K + kTileK - 1) / kTileK);
  const int columns = static_cast<int>(N / kTileN);
  const int grid = static_cast<int>(max_blocks) * ((columns + 1) / 2);
  const bool kt = K % kTileK != 0;
  if (activation == 1)
    kt ? launch_gemm<64, 4, 4, 1, 2, true, 2>(p, grid, st)
       : launch_gemm<64, 4, 4, 1, 2, false, 2>(p, grid, st);
  else
    kt ? launch_gemm<64, 4, 4, 2, 2, true, 2>(p, grid, st)
       : launch_gemm<64, 4, 4, 2, 2, false, 2>(p, grid, st);
}

// GEMM2 at row tile 64 with two 128-column records per CTA: 4 warps, each 64
// rows x 64 columns (SC 4), two stages. Same R 64 blocks, blocked intermediate
// and y2 rows as rs2_moe_gemm_g2lin. Valid only for row_tile 64, activation 0.
template <bool VS>
void rs2_moe_gemm_g2w64x(torch::stable::Tensor const& records,
                         torch::stable::Tensor const& alpha,
                         torch::stable::Tensor const& act,
                         torch::stable::Tensor const& meta,
                         torch::stable::Tensor const& rowmap,
                         torch::stable::Tensor& out, int64_t row_tile,
                         int64_t activation, int64_t top_k, int64_t N,
                         int64_t K, int64_t Nout, int64_t max_blocks) {
  using namespace vllm::rs2_moe;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(records.scalar_type() == ScalarType::Byte &&
                      alpha.scalar_type() == ScalarType::Float &&
                      act.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "rs2_moe_gemm_g2w64x: dtypes");
  STD_TORCH_CHECK(
      records.is_contiguous() && act.is_contiguous() && out.is_contiguous(),
      "rs2_moe_gemm_g2w64x: records, act and out must be contiguous");
  STD_TORCH_CHECK(N % kTileN == 0 && K % 16 == 0 && Nout % 8 == 0 && Nout <= N,
                  "rs2_moe_gemm_g2w64x: shape");
  STD_TORCH_CHECK(row_tile == 64 && activation == 0,
                  "rs2_moe_gemm_g2w64x: only row_tile 64, activation 0");
  const torch::stable::accelerator::DeviceGuard device_guard(
      act.get_device_index());
  const cudaStream_t st = get_current_cuda_stream(act.get_device_index());
  const int* m = static_cast<const int*>(meta.const_data_ptr());
  GemmParams p;
  p.w = static_cast<const uint8_t*>(records.const_data_ptr());
  p.alpha = static_cast<const float*>(alpha.const_data_ptr());
  p.act = static_cast<const __nv_bfloat16*>(act.const_data_ptr());
  p.nblocks = m;
  p.block_expert = m + 1;
  p.block_rows = m + 1 + max_blocks;
  p.rowmap = static_cast<const int*>(rowmap.const_data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  p.ws = nullptr;
  p.cnt = nullptr;
  // G2 (activation 0): topk_dm unused on device; keep default identity (no host
  // divmod).
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.Nout = static_cast<int>(Nout);
  p.units_per_cta = static_cast<int>((K + kTileK - 1) / kTileK);
  const int columns = static_cast<int>(N / kTileN);
  const int grid = static_cast<int>(max_blocks) * ((columns + 1) / 2);
  const bool kt = K % kTileK != 0;
  kt ? launch_gemm<64, 4, 4, 0, 2, true, 2, false, false, VS>(p, grid, st)
     : launch_gemm<64, 4, 4, 0, 2, false, 2, false, false, VS>(p, grid, st);
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("rs2_moe_prep", TORCH_BOX(&rs2_moe_prep));
  m.impl("rs2_moe_gemm", TORCH_BOX(&rs2_moe_gemm<false>));
  m.impl("rs2_moe_finalize", TORCH_BOX(&rs2_moe_finalize));
  m.impl("rs2_moe_gemm_stage2", TORCH_BOX(&rs2_moe_gemm<true>));
  m.impl("rs2_moe_gemm_g2lin", TORCH_BOX(&rs2_moe_gemm_g2lin));
  m.impl("rs2_moe_gemm_w64s", TORCH_BOX(&rs2_moe_gemm_w64s));
  m.impl("rs2_moe_gemm_g2w64x", TORCH_BOX(&rs2_moe_gemm_g2w64x<false>));
  m.impl("rs2_moe_gemm_g2w64v", TORCH_BOX(&rs2_moe_gemm_g2w64x<true>));
  m.impl("rs2_moe_gemm_x96", TORCH_BOX((&rs2_moe_gemm<false, true, false>)));
  m.impl("rs2_moe_gemm_q12", TORCH_BOX((&rs2_moe_gemm<false, true, true>)));
}
