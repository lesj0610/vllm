// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The Qwen4Exp hyper-connection ops.
//
// A layer's residual carries hc_count streams side by side. These five ops are
// what reads and writes that layout: a grouped RMSNorm over each stream, the
// gated mix down to one stream, the injection back out to all of them, and the
// combine fused with the norm that follows it.
//
// Four of the five move a residual-sized tensor in and one out, which on this
// device is the whole cost -- they run at the memory's speed and the only thing
// the code has to do is not get in the way: sixteen bytes a thread, a grid that
// covers the device, and no arithmetic on the address that a shift will not do.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "core/registration.h"
#include "cuda_utils.h"
#include "ops.h"

namespace {

constexpr int kWarp = 32;
// Widths the elementwise kernels launch with. The pure stream (silu) wants the
// wider block; the ones that walk every stream of a row want more blocks.
constexpr int kFlatBlock = 256;
constexpr int kBlock = 128;
// A warp owns a row of the reduction kernels, so the sum crosses lanes and
// never the block: a block-wide reduction would cost two barriers per row, and
// a row is small enough that a warp covers it in a few passes.
constexpr int kRowBlock = 128;
constexpr int kRowsPerBlock = kRowBlock / kWarp;
// Elements a thread moves at a time: sixteen bytes of a half-width type.
constexpr int kVec = 8;

// The reciprocal is the approximate one: a plain division here is IEEE-correct
// and lowers to a call into a slow path, which on a kernel this simple is most
// of the arithmetic. Two ulp is far below what the half-width result keeps.
__device__ __forceinline__ float sigmoidf_(float v) {
  return __fdividef(1.f, 1.f + __expf(-v));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// A vector of kVec elements of the tensor's own type, moved as one access.
// The alignment is what makes it one: without it the array's alignment is the
// element's, and every access lowers to kVec of them.
template <typename T>
struct __align__(kVec * sizeof(T)) Vec {
  T v[kVec];
};

// The alignment is the point of the type, so it is checked rather than
// assumed: without it an access lowers to one per element and the kernel runs
// at a third of the memory's speed with nothing else looking wrong.
static_assert(sizeof(Vec<__nv_bfloat16>) == 16,
              "the vector must be sixteen bytes");
static_assert(alignof(Vec<__nv_bfloat16>) == 16,
              "the vector must be sixteen-byte aligned");
static_assert(sizeof(Vec<__half>) == 16, "the vector must be sixteen bytes");
static_assert(alignof(Vec<__half>) == 16,
              "the vector must be sixteen-byte aligned");

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half(x);
}

/*!
 * \brief Gemma RMSNorm over each of a row's groups.
 *
 * One block owns one group. The affine is either a group's worth, shared by
 * every group, or the whole row's, following the grouped checkpoint layout.
 */
// ROW_VECS is the row's width in per-lane vectors, or zero for a width this is
// not built for. Knowing it makes the trip count compile-time, which is what
// lets the row stay in registers between the reduction and the affine rather
// than being read a second time.
template <typename T, bool W_SHARED, int ROW_VECS>
__global__ void __launch_bounds__(kRowBlock)
    grouped_gemma_rmsnorm_kernel(const T* __restrict__ x,
                                 const T* __restrict__ w, T* __restrict__ y,
                                 int64_t stride_x, int64_t stride_y,
                                 int group_dim, int num_groups, int64_t units,
                                 float eps) {
  const int lane = threadIdx.x % kWarp;
  const int64_t unit =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x / kWarp;
  if (unit >= units) return;
  const int group = static_cast<int>(unit % num_groups);
  const int64_t row = unit / num_groups;
  const T* src = x + row * stride_x + static_cast<int64_t>(group) * group_dim;
  T* dst = y + row * stride_y + static_cast<int64_t>(group) * group_dim;
  const T* weight =
      w + (W_SHARED ? 0 : static_cast<int64_t>(group) * group_dim);

  float sum = 0.f;
  Vec<T> held[ROW_VECS > 0 ? ROW_VECS : 1];
  if constexpr (ROW_VECS > 0) {
#pragma unroll
    for (int r = 0; r < ROW_VECS; ++r) {
      held[r] =
          *reinterpret_cast<const Vec<T>*>(src + (lane + r * kWarp) * kVec);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float f = to_float<T>(held[r].v[j]);
        sum += f * f;
      }
    }
  } else {
    for (int i = lane * kVec; i < group_dim; i += kWarp * kVec) {
      const Vec<T> v = *reinterpret_cast<const Vec<T>*>(src + i);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float f = to_float<T>(v.v[j]);
        sum += f * f;
      }
    }
  }
  const float rrms =
      rsqrtf(warp_sum(sum) / static_cast<float>(group_dim) + eps);

  // Gemma's one-plus-weight affine, written as the caller writes it so the
  // rounding matches: scale first, then add the weighted scale.
  if constexpr (ROW_VECS > 0) {
#pragma unroll
    for (int r = 0; r < ROW_VECS; ++r) {
      const int i = (lane + r * kWarp) * kVec;
      const Vec<T> g = *reinterpret_cast<const Vec<T>*>(weight + i);
      Vec<T> out;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float sc = to_float<T>(held[r].v[j]) * rrms;
        out.v[j] = from_float<T>(sc + sc * to_float<T>(g.v[j]));
      }
      *reinterpret_cast<Vec<T>*>(dst + i) = out;
    }
  } else {
    for (int i = lane * kVec; i < group_dim; i += kWarp * kVec) {
      const Vec<T> v = *reinterpret_cast<const Vec<T>*>(src + i);
      const Vec<T> g = *reinterpret_cast<const Vec<T>*>(weight + i);
      Vec<T> out;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float sc = to_float<T>(v.v[j]) * rrms;
        out.v[j] = from_float<T>(sc + sc * to_float<T>(g.v[j]));
      }
      *reinterpret_cast<Vec<T>*>(dst + i) = out;
    }
  }
}

/*!
 * \brief SiLU of a residual scaled down by the stream count.
 */
// The rows are contiguous, so the whole tensor is one run and the grid-stride
// step needs no division to find its place in it.
template <typename T>
__global__ void hc_silu_kernel(const T* __restrict__ x, T* __restrict__ y,
                               int64_t total, float inv_hc) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t off = idx * kVec;
    const Vec<T> v = *reinterpret_cast<const Vec<T>*>(x + off);
    Vec<T> out;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float f = to_float<T>(v.v[j]) * inv_hc;
      out.v[j] = from_float<T>(f * sigmoidf_(f));
    }
    *reinterpret_cast<Vec<T>*>(y + off) = out;
  }
}

/*!
 * \brief Mix the streams down to one, each gated by its own logit.
 */
// The stream count is a model constant, and a runtime loop bound stops the
// unroll: the loads of one stream then wait on the branch of the last.
template <typename T, int HC>
__global__ void hc_gate_mix_kernel(const T* __restrict__ x,
                                   const T* __restrict__ g, T* __restrict__ y,
                                   int64_t stride_x, int64_t stride_g,
                                   int64_t stride_y, int hc_dim, int rows,
                                   float inv_hc) {
  const int vec = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
  if (vec >= hc_dim) return;
  // The row is a grid dimension, so nothing has to be divided out of a flat
  // index, and no width has to be a power of two.
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const T* xr = x + row * stride_x + vec;
    const T* gr = g + row * stride_g + vec;
    float acc[kVec] = {};
#pragma unroll
    for (int stream = 0; stream < HC; ++stream) {
      const int64_t at = static_cast<int64_t>(stream) * hc_dim;
      const Vec<T> gv = *reinterpret_cast<const Vec<T>*>(gr + at);
      const Vec<T> xv = *reinterpret_cast<const Vec<T>*>(xr + at);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        acc[j] += sigmoidf_(to_float<T>(gv.v[j])) * to_float<T>(xv.v[j]);
      }
    }
    Vec<T> out;
#pragma unroll
    for (int j = 0; j < kVec; ++j) out.v[j] = from_float<T>(acc[j] * inv_hc);
    *reinterpret_cast<Vec<T>*>(y + row * stride_y + vec) = out;
  }
}

/*!
 * \brief Add the block's output back into every stream of the residual.
 *
 * A thread takes one slice of the block's output and writes it into all of the
 * streams. Taking one stream at a time would read that slice once per stream.
 */
template <typename T, int HC>
__global__ void hc_combine_kernel(const T* __restrict__ block,
                                  const T* __restrict__ res,
                                  const T* __restrict__ inj,
                                  T* __restrict__ out, int64_t stride_block,
                                  int64_t stride_res, int64_t stride_inj,
                                  int64_t stride_out, int hc_dim, int rows,
                                  float inv_hc) {
  const int vec = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
  if (vec >= hc_dim) return;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const Vec<T> bv =
        *reinterpret_cast<const Vec<T>*>(block + row * stride_block + vec);
    const T* rr = res + row * stride_res + vec;
    T* orow = out + row * stride_out + vec;
    const T* injr = inj + row * stride_inj;
#pragma unroll
    for (int stream = 0; stream < HC; ++stream) {
      const float gate = 2.f * sigmoidf_(to_float<T>(injr[stream]) * inv_hc);
      const int64_t at = static_cast<int64_t>(stream) * hc_dim;
      const Vec<T> rv = *reinterpret_cast<const Vec<T>*>(rr + at);
      Vec<T> o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o.v[j] =
            from_float<T>(to_float<T>(rv.v[j]) + to_float<T>(bv.v[j]) * gate);
      }
      *reinterpret_cast<Vec<T>*>(orow + at) = o;
    }
  }
}

int shift_of(int64_t v) {
  if (v <= 0 || (v & (v - 1)) != 0) return -1;
  int sh = 0;
  while ((int64_t{1} << sh) < v) ++sh;
  return sh;
}

/*!
 * \brief The combine, then the RMSNorm of what it produced.
 *
 * One block owns one stream of one row. The combined residual is written out
 * before the norm reads it back, which is where the caller's unfused pair
 * rounds it.
 */
template <typename T, bool W_SHARED, int ROW_VECS>
__global__ void __launch_bounds__(kRowBlock)
    hc_combine_norm_kernel(const T* __restrict__ block,
                           const T* __restrict__ res, const T* __restrict__ inj,
                           const T* __restrict__ w, T* __restrict__ out,
                           T* __restrict__ y, int64_t stride_block,
                           int64_t stride_res, int64_t stride_inj,
                           int64_t stride_out, int64_t stride_y, int hc_dim,
                           int hc, int64_t units, float inv_hc, float eps) {
  const int lane = threadIdx.x % kWarp;
  const int64_t unit =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x / kWarp;
  if (unit >= units) return;
  const int stream = static_cast<int>(unit % hc);
  const int64_t row = unit / hc;
  const int64_t base = static_cast<int64_t>(stream) * hc_dim;
  const T* bsrc = block + row * stride_block;
  const T* rsrc = res + row * stride_res + base;
  T* odst = out + row * stride_out + base;
  T* ydst = y + row * stride_y + base;
  const T* weight = w + (W_SHARED ? 0 : base);
  const float gate =
      2.f * sigmoidf_(to_float<T>(inj[row * stride_inj + stream]) * inv_hc);

  float sum = 0.f;
  Vec<T> held[ROW_VECS > 0 ? ROW_VECS : 1];
  // Rounded on the way out, as the unfused combine would leave it, so the norm
  // sees the same values either way.
  if constexpr (ROW_VECS > 0) {
#pragma unroll
    for (int r = 0; r < ROW_VECS; ++r) {
      const int i = (lane + r * kWarp) * kVec;
      const Vec<T> bv = *reinterpret_cast<const Vec<T>*>(bsrc + i);
      const Vec<T> rv = *reinterpret_cast<const Vec<T>*>(rsrc + i);
      Vec<T> o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o.v[j] =
            from_float<T>(to_float<T>(rv.v[j]) + to_float<T>(bv.v[j]) * gate);
        const float f = to_float<T>(o.v[j]);
        sum += f * f;
      }
      held[r] = o;
      *reinterpret_cast<Vec<T>*>(odst + i) = o;
    }
  } else {
    for (int i = lane * kVec; i < hc_dim; i += kWarp * kVec) {
      const Vec<T> bv = *reinterpret_cast<const Vec<T>*>(bsrc + i);
      const Vec<T> rv = *reinterpret_cast<const Vec<T>*>(rsrc + i);
      Vec<T> o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o.v[j] =
            from_float<T>(to_float<T>(rv.v[j]) + to_float<T>(bv.v[j]) * gate);
        const float f = to_float<T>(o.v[j]);
        sum += f * f;
      }
      *reinterpret_cast<Vec<T>*>(odst + i) = o;
    }
  }
  const float rrms = rsqrtf(warp_sum(sum) / static_cast<float>(hc_dim) + eps);

  if constexpr (ROW_VECS > 0) {
#pragma unroll
    for (int r = 0; r < ROW_VECS; ++r) {
      const int i = (lane + r * kWarp) * kVec;
      const Vec<T> g = *reinterpret_cast<const Vec<T>*>(weight + i);
      Vec<T> yv;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float sc = to_float<T>(held[r].v[j]) * rrms;
        yv.v[j] = from_float<T>(sc + sc * to_float<T>(g.v[j]));
      }
      *reinterpret_cast<Vec<T>*>(ydst + i) = yv;
    }
  } else {
    for (int i = lane * kVec; i < hc_dim; i += kWarp * kVec) {
      const Vec<T> o = *reinterpret_cast<const Vec<T>*>(odst + i);
      const Vec<T> g = *reinterpret_cast<const Vec<T>*>(weight + i);
      Vec<T> yv;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float sc = to_float<T>(o.v[j]) * rrms;
        yv.v[j] = from_float<T>(sc + sc * to_float<T>(g.v[j]));
      }
      *reinterpret_cast<Vec<T>*>(ydst + i) = yv;
    }
  }
}

int elementwise_grid(int64_t work, int block) {
  static thread_local int num_sms = 0;
  if (num_sms == 0) {
    int dev = 0;
    STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
    STD_TORCH_CHECK(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev) ==
            cudaSuccess,
        "cudaDeviceGetAttribute failed");
  }
  // Enough blocks to cover the device several times over, and never more than
  // there is work for.
  const int64_t want = (work + block - 1) / block;
  const int64_t cap = static_cast<int64_t>(num_sms) * 8;
  return static_cast<int>(want < cap ? (want > 0 ? want : 1) : cap);
}

}  // namespace

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace {

void check_hc_dtype(const Tensor& t, ScalarType want, const char* name) {
  STD_TORCH_CHECK(t.scalar_type() == want, name, " has the wrong dtype");
}

// The stream counts a Qwen4Exp checkpoint uses. Anything else keeps the
// caller's other path rather than paying a runtime loop here.
#define QWEN4EXP_HC_STREAMS(hc, go)                                       \
  switch (hc) {                                                           \
    case 2:                                                               \
      (go)(std::integral_constant<int, 2>{});                             \
      break;                                                              \
    case 4:                                                               \
      (go)(std::integral_constant<int, 4>{});                             \
      break;                                                              \
    case 8:                                                               \
      (go)(std::integral_constant<int, 8>{});                             \
      break;                                                              \
    default:                                                              \
      STD_TORCH_CHECK(false,                                              \
                      "Qwen4Exp HC ops take two, four or eight streams"); \
  }

// Row widths, in per-lane vectors, that the held-row form is built for. Wider
// than four and the held row spills, which costs more than the read it saves,
// so those walk the row at runtime instead. The form is also only taken when
// the launch is small: holding costs registers, and a launch that fills the
// device would rather have the warps.
#define QWEN4EXP_HC_ROW(row_dim, small, fn)      \
  switch (!(small) || (row_dim) % (kWarp * kVec) \
              ? -1                               \
              : (row_dim) / (kWarp * kVec)) {    \
    case 1:                                      \
      fn(std::integral_constant<int, 1>{});      \
      break;                                     \
    case 2:                                      \
      fn(std::integral_constant<int, 2>{});      \
      break;                                     \
    case 4:                                      \
      fn(std::integral_constant<int, 4>{});      \
      break;                                     \
    default:                                     \
      fn(std::integral_constant<int, 0>{});      \
  }

// A launch small enough that a row's registers are cheaper than more warps.
inline bool hc_small_launch(int64_t units) {
  static thread_local int num_sms = 0;
  if (num_sms == 0) {
    int dev = 0;
    STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
    STD_TORCH_CHECK(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev) ==
            cudaSuccess,
        "cudaDeviceGetAttribute failed");
  }
  return units < static_cast<int64_t>(num_sms) * kRowsPerBlock;
}

#define QWEN4EXP_HC_DISPATCH(dtype, ...)                                \
  if ((dtype) == ScalarType::BFloat16) {                                \
    using scalar_t = __nv_bfloat16;                                     \
    __VA_ARGS__;                                                        \
  } else if ((dtype) == ScalarType::Half) {                             \
    using scalar_t = __half;                                            \
    __VA_ARGS__;                                                        \
  } else {                                                              \
    STD_TORCH_CHECK(false, "Qwen4Exp HC ops take float16 or bfloat16"); \
  }

}  // namespace

void qwen4_exp_grouped_gemma_rmsnorm(const Tensor& x, const Tensor& weight,
                                     double eps, int64_t num_groups,
                                     Tensor& y) {
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  STD_TORCH_CHECK(num_groups > 0 && dim % num_groups == 0,
                  "the row must divide into the groups");
  const int64_t group_dim = dim / num_groups;
  STD_TORCH_CHECK(group_dim % kVec == 0, "a group must be a multiple of ",
                  kVec);
  STD_TORCH_CHECK(x.stride(1) == 1 && y.stride(1) == 1,
                  "rows must be contiguous");
  check_hc_dtype(weight, x.scalar_type(), "weight");
  check_hc_dtype(y, x.scalar_type(), "y");
  const bool shared = weight.numel() == group_dim;
  STD_TORCH_CHECK(shared || weight.numel() == dim,
                  "the weight is a group or a row wide");
  if (rows == 0) return;

  const int64_t units = rows * num_groups;
  const dim3 grid(
      static_cast<unsigned>((units + kRowsPerBlock - 1) / kRowsPerBlock));
  QWEN4EXP_HC_DISPATCH(x.scalar_type(), {
    auto launch_one = [&](auto shared_tag, auto row_tag) {
      grouped_gemma_rmsnorm_kernel<scalar_t, decltype(shared_tag)::value,
                                   decltype(row_tag)::value>
          <<<grid, kRowBlock, 0, 0>>>(
              reinterpret_cast<const scalar_t*>(x.data_ptr()),
              reinterpret_cast<const scalar_t*>(weight.data_ptr()),
              reinterpret_cast<scalar_t*>(y.data_ptr()), x.stride(0),
              y.stride(0), static_cast<int>(group_dim),
              static_cast<int>(num_groups), units, static_cast<float>(eps));
    };
    if (shared) {
      auto fn = [&](auto row_tag) { launch_one(std::true_type{}, row_tag); };
      QWEN4EXP_HC_ROW(group_dim, hc_small_launch(units), fn);
    } else {
      auto fn = [&](auto row_tag) { launch_one(std::false_type{}, row_tag); };
      QWEN4EXP_HC_ROW(group_dim, hc_small_launch(units), fn);
    }
  });
}

void qwen4_exp_hc_silu(const Tensor& x, int64_t hc_count, Tensor& y) {
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  STD_TORCH_CHECK(hc_count > 0, "hc_count must be positive");
  STD_TORCH_CHECK(dim % kVec == 0, "the row must be a multiple of ", kVec);
  STD_TORCH_CHECK(x.stride(1) == 1 && y.stride(1) == 1,
                  "rows must be contiguous");
  check_hc_dtype(y, x.scalar_type(), "y");
  if (rows == 0) return;

  // The flat walk below needs the rows packed; a padded row is rare enough to
  // leave to the caller's other path.
  STD_TORCH_CHECK(x.stride(0) == dim && y.stride(0) == dim,
                  "rows must be packed");
  const int64_t total = rows * (dim / kVec);
  const int grid = elementwise_grid(total, kFlatBlock);
  QWEN4EXP_HC_DISPATCH(x.scalar_type(), {
    hc_silu_kernel<scalar_t><<<grid, kFlatBlock, 0, 0>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<scalar_t*>(y.data_ptr()), total,
        1.f / static_cast<float>(hc_count));
  });
}

void qwen4_exp_hc_gate_mix(const Tensor& x, const Tensor& gate,
                           int64_t hc_count, Tensor& y) {
  const int64_t rows = gate.size(0);
  const int64_t dim = gate.size(1);
  STD_TORCH_CHECK(hc_count > 0 && dim % hc_count == 0,
                  "the row must divide into streams");
  const int64_t hc_dim = dim / hc_count;
  STD_TORCH_CHECK(hc_dim % kVec == 0, "a stream must be a multiple of ", kVec);
  STD_TORCH_CHECK(x.stride(1) == 1 && gate.stride(1) == 1 && y.stride(1) == 1,
                  "rows must be contiguous");
  check_hc_dtype(gate, x.scalar_type(), "gate");
  check_hc_dtype(y, x.scalar_type(), "y");
  if (rows == 0) return;

  const dim3 grid(static_cast<unsigned>((hc_dim / kVec + kBlock - 1) / kBlock),
                  static_cast<unsigned>(rows < 65535 ? rows : 65535));
  QWEN4EXP_HC_DISPATCH(x.scalar_type(), {
    auto go = [&](auto hc_tag) {
      hc_gate_mix_kernel<scalar_t, decltype(hc_tag)::value>
          <<<grid, kBlock, 0, 0>>>(
              reinterpret_cast<const scalar_t*>(x.data_ptr()),
              reinterpret_cast<const scalar_t*>(gate.data_ptr()),
              reinterpret_cast<scalar_t*>(y.data_ptr()), x.stride(0),
              gate.stride(0), y.stride(0), static_cast<int>(hc_dim),
              static_cast<int>(rows), 1.f / static_cast<float>(hc_count));
    };
    QWEN4EXP_HC_STREAMS(hc_count, go);
  });
}

void qwen4_exp_hc_combine(const Tensor& residual, const Tensor& block_output,
                          const Tensor& injection_logits, int64_t hc_count,
                          Tensor& out) {
  const int64_t rows = residual.size(0);
  const int64_t dim = residual.size(1);
  STD_TORCH_CHECK(hc_count > 0 && dim % hc_count == 0,
                  "the row must divide into streams");
  const int64_t hc_dim = dim / hc_count;
  STD_TORCH_CHECK(hc_dim % kVec == 0, "a stream must be a multiple of ", kVec);
  STD_TORCH_CHECK(residual.stride(1) == 1 && block_output.stride(1) == 1 &&
                      injection_logits.stride(1) == 1 && out.stride(1) == 1,
                  "rows must be contiguous");
  check_hc_dtype(block_output, residual.scalar_type(), "block_output");
  check_hc_dtype(injection_logits, residual.scalar_type(), "injection_logits");
  check_hc_dtype(out, residual.scalar_type(), "out");
  if (rows == 0) return;

  const dim3 grid(static_cast<unsigned>((hc_dim / kVec + kBlock - 1) / kBlock),
                  static_cast<unsigned>(rows < 65535 ? rows : 65535));
  QWEN4EXP_HC_DISPATCH(residual.scalar_type(), {
    auto go = [&](auto hc_tag) {
      hc_combine_kernel<scalar_t, decltype(hc_tag)::value>
          <<<grid, kBlock, 0, 0>>>(
              reinterpret_cast<const scalar_t*>(block_output.data_ptr()),
              reinterpret_cast<const scalar_t*>(residual.data_ptr()),
              reinterpret_cast<const scalar_t*>(injection_logits.data_ptr()),
              reinterpret_cast<scalar_t*>(out.data_ptr()),
              block_output.stride(0), residual.stride(0),
              injection_logits.stride(0), out.stride(0),
              static_cast<int>(hc_dim), static_cast<int>(rows),
              1.f / static_cast<float>(hc_count));
    };
    QWEN4EXP_HC_STREAMS(hc_count, go);
  });
}

void qwen4_exp_hc_combine_norm(const Tensor& residual,
                               const Tensor& block_output,
                               const Tensor& injection_logits,
                               const Tensor& norm_weight, double eps,
                               int64_t hc_count, Tensor& out, Tensor& y) {
  const int64_t rows = residual.size(0);
  const int64_t dim = residual.size(1);
  STD_TORCH_CHECK(hc_count > 0 && dim % hc_count == 0,
                  "the row must divide into streams");
  const int64_t hc_dim = dim / hc_count;
  STD_TORCH_CHECK(hc_dim % kVec == 0, "a stream must be a multiple of ", kVec);
  STD_TORCH_CHECK(residual.stride(1) == 1 && block_output.stride(1) == 1 &&
                      injection_logits.stride(1) == 1 && out.stride(1) == 1 &&
                      y.stride(1) == 1,
                  "rows must be contiguous");
  check_hc_dtype(block_output, residual.scalar_type(), "block_output");
  check_hc_dtype(injection_logits, residual.scalar_type(), "injection_logits");
  check_hc_dtype(norm_weight, residual.scalar_type(), "norm_weight");
  check_hc_dtype(out, residual.scalar_type(), "out");
  check_hc_dtype(y, residual.scalar_type(), "y");
  const bool shared = norm_weight.numel() == hc_dim;
  STD_TORCH_CHECK(shared || norm_weight.numel() == dim,
                  "the weight is a stream or a row wide");
  if (rows == 0) return;

  const int64_t units = rows * hc_count;
  const dim3 grid(
      static_cast<unsigned>((units + kRowsPerBlock - 1) / kRowsPerBlock));
  QWEN4EXP_HC_DISPATCH(residual.scalar_type(), {
    auto launch_one = [&](auto shared_tag, auto row_tag) {
      hc_combine_norm_kernel<scalar_t, decltype(shared_tag)::value,
                             decltype(row_tag)::value>
          <<<grid, kRowBlock, 0, 0>>>(
              reinterpret_cast<const scalar_t*>(block_output.data_ptr()),
              reinterpret_cast<const scalar_t*>(residual.data_ptr()),
              reinterpret_cast<const scalar_t*>(injection_logits.data_ptr()),
              reinterpret_cast<const scalar_t*>(norm_weight.data_ptr()),
              reinterpret_cast<scalar_t*>(out.data_ptr()),
              reinterpret_cast<scalar_t*>(y.data_ptr()), block_output.stride(0),
              residual.stride(0), injection_logits.stride(0), out.stride(0),
              y.stride(0), static_cast<int>(hc_dim), static_cast<int>(hc_count),
              units, 1.f / static_cast<float>(hc_count),
              static_cast<float>(eps));
    };
    if (shared) {
      auto fn = [&](auto row_tag) { launch_one(std::true_type{}, row_tag); };
      QWEN4EXP_HC_ROW(hc_dim, hc_small_launch(units), fn);
    } else {
      auto fn = [&](auto row_tag) { launch_one(std::false_type{}, row_tag); };
      QWEN4EXP_HC_ROW(hc_dim, hc_small_launch(units), fn);
    }
  });
}
