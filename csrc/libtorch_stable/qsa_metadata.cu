// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-step metadata for the Qwen4Exp QSA side cache.
//
// One pass turns the scheduler's per-request view into the per-token one the
// QSA kernels read: which request a token belongs to, where it sits in that
// request's sequence, and which physical slot of the side cache it writes. A
// second, independent tile of the same launch builds the pre-indexer's work
// list. Doing this as tensor ops costs several hundred microseconds a step;
// the values are small and the arithmetic is trivial, so the cost is launches.

#include <cuda_runtime.h>

#include <algorithm>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "core/registration.h"
#include "cuda_utils.h"
#include "ops.h"

namespace {

// One block covers a tile of tokens and a tile of work items, so a step is one
// launch. The width is set by the request prefix the work half scans: every
// block rebuilds that prefix rather than reading one another block wrote, which
// is what lets the tiles stay independent without a grid-wide barrier. A step
// with more requests than this width scans them a window at a time, carrying
// the running total across windows.
constexpr int kRequestScan = 128;
constexpr int kTokenBlock = kRequestScan;
constexpr int kWorkBlock = kRequestScan;

__device__ __forceinline__ int32_t last_start_at_or_before(
    const int32_t* __restrict__ query_start_loc, int32_t token,
    int32_t num_reqs, int32_t num_search_steps, int32_t sentinel) {
  int32_t request = 0;
#pragma unroll 1
  for (int32_t step = 0; step < num_search_steps; ++step) {
    const int32_t candidate = request + (1 << (num_search_steps - step - 1));
    const bool in_range = candidate < num_reqs;
    const int32_t start = in_range ? query_start_loc[candidate] : sentinel;
    if (in_range && start <= token) request = candidate;
  }
  return request;
}

/*!
 * \brief Build the pre-indexer's work list: one item per compression group.
 *
 * A non-empty request needs an item even without a completed group, because
 * item zero also commits that request's raw-key suffix.
 */
__device__ __forceinline__ void build_work_tile(
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens, int32_t* __restrict__ work_metadata,
    int32_t num_reqs, int32_t max_num_work, int32_t compress_ratio,
    int32_t ratio_shift, int32_t block) {
  // Every work block rebuilds the request prefix rather than reading one a
  // previous block wrote: the prefix is a few kilobytes and recomputing it
  // keeps the blocks independent, so they need no grid-wide barrier.
  // A block past the last work tile has nothing to name, and the scan below is
  // the expensive half of this kernel.
  if (block * kWorkBlock >= max_num_work) return;

  __shared__ int32_t prefix[kRequestScan];
  __shared__ int32_t warp_totals[kRequestScan / 32];

  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = lane >> 5;
  const int32_t in_warp = lane & 31;
  const int32_t work = block * kWorkBlock + lane;

  // Walk the requests a window at a time so a step with more of them than this
  // block is wide still resolves, carrying the running total across windows.
  int32_t carry = 0;
  int32_t request = -1;
  int32_t start = 0;
  for (int32_t base = 0; base < num_reqs; base += kRequestScan) {
    const int32_t r = base + lane;
    int32_t count = 0;
    if (r < num_reqs) {
      const int32_t query_len = query_start_loc[r + 1] - query_start_loc[r];
      const int32_t seq_len = seq_lens[r];
      const int32_t chunk_start = seq_len - query_len;
      // A power-of-two ratio is the only one this model uses, and an integer
      // divide here costs more than the rest of the scan.
      const int32_t groups =
          ratio_shift >= 0
              ? (seq_len >> ratio_shift) - (chunk_start >> ratio_shift)
              : seq_len / compress_ratio - chunk_start / compress_ratio;
      // A non-empty request needs an item even without a completed group,
      // because item zero also commits its raw-key suffix.
      count = query_len > 0 ? max(groups, 1) : 0;
    }

    // Scan within a warp first, then across the warps' totals. A flat
    // shared-memory scan of this width would cost twenty barriers, and a step
    // has only tens of requests to sum. A step whose requests all fit in one
    // warp skips the cross-warp half, and with it two of the three barriers --
    // that half is what the latency of a one-request step is made of.
    const int32_t window = min(num_reqs - base, kRequestScan);
    int32_t running = count;
    if (window <= 32) {
      if (warp == 0) {
#pragma unroll
        for (int32_t offset = 1; offset < 32; offset <<= 1) {
          const int32_t carried = __shfl_up_sync(0xffffffffu, running, offset);
          if (in_warp >= offset) running += carried;
        }
        prefix[in_warp] = running;
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int32_t offset = 1; offset < 32; offset <<= 1) {
        const int32_t carried = __shfl_up_sync(0xffffffffu, running, offset);
        if (in_warp >= offset) running += carried;
      }
      if (in_warp == 31) warp_totals[warp] = running;
      __syncthreads();
      if (warp == 0) {
        int32_t total_running =
            in_warp < kRequestScan / 32 ? warp_totals[in_warp] : 0;
#pragma unroll
        for (int32_t offset = 1; offset < kRequestScan / 32; offset <<= 1) {
          const int32_t carried =
              __shfl_up_sync(0xffffffffu, total_running, offset);
          if (in_warp >= offset) total_running += carried;
        }
        if (in_warp < kRequestScan / 32) warp_totals[in_warp] = total_running;
      }
      __syncthreads();
      if (warp > 0) running += warp_totals[warp - 1];
      prefix[lane] = running;
      __syncthreads();
    }

    const int32_t window_total = prefix[window - 1];
    if (request < 0 && work < carry + window_total) {
      // The first request in this window whose running total passes the item.
      // The search walks only as far as the window reaches.
      int32_t span = 1;
      while (span < window) span <<= 1;
      int32_t found = 0;
      for (int32_t step = span >> 1; step > 0; step >>= 1) {
        const int32_t candidate = found + step;
        if (candidate < window && carry + prefix[candidate - 1] <= work) {
          found = candidate;
        }
      }
      request = base + found;
      start = carry + (found > 0 ? prefix[found - 1] : 0);
    }
    carry += window_total;
    __syncthreads();
  }

  if (work >= max_num_work) return;
  if (request < 0) {
    // Past the real work. Both halves carry the sentinel: a consumer that reads
    // the offset without checking the request has to see something inert.
    work_metadata[2 * work] = -1;
    work_metadata[2 * work + 1] = -1;
    return;
  }
  work_metadata[2 * work] = request;
  work_metadata[2 * work + 1] = work - start;
}

/*!
 * \brief Map each scheduled token to its request, position and cache slot.
 *
 * \tparam CIRCULAR whether the side cache is a per-request ring; the other
 *   owner compresses instead, and each compiles the other's rule away
 */
template <bool CIRCULAR, bool POW2>
__global__ void qsa_metadata_kernel(
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ common_slot_mapping,
    const int32_t* __restrict__ block_table, int32_t* __restrict__ token_to_req,
    int64_t* __restrict__ logical_positions, int64_t* __restrict__ slot_mapping,
    int32_t block_table_stride_0, int32_t block_table_stride_1,
    int32_t num_reqs, int32_t num_mapped_tokens, int32_t num_tokens,
    int32_t num_search_steps, int32_t storage_block_size,
    int32_t compress_ratio, int32_t circular_buffer_size,
    int32_t num_block_table_columns, int32_t* __restrict__ work_metadata,
    int32_t max_num_work, int32_t work_blocks, int32_t ratio_shift,
    int32_t block_shift, int32_t ring_shift) {
  // The work tiles get blocks of their own rather than riding along with the
  // token mapping. Both start from the same request table, so sharing a block
  // would put one latency chain behind the other, and a small step has only one
  // block of each.
  if (static_cast<int32_t>(blockIdx.x) < work_blocks) {
    build_work_tile(query_start_loc, seq_lens, work_metadata, num_reqs,
                    max_num_work, compress_ratio, POW2 ? ratio_shift : -1,
                    static_cast<int32_t>(blockIdx.x));
    return;
  }
  const int32_t token =
      (static_cast<int32_t>(blockIdx.x) - work_blocks) * kTokenBlock +
      threadIdx.x;
  if (token >= num_tokens) return;

  const bool mapped = token < num_mapped_tokens;
  const int32_t search_token = min(token, num_mapped_tokens - 1);
  const int32_t request =
      last_start_at_or_before(query_start_loc, search_token, num_reqs,
                              num_search_steps, num_mapped_tokens + 1);

  const int32_t query_start = mapped ? query_start_loc[request] : 0;
  const int32_t query_end = mapped ? query_start_loc[request + 1] : 0;
  const int32_t seq_len = mapped ? seq_lens[request] : 0;
  // Where this token sits in the request's sequence, counting the context it
  // did not schedule.
  const int32_t position =
      mapped ? seq_len - (query_end - query_start) + token - query_start : -1;

  token_to_req[token] = mapped ? request : 0;
  logical_positions[token] = static_cast<int64_t>(position);

  int64_t slot = -1;
  if constexpr (CIRCULAR) {
    // A ring keeps the last circular_buffer_size tokens of each request.
    bool valid = mapped && position >= 0 &&
                 token + circular_buffer_size >= query_end &&
                 num_block_table_columns > 0;
    const int32_t page =
        valid ? block_table[request * block_table_stride_0] : -1;
    valid = valid && page >= 0;
    if (valid) {
      const int32_t in_ring = POW2 ? (position & (circular_buffer_size - 1))
                                   : position % circular_buffer_size;
      slot = static_cast<int64_t>(page) * circular_buffer_size + in_ring;
    }
  } else {
    // Only the token that closes a compression group writes the side cache.
    const int32_t clamped = position > 0 ? position : 0;
    const int32_t compressed =
        POW2 ? (clamped >> ratio_shift) : clamped / compress_ratio;
    const int32_t logical_block =
        POW2 ? (compressed >> block_shift) : compressed / storage_block_size;
    const bool closes = POW2 ? ((position + 1) & (compress_ratio - 1)) == 0
                             : (position + 1) % compress_ratio == 0;
    bool valid = mapped && position >= 0 && closes &&
                 logical_block < num_block_table_columns;
    const int32_t page = valid
                             ? block_table[request * block_table_stride_0 +
                                           logical_block * block_table_stride_1]
                             : -1;
    valid = valid && page >= 0;
    // The main cache decides whether this token is stored at all.
    valid = valid && (mapped ? common_slot_mapping[token] : int64_t{-1}) >= 0;
    if (valid) {
      const int32_t in_block = POW2 ? (compressed & (storage_block_size - 1))
                                    : compressed % storage_block_size;
      slot = static_cast<int64_t>(page) * storage_block_size + in_block;
    }
  }
  if (circular_buffer_size > 0 || compress_ratio != 1) {
    slot_mapping[token] = slot;
  } else {
    // Neither owner rewrites the mapping, so the side cache follows the main
    // one token for token.
    slot_mapping[token] = mapped ? common_slot_mapping[token] : int64_t{-1};
  }
}

int32_t search_steps_for(int32_t num_reqs) {
  int32_t steps = 0;
  while ((1 << steps) < num_reqs) ++steps;
  return steps;
}

}  // namespace

using torch::stable::Tensor;

void qsa_build_metadata(const Tensor& query_start_loc, const Tensor& seq_lens,
                        const Tensor& common_slot_mapping,
                        const Tensor& block_table, Tensor& token_to_req,
                        Tensor& logical_positions, Tensor& slot_mapping,
                        const std::optional<Tensor>& work_metadata,
                        int64_t storage_block_size, int64_t compress_ratio,
                        int64_t circular_buffer_size,
                        int64_t num_mapped_tokens) {
  const int64_t num_tokens = token_to_req.size(0);
  const int64_t num_reqs = query_start_loc.size(0) - 1;
  STD_TORCH_CHECK(num_reqs > 0,
                  "qsa_build_metadata needs at least one request");
  STD_TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
  STD_TORCH_CHECK(storage_block_size > 0,
                  "storage_block_size must be positive");
  STD_TORCH_CHECK(logical_positions.size(0) == num_tokens,
                  "logical_positions must cover every token");
  STD_TORCH_CHECK(slot_mapping.size(0) == num_tokens,
                  "slot_mapping must cover every token");
  // The kernel reads and writes these through typed pointers, so a caller that
  // hands over the wrong width would corrupt them without complaint.
  using torch::headeronly::ScalarType;
  auto check_dtype = [](const Tensor& t, ScalarType want, const char* name) {
    STD_TORCH_CHECK(t.scalar_type() == want, name, " has the wrong dtype");
  };
  check_dtype(query_start_loc, ScalarType::Int, "query_start_loc");
  check_dtype(seq_lens, ScalarType::Int, "seq_lens");
  check_dtype(common_slot_mapping, ScalarType::Long, "common_slot_mapping");
  check_dtype(block_table, ScalarType::Int, "block_table");
  check_dtype(token_to_req, ScalarType::Int, "token_to_req");
  check_dtype(logical_positions, ScalarType::Long, "logical_positions");
  check_dtype(slot_mapping, ScalarType::Long, "slot_mapping");
  if (work_metadata.has_value()) {
    check_dtype(*work_metadata, ScalarType::Int, "work_metadata");
    STD_TORCH_CHECK(work_metadata->size(1) == 2,
                    "work_metadata must be [n, 2]");
  }

  const cudaStream_t stream = 0;
  const int32_t search_steps = search_steps_for(static_cast<int32_t>(num_reqs));
  const int64_t max_num_work =
      work_metadata.has_value() ? work_metadata->size(0) : 0;
  const int64_t token_blocks = (num_tokens + kTokenBlock - 1) / kTokenBlock;
  const int64_t work_blocks = (max_num_work + kWorkBlock - 1) / kWorkBlock;
  const dim3 grid(static_cast<unsigned>(token_blocks + work_blocks));

  auto shift_of = [](int64_t v) -> int32_t {
    if (v <= 0 || (v & (v - 1)) != 0) return -1;
    int32_t s = 0;
    while ((int64_t{1} << s) < v) ++s;
    return s;
  };
  const int32_t ratio_shift = shift_of(compress_ratio);
  const int32_t block_shift = shift_of(storage_block_size);
  const int32_t ring_shift = shift_of(circular_buffer_size);
  // Every divisor this model uses is a power of two; the general form is kept
  // so a configuration that is not still runs.
  const bool pow2 = ratio_shift >= 0 && block_shift >= 0 &&
                    (circular_buffer_size <= 0 || ring_shift >= 0);

  auto launch = [&](auto circular_tag, auto pow2_tag) {
    constexpr bool CIRCULAR = decltype(circular_tag)::value;
    constexpr bool POW2 = decltype(pow2_tag)::value;
    qsa_metadata_kernel<CIRCULAR, POW2><<<grid, kTokenBlock, 0, stream>>>(
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<const int32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const int64_t*>(common_slot_mapping.data_ptr()),
        reinterpret_cast<const int32_t*>(block_table.data_ptr()),
        reinterpret_cast<int32_t*>(token_to_req.data_ptr()),
        reinterpret_cast<int64_t*>(logical_positions.data_ptr()),
        reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
        static_cast<int32_t>(block_table.stride(0)),
        static_cast<int32_t>(block_table.stride(1)),
        static_cast<int32_t>(num_reqs), static_cast<int32_t>(num_mapped_tokens),
        static_cast<int32_t>(num_tokens), search_steps,
        static_cast<int32_t>(storage_block_size),
        static_cast<int32_t>(compress_ratio),
        static_cast<int32_t>(circular_buffer_size),
        static_cast<int32_t>(block_table.size(1)),
        work_metadata.has_value()
            ? reinterpret_cast<int32_t*>(work_metadata->data_ptr())
            : nullptr,
        static_cast<int32_t>(max_num_work), static_cast<int32_t>(work_blocks),
        ratio_shift, block_shift, ring_shift);
  };
  if (circular_buffer_size > 0) {
    pow2 ? launch(std::true_type{}, std::true_type{})
         : launch(std::true_type{}, std::false_type{});
  } else {
    pow2 ? launch(std::false_type{}, std::true_type{})
         : launch(std::false_type{}, std::false_type{});
  }
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qsa_build_metadata", TORCH_BOX(&qsa_build_metadata));
}
