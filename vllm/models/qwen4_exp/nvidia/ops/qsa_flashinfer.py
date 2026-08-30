# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer block-sparse backend for the Qwen4Exp QSA attention step.

The QSA top-k route is a per-token list of logical KV positions padded with
``-1``. FlashInfer's block-sparse prefill wrapper takes a BSR route of physical
slots plus a packed validity mask, so this module owns the translation:

    logical route (-1 padded)
      -> validity: in-range logical page, mapped page in the cache
      -> physical slot for the valid entries, slot 0 for the rest
      -> packed little-endian mask, ceil(width/8) bytes per row

The route names entries of the cache as it is allocated, so no layout change
is needed: an NVFP4 QSA cache stays HND and an unquantized one stays NHD.

Geometry is fixed at ``plan()`` time and cached per row bucket; only the route
and mask contents change per forward, written in place into the buffers the
wrapper already holds. Everything here runs inside the QSA custom op, so the
ATen sequence never reaches Inductor and never becomes Triton.
"""

from __future__ import annotations

import functools
import inspect
from collections import OrderedDict

import torch

from vllm.platforms import current_platform

# The software E2M1 decode this path relies on is a pre-SM100 construct: from
# SM100 the conversion is a single instruction and a different specialization
# applies. See FlashInfer include/flashinfer/attention/prefill.cuh.
_NATIVE_FP4_CAPABILITY = 100
# Row buckets keep the number of distinct plans bounded: a step pads to the
# next bucket and marks the padding rows invalid.
_ROW_BUCKET = 128
# Each retained plan owns its own wrapper workspaces, so the cache has to be
# bounded or a long-running server accumulates them per geometry visited.
_MAX_PLANS = 8


@functools.cache
def _sparse_wrapper_features() -> frozenset[str]:
    """Which of the features this backend needs the installed FlashInfer has.

    ``paged`` is a route over the cache as allocated; ``nvfp4`` is the packed
    E2M1 path. Older releases serve neither, and reaching for one that is
    missing would either raise or silently read the wrong bytes.
    """
    try:
        from flashinfer import BlockSparseAttentionWrapper

        run_params = inspect.signature(BlockSparseAttentionWrapper.run).parameters
        plan_params = inspect.signature(BlockSparseAttentionWrapper.plan).parameters
        init_params = inspect.signature(BlockSparseAttentionWrapper.__init__).parameters
    except Exception:
        # A missing wrapper, a C-level signature, or a partially initialized
        # module all mean the same thing here: this backend cannot be used.
        return frozenset()
    found = {"sparse"}
    if "kv_cache_sf" in run_params:
        found.add("nvfp4")
    if "kv_cache_page_size" in plan_params and "kv_layout" in init_params:
        found.add("paged")
    return frozenset(found)


def supports_qsa_flashinfer(head_dim: int, kv_cache_dtype: str) -> bool:
    """Gate: pre-SM100 CUDA, a shape the FA2 sparse kernel serves, FlashInfer.

    NVFP4 additionally needs a FlashInfer whose block-sparse wrapper accepts
    ``kv_cache_sf``; older releases only serve the unpacked path.
    """
    if not current_platform.is_cuda():
        return False
    if current_platform.has_device_capability(_NATIVE_FP4_CAPABILITY):
        return False
    if head_dim > 256 or head_dim % 16:
        return False
    features = _sparse_wrapper_features()
    if "paged" not in features:
        return False
    if kv_cache_dtype == "nvfp4":
        return "nvfp4" in features
    # Per-tensor FP8 has no block-sparse path yet; Triton keeps serving it.
    return kv_cache_dtype in ("auto", "bfloat16")


@functools.cache
def _workspace_for(device_index: int, nbytes: int) -> torch.Tensor:
    """One workspace per device, shared by every plan and every QSA layer.

    Plans run sequentially inside a forward, so they can share it; giving each
    layer its own would cost nbytes * layer_count of KV-cache headroom.
    """
    return torch.empty(
        nbytes, dtype=torch.uint8, device=torch.device("cuda", device_index)
    )


class _RoutePlan:
    """One wrapper plus its caller-owned route and mask buffers, per geometry."""

    def __init__(
        self,
        rows,
        width,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        num_slots,
        page_size,
        layout,
        kv_dtype,
        device,
        workspace,
    ):
        import flashinfer

        self.rows = rows
        self.width = width
        self.nbytes = -(-width // 8)
        # plan() validates indices.max(), so the buffer must hold a legal route
        # before the first stage() call fills it.
        self.route = torch.zeros(rows * width, dtype=torch.int32, device=device)
        self.valid = torch.empty(rows, width, dtype=torch.bool, device=device)
        self._log = torch.empty(rows, width, dtype=torch.int64, device=device)
        self._page = torch.empty(rows, width, dtype=torch.int64, device=device)
        self._phys = torch.empty(rows, width, dtype=torch.int64, device=device)
        self._bits = torch.empty(rows, self.nbytes, 8, dtype=torch.int32, device=device)
        self._sum = torch.empty(rows, self.nbytes, dtype=torch.int32, device=device)
        self._weights = 1 << torch.arange(8, dtype=torch.int32, device=device)
        self._packed = torch.empty(rows * self.nbytes, dtype=torch.uint8, device=device)
        self._table = torch.empty(rows, 1, dtype=torch.int64, device=device)
        self._num_slots = num_slots

        indptr = torch.arange(
            0, (rows + 1) * width, width, dtype=torch.int32, device=device
        )
        self.wrapper = flashinfer.BlockSparseAttentionWrapper(
            workspace, kv_layout=layout
        )
        # An all-true mask only fixes the geometry; contents are replaced below.
        self.wrapper.plan(
            indptr,
            self.route,
            rows,
            num_slots,
            1,
            1,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
            o_data_type=torch.bfloat16,
            # The route addresses KV entries of a cache that stores whole
            # pages, so the wrapper has to divide each index back into
            # (page, entry) instead of treating it as a block id.
            kv_cache_page_size=page_size,
        )
        # plan() may or may not alias the caller's tensor; bind whatever it kept
        # so stage() always writes the buffer run() reads.
        self._route_buf = self.wrapper._paged_kv_indices_buf
        self._mask_buf = self.wrapper._packed_mask_buf

    def stage(self, logical, block_table, token_to_req, page_size):
        """Write this step's route and mask into the wrapper's buffers."""
        rows_in = logical.shape[0]
        table_width = block_table.shape[1]
        self._log.fill_(-1)
        self._log[:rows_in].copy_(logical)

        # Validity folds every bound the Triton kernel checks: a negative
        # sentinel, a logical page past the block table, an unmapped page, and
        # a physical slot outside the cache. Padding rows stay false because
        # their logical entries are -1.
        valid = self.valid
        torch.ge(self._log, 0, out=valid)
        safe = torch.clamp_min(self._log, 0, out=self._phys)
        torch.div(safe, page_size, rounding_mode="floor", out=self._page)
        valid &= self._page < table_width
        self._page.clamp_(max=table_width - 1)

        rows_req = self._table[:rows_in]
        rows_req.copy_(token_to_req.long().unsqueeze(1))
        table = block_table.long()[rows_req.squeeze(1)]
        if rows_in < self.rows:
            table = torch.cat(
                [table, table.new_zeros(self.rows - rows_in, table_width)], dim=0
            )
        page = torch.gather(table, 1, self._page, out=self._page)
        valid &= page >= 0
        page.clamp_(min=0).mul_(page_size)
        phys = safe.remainder_(page_size).add_(page)
        valid &= phys < self._num_slots
        phys.mul_(valid)
        self._route_buf.view(self.rows, self.width).copy_(phys)

        self._bits.zero_()
        self._bits.view(self.rows, -1)[:, : self.width].copy_(valid)
        self._bits.mul_(self._weights)
        torch.sum(self._bits, -1, dtype=torch.int32, out=self._sum)
        self._mask_buf.view(self.rows, self.nbytes).copy_(self._sum)


class QSAFlashInferRunner:
    """Owns the bounded plan cache for one attention layer."""

    def __init__(self, device: torch.device, workspace_bytes: int = 256 << 20):
        self._device = device
        self._workspace = _workspace_for(device.index or 0, workspace_bytes)
        self._plans: OrderedDict[tuple, _RoutePlan] = OrderedDict()

    def _plan_for(self, key, *args):
        plan = self._plans.get(key)
        if plan is not None:
            self._plans.move_to_end(key)
            return plan
        plan = _RoutePlan(*args)
        self._plans[key] = plan
        while len(self._plans) > _MAX_PLANS:
            self._plans.popitem(last=False)
        return plan

    def run(
        self,
        query,
        k_data,
        v_data,
        k_sf,
        v_sf,
        logical_indices,
        block_table,
        token_to_req,
        out,
        page_size,
        layout,
        k_scale,
        v_scale,
    ):
        rows_in, num_qo_heads, head_dim = query.shape
        width = logical_indices.shape[1]
        rows = -(-rows_in // _ROW_BUCKET) * _ROW_BUCKET
        num_kv_heads = k_data.shape[1 if layout == "HND" else 2]
        num_slots = k_data.shape[0] * page_size
        key = (
            rows,
            width,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            num_slots,
            page_size,
            layout,
            k_data.dtype,
        )
        plan = self._plan_for(
            key,
            rows,
            width,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            num_slots,
            page_size,
            layout,
            k_data.dtype,
            self._device,
            self._workspace,
        )
        plan.stage(logical_indices, block_table, token_to_req, page_size)

        q = query
        if rows > rows_in:
            q = torch.zeros(
                rows, num_qo_heads, head_dim, dtype=query.dtype, device=query.device
            )
            q[:rows_in].copy_(query)
        kwargs = {}
        if k_sf is not None:
            kwargs = {
                "kv_cache_sf": (k_sf, v_sf),
                "k_scale": k_scale,
                "v_scale": v_scale,
            }
        result = plan.wrapper.run(q, k_data, v_data, **kwargs)
        out.copy_(result[:rows_in])
        return out
