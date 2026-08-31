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
    """Whether the FlashInfer block-sparse route serves this configuration.

    The packed NVFP4 path decodes E2M1 in software, which is a pre-SM100
    construct: from SM100 the conversion is one instruction and a different
    specialization applies. Nothing else here is architecture-dependent.
    """
    if not current_platform.is_cuda():
        return False
    if head_dim > 256 or head_dim % 16:
        return False
    features = _sparse_wrapper_features()
    if "paged" not in features:
        return False
    if kv_cache_dtype == "nvfp4":
        if current_platform.has_device_capability(_NATIVE_FP4_CAPABILITY):
            return False
        return "nvfp4" in features
    # FP8 rides the same route, its scale folded per tensor.
    return kv_cache_dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3")


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
        self._num_slots = num_slots
        self._page_size = page_size

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

    def stage(self, logical, block_table, token_to_req):
        """Write this step's route and mask into the wrapper's buffers.

        One kernel turns the logical route into physical slots and packs the
        validity: every bound the Triton kernel checks -- a negative sentinel, a
        logical page past the block table, an unmapped page, a slot outside the
        cache -- clears the same bit, and a cleared entry still holds an in-range
        slot because the kernel reads the slot before the mask applies.
        """
        import flashinfer

        # A shorter step hands over its own rows: the kernel reads the logical
        # route only below the live count and masks the rest of the buffer off.
        flashinfer.qsa_route_from_logical(
            logical,
            token_to_req,
            block_table,
            self._route_buf.view(self.rows, self.width),
            self._mask_buf,
            logical.shape[0],
            self._page_size,
            self._num_slots,
        )


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
        plan.stage(logical_indices, block_table, token_to_req)

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
        elif k_data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # QSA quantizes FP8 per tensor, which folds outside the dots the
            # same way the packed NVFP4 scale does.
            kwargs = {"k_scale": k_scale, "v_scale": v_scale}
        result = plan.wrapper.run(q, k_data, v_data, **kwargs)
        out.copy_(result[:rows_in])
        return out
