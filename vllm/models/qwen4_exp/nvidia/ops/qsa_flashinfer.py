# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp QSA step, served by FlashInfer.

What is here is the join: which view of the KV cache each format is, where the
workspace comes from, and two calls. The step itself -- scoring the compressed
cache, taking the top blocks, expanding them into a token route, mapping that
route through the block table, attending over it and folding in the output
gate -- is FlashInfer's, behind ``QSASelection`` and ``QSAAttention``.

Nothing here computes a route, packs a mask, keeps a plan cache, reaches into a
wrapper's private buffers, or infers a cache format from a dtype. All of that
used to live in this file; it is the library's now, and re-adding one of them
here is how the two implementations drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

if TYPE_CHECKING:
    import flashinfer

logger = init_logger(__name__)

# The software E2M1 decode this route relies on is a pre-SM100 construct.
_NATIVE_FP4_CAPABILITY = 100
# What qsa_cache_kinds has a format for.
_KV_CACHE_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")


def require_qsa_flashinfer(head_dim: int, kv_cache_dtype: str) -> None:
    """Refuse to build a QSA layer this backend cannot serve.

    One function rather than a predicate and a policy: there is nothing to do
    with the answer but proceed or stop. The Triton kernels are still in the
    tree as the reference implementation and this deployment does not serve on
    them -- a missing capability that quietly routed there would change the
    kernel, the memory profile, and the answer to why a step got slower, with
    nothing said.

    From SM100 a packed NVFP4 cache converts in a single instruction and this
    route -- a software decode of the raw bytes -- is the wrong one. See
    FlashInfer ``include/flashinfer/attention/prefill.cuh``.
    """
    if not current_platform.is_cuda():
        raise RuntimeError("Qwen4Exp QSA requires CUDA")
    if current_platform.has_device_capability(_NATIVE_FP4_CAPABILITY):
        raise RuntimeError(
            "Qwen4Exp QSA has no kernel for SM100 and later: this route decodes "
            "a packed NVFP4 cache in software, which those architectures do in "
            "one instruction through a specialization this backend does not "
            "carry. Serving would fall back to Triton, which this deployment "
            "does not do."
        )
    if head_dim % 16:
        raise RuntimeError(
            "Qwen4Exp QSA packs one NVFP4 scale per sixteen values, so "
            f"head_dim has to be a multiple of sixteen, got {head_dim}"
        )
    if kv_cache_dtype not in _KV_CACHE_DTYPES:
        raise RuntimeError(
            f"Qwen4Exp QSA has no cache format for {kv_cache_dtype!r}; it "
            f"serves {sorted(_KV_CACHE_DTYPES)}"
        )
    try:
        from flashinfer import (
            QSA_CAP_ATTENTION_PAGED,
            QSA_CAP_OUTPUT_GATE,
            QSA_CAP_SELECTION,
            qsa_capabilities,
            qsa_capability_names,
        )

        needed = QSA_CAP_SELECTION | QSA_CAP_ATTENTION_PAGED | QSA_CAP_OUTPUT_GATE
        carried = qsa_capabilities(torch.accelerator.current_accelerator())
        if carried & needed == needed:
            return
        names = sorted(qsa_capability_names())
    except ImportError:
        names = []
    raise RuntimeError(
        "Qwen4Exp QSA requires FlashInfer to serve attention: this build "
        f"carries {names}. Install a FlashInfer that carries the selection, "
        "the paged route and the output gate -- this deployment does not fall "
        "back to Triton, whose kernels and memory profile differ."
    )


# The model allocated the cache, so the model says what its bytes are. Two
# formats arrive as raw memory -- e4m3 values and packed NVFP4 -- and a uint8
# tensor cannot say which it is, so the format travels beside the views.


@dataclass(frozen=True)
class QSACacheViews:
    """The cache as FlashInfer takes it: what the bytes are and where."""

    k_data: torch.Tensor
    v_data: torch.Tensor
    k_sf: torch.Tensor | None
    v_sf: torch.Tensor | None
    format: str
    layout: str
    page_size: int


def qsa_cache_kinds(kv_cache_dtype: str) -> tuple[str, str, torch.dtype]:
    """What the cache will be, from the config alone: format, layout, dtype.

    The workspace has to be reserved before the cache is allocated, and the
    planner is instantiated per format. This is the one place that maps a
    config string onto them, so the reservation and the views cannot drift.
    """
    if kv_cache_dtype.startswith("nvfp4"):
        return "nvfp4", "HND", torch.uint8
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return "fp8_e4m3", "NHD", torch.float8_e4m3fn
    return "dense", "NHD", torch.bfloat16


def qsa_cache_views(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_cache_dtype: str,
) -> QSACacheViews:
    """Split the allocation into the planes FlashInfer reads. All views.

    An NVFP4 QSA cache is stored HND as ``(pages, heads, page_size, full)``
    where the tail of ``full`` is the e4m3 block-scale plane; an unquantized or
    e4m3 cache is NHD and has no plane. Nothing is permuted and nothing is
    copied.
    """
    kind, layout, _dtype = qsa_cache_kinds(kv_cache_dtype)
    if kind == "nvfp4":
        _pages, _heads, page_size, full = key_cache.shape
        data = full - full // 9
        return QSACacheViews(
            k_data=key_cache[..., :data],
            v_data=value_cache[..., :data],
            k_sf=key_cache[..., data:].view(torch.float8_e4m3fn),
            v_sf=value_cache[..., data:].view(torch.float8_e4m3fn),
            format=kind,
            layout=layout,
            page_size=page_size,
        )
    _pages, page_size, _heads, _dim = key_cache.shape
    if kind == "fp8_e4m3":
        # The cache is allocated as uint8; the format lives in the type from
        # here on, which is a view away and costs nothing.
        return QSACacheViews(
            k_data=key_cache.view(torch.float8_e4m3fn),
            v_data=value_cache.view(torch.float8_e4m3fn),
            k_sf=None,
            v_sf=None,
            format=kind,
            layout=layout,
            page_size=page_size,
        )
    return QSACacheViews(
        k_data=key_cache,
        v_data=value_cache,
        k_sf=None,
        v_sf=None,
        format=kind,
        layout=layout,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# The runtime
#
# One object, the library's, holding everything the step needs. This side says
# what the deployment is, hands over the bytes the library asked for, and calls
# it: where the float workspace, the plan arena, the selection scratch, the
# padded query and output and the physical route and mask sit inside those
# bytes is not visible from here and must not become so.
# ---------------------------------------------------------------------------


def qsa_config(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_rows: int,
    kv_cache_dtype: str,
    max_columns: int,
    compress_ratio: int,
    token_topk: int,
    index_num_heads: int,
    index_head_dim: int,
) -> flashinfer.QSAConfig:
    """This deployment's QSA, in the library's terms.

    Assembled from the config alone, because it is needed before the cache is
    allocated: the workspace has to be reserved while it can still grow, and
    what it costs is a question about this and nothing else. How many slots the
    cache holds and how many of them a page holds are deliberately absent --
    both are settled when the cache is allocated, which is later, and both
    belong to ``plan_cache``.
    """
    import flashinfer

    kind, layout, kv_dtype = qsa_cache_kinds(kv_cache_dtype)
    return flashinfer.QSAConfig(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_rows=max_rows,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        o_data_type=torch.bfloat16,
        kv_cache_format=kind,
        kv_layout=layout,
        max_columns=max_columns,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )


def qsa_workspace_needs(
    config: flashinfer.QSAConfig, device: torch.device
) -> flashinfer.QSAWorkspaceRequirements:
    """What the library asks for: two byte counts with two lifetimes.

    The first has to be memory nobody else writes, because the plans are read
    back on every call. The second is scratch a call rewrites before it reads,
    so it comes out of the workspace every consumer of a step shares. Neither
    says what is inside it.

    Handed on as the object the library returned rather than as loose numbers:
    the alignment travels with them, and a caller that unpacks the counts is a
    caller that can forget it.
    """
    import flashinfer

    return flashinfer.QSA.workspace_requirements(config, device=device)


def allocate_qsa_persistent(
    needs: flashinfer.QSAWorkspaceRequirements, device: torch.device
) -> torch.Tensor:
    """The bytes the plans live in, which are this side's to keep alive.

    Not the worker's workspace: that is scratch every consumer reuses from its
    first byte, and a plan read back on a later call cannot live in memory
    somebody else writes in between.
    """
    buffer = torch.empty(needs.persistent_bytes, dtype=torch.uint8, device=device)
    if buffer.data_ptr() % needs.alignment:
        raise RuntimeError(
            f"the QSA persistent buffer has to start on {needs.alignment} bytes, "
            f"got {buffer.data_ptr():#x}"
        )
    return buffer


def _manager():
    """The workspace this execution context draws on, or an error saying not.

    There is no fallback to a private allocation: scratch nothing reserved is
    scratch the memory profile never counted, and a runtime holding some is the
    failure this path exists to prevent.
    """
    if not is_workspace_manager_initialized():
        raise RuntimeError(
            "Qwen4Exp QSA needs a workspace manager: its scratch comes out of "
            "the worker's workspace, reserved while it can still grow"
        )
    return current_workspace_manager()


def reserve_qsa_transient(needs: flashinfer.QSAWorkspaceRequirements) -> None:
    """Claim the scratch while the workspace can still grow.

    Called from the owner's constructor, which is the only place this can
    happen: the profiling run returns before QSA executes, so nothing else
    would ask for these bytes until a real request did -- and by then the
    workspace is locked and cannot grow.
    """
    _manager().get_simultaneous(((needs.transient_bytes,), torch.uint8))


def take_qsa_transient(
    needs: flashinfer.QSAWorkspaceRequirements,
) -> torch.Tensor:
    """The scratch view as the manager hands it out right now.

    Not once and kept: the engine binds the KV cache before it locks the
    workspace -- the memory profile needs a cache to capture against -- and an
    unlocked workspace grows by reallocating, which frees whatever earlier
    views point into. So this is asked again at every bind, and the caller
    compares what it gets with what its runtime is holding.
    """
    view = _manager().get_simultaneous(((needs.transient_bytes,), torch.uint8))[0]
    if view.data_ptr() % needs.alignment:
        raise RuntimeError(
            f"the QSA scratch has to start on {needs.alignment} bytes, got "
            f"{view.data_ptr():#x}"
        )
    return view


def build_qsa_runtime(
    config: flashinfer.QSAConfig,
    persistent: torch.Tensor,
    transient: torch.Tensor,
) -> flashinfer.QSA:
    """Hand the library both buffers and let it lay them out.

    Called at a bind rather than once, because the first bind happens before
    the workspace is locked: the memory profile needs a cache to capture
    against, and until the lock another consumer may still grow the workspace
    and move the scratch. So the caller takes the scratch again at every bind
    and builds a new runtime whenever the address or the cache differs from
    what the one it holds was built on; after the lock the address stops
    moving and the last runtime is the one that serves.

    Never inside a forward: the plans keep byte offsets into the persistent
    buffer, a CUDA graph replays the pointers of both, and a plan built inside
    a capture is not a thing.
    """
    import flashinfer

    runtime = flashinfer.QSA(config, persistent)
    runtime.bind_transient_workspace(transient)
    return runtime


def qsa_compressed_view(k_compressed: torch.Tensor) -> torch.Tensor:
    """The compressed key cache as the scorer reads it.

    It is bound as ``[pages, states, 1, head_dim]`` -- the KV cache spec gives
    every page a head axis and this one holds a single head -- and the scorer
    reads ``[pages, page_size, head_dim]``. Checked and squeezed rather than
    reshaped: a reshape would accept any 4D tensor whose numbers happen to
    multiply out, and this is the only place the join changes a shape at all.
    """
    if k_compressed.ndim != 4:
        return k_compressed
    if k_compressed.shape[2] != 1:
        raise ValueError(
            "the compressed key cache carries one head per page, got "
            f"{k_compressed.shape[2]}"
        )
    return k_compressed.squeeze(2)
