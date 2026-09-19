# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the vLLM side of the QSA join owes, and nothing past it.

Five things: refuse a deployment FlashInfer cannot serve, say what the KV
cache's bytes are, describe the configuration in the library's terms, reserve
the bytes the library asked for, and hand them over -- then bind the one
runtime a rank shares and run a step out of it. Everything else -- the row
ladder, the route and its mask, the plans, the top-k, the output gate, where
each buffer sits inside the workspace -- is the library's, and a test for one
of those belongs in its suite.

``WorkspaceManager``'s own growth and lock behaviour is vLLM's and is covered
where that class lives; ``QSA.plan_cache``'s refusal to replan after a run is
FlashInfer's and is covered in its suite. What is here is only what this join
owes. The NVFP4 writer-to-reader roundtrip stays in its own file: it is about
what the cache's bytes mean, not about who owns the runtime.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.utils import bind_kv_cache_to_layers
from vllm.v1.worker.workspace import WorkspaceManager

flashinfer = pytest.importorskip("flashinfer")

from vllm.models.qwen4_exp.nvidia import model as _qwen4_exp_model  # noqa: E402,F401
from vllm.models.qwen4_exp.nvidia import qsa as qsa_module  # noqa: E402
from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa_flashinfer  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="QSA runs on CUDA"
)

PAGE_SIZE = 16
PAGES = 32
NUM_SLOTS = PAGES * PAGE_SIZE
HEAD_DIM = 128
NUM_QO_HEADS = 8
NUM_KV_HEADS = 2
INDEX_HEADS = 4
COMPRESS_RATIO = 4
TOKEN_TOPK = 32
MAX_ROWS = 64
MAX_COLUMNS = 64
NUM_LAYERS = 4
ROUTE_WIDTH = flashinfer.selection_route_width(TOKEN_TOPK, COMPRESS_RATIO)

SUPPORTED_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")


@pytest.fixture
def device():
    if not torch.accelerator.is_available():
        pytest.skip("an accelerator is required")
    return torch.accelerator.current_accelerator()


@pytest.fixture
def worker(device):
    """A worker's workspace manager, installed and torn down like the real one."""
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        yield
    finally:
        reset_workspace_manager()


def _config(**overrides):
    """The deployment, as the adapter describes it to the library."""
    arguments = {
        "num_qo_heads": NUM_QO_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_rows": MAX_ROWS,
        "kv_cache_dtype": "auto",
        "max_columns": MAX_COLUMNS,
        "compress_ratio": COMPRESS_RATIO,
        "token_topk": TOKEN_TOPK,
        "index_num_heads": INDEX_HEADS,
        "index_head_dim": HEAD_DIM,
    }
    arguments.update(overrides)
    return qsa_flashinfer.qsa_config(**arguments)


# --- the gate -------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("kv_cache_dtype", SUPPORTED_DTYPES)
def test_every_supported_configuration_is_served(kv_cache_dtype):
    """The dtypes this backend has a cache format for are the ones it takes."""
    if current_platform.has_device_capability(100):
        pytest.skip("this route is pre-SM100")
    qsa_flashinfer.require_qsa_flashinfer(HEAD_DIM, kv_cache_dtype)


@requires_cuda
def test_a_missing_capability_refuses_the_layer(monkeypatch):
    """No silent Triton. A build without the kernels stops the deployment.

    The Triton implementation is still in the tree and this deployment does not
    serve on it: falling through would change the kernel, the memory profile
    and the answer to why a step got slower, with nothing said.
    """
    monkeypatch.setattr(flashinfer, "qsa_capabilities", lambda device: 0)
    with pytest.raises(RuntimeError, match="requires FlashInfer"):
        qsa_flashinfer.require_qsa_flashinfer(HEAD_DIM, "nvfp4")


@requires_cuda
def test_sm100_and_later_are_refused_rather_than_served_by_triton(monkeypatch):
    """This route decodes packed NVFP4 in software; those do it in one op."""
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda capability: True
    )
    with pytest.raises(RuntimeError, match="SM100"):
        qsa_flashinfer.require_qsa_flashinfer(HEAD_DIM, "nvfp4")


@requires_cuda
@pytest.mark.parametrize(
    ("reason", "head_dim", "kv_cache_dtype", "config_overrides", "match"),
    [
        # NVFP4 packs one scale per sixteen values; a head between two has none.
        ("head", 100, "nvfp4", None, "sixteen"),
        # The format is what the planner is instantiated for, not a guess.
        ("dtype", HEAD_DIM, "int8", None, "no cache format"),
        # The scorer is instantiated per head dimension and its mma tile bounds
        # the query heads. There is no probe that answers yes or no here: the
        # library is the only thing that knows what it was built with, so the
        # refusal comes from where the configuration is described.
        ("shape", None, None, {"index_num_heads": 64}, "scorer"),
    ],
)
def test_a_configuration_the_library_cannot_serve_is_refused(
    device, worker, reason, head_dim, kv_cache_dtype, config_overrides, match
):
    """Refused at startup, with the library's reason, not at the first step."""
    if config_overrides is None:
        with pytest.raises(RuntimeError, match=match):
            qsa_flashinfer.require_qsa_flashinfer(head_dim, kv_cache_dtype)
    else:
        with pytest.raises(ValueError, match=match):
            qsa_flashinfer.qsa_workspace_needs(_config(**config_overrides), device)


# --- the cache ------------------------------------------------------------


def _cache(device, kv_cache_dtype):
    kind, _layout, _dtype = qsa_flashinfer.qsa_cache_kinds(kv_cache_dtype)
    if kind == "nvfp4":
        full = HEAD_DIM // 2 + HEAD_DIM // 16
        return torch.zeros(
            PAGES, NUM_KV_HEADS, PAGE_SIZE, full, dtype=torch.uint8, device=device
        )
    storage = torch.uint8 if kind == "fp8_e4m3" else torch.bfloat16
    return torch.zeros(
        PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=storage, device=device
    )


@requires_cuda
@pytest.mark.parametrize(
    "kv_cache_dtype,expected_format,expected_layout",
    [
        ("auto", "dense", "NHD"),
        ("bfloat16", "dense", "NHD"),
        ("fp8", "fp8_e4m3", "NHD"),
        ("fp8_e4m3", "fp8_e4m3", "NHD"),
        ("nvfp4", "nvfp4", "HND"),
    ],
)
def test_the_cache_views_say_what_the_bytes_are(
    device, kv_cache_dtype, expected_format, expected_layout
):
    """Two formats arrive as raw bytes, so the format travels with the views.

    A uint8 tensor is a view of memory and says nothing on its own: e4m3 values
    and packed NVFP4 both look like one. This side allocated the cache, so this
    side names it -- and the configuration the workspace was sized for is built
    through the same mapping, so the two cannot disagree.
    """
    cache = _cache(device, kv_cache_dtype)
    views = qsa_flashinfer.qsa_cache_views(cache, cache, kv_cache_dtype)

    assert (views.format, views.layout) == (expected_format, expected_layout)
    assert views.page_size == PAGE_SIZE
    if expected_format == "nvfp4":
        assert views.k_sf is not None and views.v_sf is not None
        assert views.k_sf.dtype == torch.float8_e4m3fn
    else:
        assert views.k_sf is None and views.v_sf is None
    if expected_format == "fp8_e4m3":
        assert views.k_data.dtype == torch.float8_e4m3fn

    config = _config(kv_cache_dtype=kv_cache_dtype)
    assert (config.kv_cache_format, config.kv_layout) == (views.format, views.layout)
    assert config.kv_data_type == views.k_data.dtype


@requires_cuda
@pytest.mark.parametrize("kv_cache_dtype", SUPPORTED_DTYPES)
def test_the_planes_are_views_and_not_copies(device, kv_cache_dtype):
    """A copy would be a second cache, and a stale one after the first write."""
    cache = _cache(device, kv_cache_dtype)
    views = qsa_flashinfer.qsa_cache_views(cache, cache, kv_cache_dtype)
    assert views.k_data.data_ptr() == cache.data_ptr()
    if views.k_sf is not None:
        begin = cache.data_ptr()
        end = begin + cache.numel() * cache.element_size()
        assert begin < views.k_sf.data_ptr() < end


@requires_cuda
def test_the_compressed_cache_reaches_the_scorer_as_three_axes(device):
    """It is bound with a head axis of one; the scorer reads it without.

    Squeezed rather than reshaped: a reshape would take any four-axis tensor
    whose numbers happen to multiply out, and this is the only place the join
    changes a shape at all.
    """
    four_d = torch.zeros(PAGES, PAGE_SIZE, 1, HEAD_DIM, device=device)
    three_d = qsa_flashinfer.qsa_compressed_view(four_d)
    assert tuple(three_d.shape) == (PAGES, PAGE_SIZE, HEAD_DIM)
    assert three_d.data_ptr() == four_d.data_ptr(), "the squeeze copied"

    with pytest.raises(ValueError, match="one head per page"):
        qsa_flashinfer.qsa_compressed_view(
            torch.zeros(PAGES, PAGE_SIZE, 2, HEAD_DIM, device=device)
        )


# --- the workspace --------------------------------------------------------


@requires_cuda
def test_describing_the_workspace_costs_no_memory(device, worker):
    """Asked while the memory profiler is deciding what is left for the cache.

    Building anything to get the answer would put its buffers into the startup
    peak and take them back out of the KV cache.
    """
    config = _config()
    qsa_flashinfer.qsa_workspace_needs(config, device)  # pay any imports
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated, reserved = (
        torch.accelerator.memory_allocated(),
        torch.accelerator.memory_reserved(),
    )

    sizes = [
        tuple(qsa_flashinfer.qsa_workspace_needs(config, device)) for _ in range(8)
    ]
    torch.accelerator.synchronize()

    assert len(set(sizes)) == 1, "the same configuration sized differently"
    persistent, transient, alignment = sizes[0]
    assert persistent > 0 and transient > 0 and alignment > 0
    assert torch.accelerator.memory_allocated() == allocated
    assert torch.accelerator.memory_reserved() == reserved
    assert torch.accelerator.max_memory_allocated() <= allocated


@requires_cuda
def test_scratch_comes_from_a_worker_that_reserved_it_or_not_at_all(device, worker):
    """Both ways of asking for scratch nobody counted are refused.

    There is no private allocation to fall back to, on purpose: scratch nothing
    reserved is scratch the memory profile never saw, and a runtime holding
    some is the failure this path exists to prevent.
    """
    from vllm.v1.worker.workspace import lock_workspace, reset_workspace_manager

    needs = qsa_flashinfer.qsa_workspace_needs(_config(), device)

    # Locked, with nothing reserved: the buffer can no longer grow to fit.
    lock_workspace()
    with pytest.raises(AssertionError, match="Workspace is locked"):
        qsa_flashinfer.take_qsa_transient(needs)

    # No manager at all, which is what a call outside a worker gets.
    reset_workspace_manager()
    with pytest.raises(RuntimeError, match="needs a workspace manager"):
        qsa_flashinfer.reserve_qsa_transient(needs)


@requires_cuda
def test_the_scratch_is_the_managers_and_the_plans_are_not(device, worker):
    """Reserve the scratch while it can grow, build once it cannot.

    The two buffers have to come from different places, and this is what says
    so: the transient one starts where the manager's allocation starts, and the
    persistent one is nowhere inside it. Anything read back that lived in the
    manager's bytes would be overwritten by the next consumer of the step.
    Both start on the alignment the library asked for, which the adapter checks
    rather than computes.
    """
    from vllm.v1.worker.workspace import current_workspace_manager, lock_workspace

    config = _config()
    needs = qsa_flashinfer.qsa_workspace_needs(config, device)
    persistent = qsa_flashinfer.allocate_qsa_persistent(needs, device)
    qsa_flashinfer.reserve_qsa_transient(needs)
    lock_workspace()

    transient = qsa_flashinfer.take_qsa_transient(needs)
    assert persistent.data_ptr() % needs.alignment == 0
    assert transient.data_ptr() % needs.alignment == 0
    runtime = qsa_flashinfer.build_qsa_runtime(config, persistent, transient)
    runtime.plan_cache(NUM_SLOTS, PAGE_SIZE)

    manager = current_workspace_manager()
    base = manager.get_simultaneous(((1,), torch.uint8))[0].data_ptr()
    assert transient.data_ptr() == base, "the scratch is not the manager's"

    begin = persistent.data_ptr()
    end = begin + persistent.numel()
    assert not (base <= begin < base + needs.transient_bytes), (
        "the plans live in the scratch every other consumer reuses"
    )
    assert end <= begin + needs.persistent_bytes
    assert runtime.num_slots == NUM_SLOTS


@requires_cuda
def test_the_scratch_view_follows_the_manager_when_it_grows(device, worker):
    """Which is why it is asked for again at every bind rather than kept.

    The engine binds the KV cache before it locks the workspace -- the memory
    profile needs a cache to capture against -- so another consumer asking for
    more room between two binds reallocates the buffer. A view taken before
    that points into memory the manager has given away.
    """
    from vllm.v1.worker.workspace import current_workspace_manager

    manager = current_workspace_manager()
    needs = qsa_flashinfer.qsa_workspace_needs(_config(), device)
    qsa_flashinfer.reserve_qsa_transient(needs)
    qsa_flashinfer.take_qsa_transient(needs)
    held = manager._current_workspaces[0].numel()

    # Another consumer, asking for more than QSA did.
    manager.get_simultaneous(((needs.transient_bytes + (1 << 20),), torch.uint8))
    assert manager._current_workspaces[0].numel() > held, "the manager did not grow"

    after = qsa_flashinfer.take_qsa_transient(needs)
    live = manager._current_workspaces[0]
    assert after.data_ptr() % needs.alignment == 0
    assert after.numel() == needs.transient_bytes
    # What the view has to be is inside the buffer the manager holds *now*.
    # Whether that is the same address as before is the allocator's business;
    # what the caller owes is asking again, which is what this does.
    assert live.data_ptr() <= after.data_ptr()
    assert (
        after.data_ptr() + after.numel()
        <= live.data_ptr() + live.numel() * live.element_size()
    )


# --- the order a runtime comes up in --------------------------------------


def _library_batch(device, rows, seed):
    """Query, cache and route for a call straight at the library."""
    generator = torch.Generator(device=device).manual_seed(seed)
    block_table = (
        torch.randperm(PAGES, device=device, generator=generator)
        .reshape(1, PAGES)
        .contiguous()
        .to(torch.int32)
    )
    k = torch.randn(
        PAGES,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn_like(k)
    q = torch.randn(
        rows,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    route = torch.full((rows, ROUTE_WIDTH), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        route[row] = torch.randperm(NUM_SLOTS, device=device, generator=generator)[
            :ROUTE_WIDTH
        ].to(torch.int32)
    gate = torch.randn(
        rows,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return q, k, v, route, block_table, token_to_req, gate


@requires_cuda
def test_both_halves_of_a_step_come_out_of_one_reservation(device):
    """One reservation, one buffer, both halves, and nothing else taken.

    The two halves are sized from different geometries -- the index heads are
    not the attention heads -- and asking twice would give two layouts of one
    allocation. Asking once is also what makes the memory budget true: whatever
    the pair holds after this is either inside those bytes or is memory the
    engine never counted.
    """
    manager = WorkspaceManager(device)
    config = _config()
    need = flashinfer.QSA.workspace_requirements(config, device=device)
    persistent = torch.empty(need.persistent_bytes, dtype=torch.uint8, device=device)
    manager.get_simultaneous(((need.transient_bytes,), torch.uint8))
    manager.lock()

    transient = manager.get_simultaneous(((need.transient_bytes,), torch.uint8))[0]
    gc.collect()
    torch.accelerator.synchronize()
    mark = torch.accelerator.memory_allocated()

    runtime = flashinfer.QSA(config, persistent)
    runtime.bind_transient_workspace(transient)
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == mark, (
        "the pair took memory outside the reservation"
    )

    runtime.plan_cache(NUM_SLOTS, PAGE_SIZE)
    rows = 16
    q, k, v, route, block_table, token_to_req, gate = _library_batch(device, rows, 4)
    generator = torch.Generator(device=device).manual_seed(5)
    compressed = torch.randn(
        PAGES,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = torch.full((rows,), NUM_SLOTS - 1, dtype=torch.int32, device=device)
    lengths = torch.full((1,), NUM_SLOTS, dtype=torch.int32, device=device)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    # The index query is its own geometry -- that the two halves are sized from
    # different ones is the reason a single reservation has to serve both.
    index_q = torch.randn(
        rows,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    # The route is the caller's buffer in both calls: the selection fills it
    # and the attention reads it.
    runtime.run_selection(
        index_q,
        compressed,
        block_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
    )
    runtime.run_attention(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gate,
        out=out,
    )
    torch.accelerator.synchronize()
    assert bool(torch.isfinite(out.to(torch.float32)).all())


# --- the layers of a rank -------------------------------------------------


def _indexer():
    """A QSA indexer carrying what its selection half reads, and no more."""
    indexer = QSAIndexer.__new__(QSAIndexer)
    indexer._qsa_selection = None
    indexer._page_size = PAGE_SIZE
    indexer.index_n_heads = INDEX_HEADS
    indexer.index_head_dim = HEAD_DIM
    indexer.token_topk = TOKEN_TOPK
    indexer.compress_ratio = COMPRESS_RATIO
    indexer._max_batched_tokens = MAX_ROWS
    indexer._max_selection_columns = MAX_COLUMNS
    indexer.qsa_runtime = None
    return indexer


def _impl():
    """The QSA attention impl, reserved as the owner's constructor reserves it."""
    impl = qsa_module.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_module.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.is_kvcache_nvfp4 = False
    impl.kv_cache_dtype = "auto"
    impl.head_size = HEAD_DIM
    impl.num_heads = NUM_QO_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    return impl


class _Owner(qsa_module.Qwen4ExpQSAAttention):
    """One decoder block's QSA owner, with only what binding reads.

    ``Qwen4ExpQSAAttention.bind_kv_cache`` is the real method under test; what
    is stood in for is everything the constructor would also have built.
    """

    def __init__(self, layer_name):
        torch.nn.Module.__init__(self)
        self.layer_name = layer_name
        self.kv_cache = torch.tensor([])
        # What the real constructor declares before the model settles it.
        self.qsa_backend = "triton"
        self.qsa_runtime = None
        self.qsa = None
        self.head_dim = HEAD_DIM
        self.num_heads = NUM_QO_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.kv_cache_dtype = "auto"
        self.impl = _impl()
        self.indexer = _indexer()
        # The route buffer the real constructor allocates at the packed width
        # the Triton path reads. Attaching the runtime replaces it with the
        # width the library reads, which is what the assertions below are
        # about. Zeroed, not empty: a step can reach the attention half before
        # anything has written a route, and an uninitialised int32 page carries
        # whatever the last tenant left.
        self.register_buffer(
            "topk_indices_buffer",
            torch.zeros(MAX_ROWS, TOKEN_TOPK + COMPRESS_RATIO, dtype=torch.int32),
            persistent=False,
        )


class _Block(torch.nn.Module):
    """A decoder block, which is what the model hands the attacher."""

    def __init__(self, owner):
        super().__init__()
        self.self_attn = owner


def _owners(count=NUM_LAYERS):
    """The rank's QSA layers, with the one runtime the model gives them all."""
    from vllm.v1.worker.workspace import lock_workspace

    owners = [_Owner(f"model.layers.{index}.self_attn") for index in range(count)]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=PAGE_SIZE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_ROWS),
    )
    qsa_module.attach_qsa_runtime(vllm_config, [_Block(owner) for owner in owners])
    # The engine locks the workspace once every consumer has reserved and
    # before anything binds, and QSA's scratch may only be taken after that.
    lock_workspace()
    return owners


def _caches(device, count=NUM_LAYERS, pages=PAGES):
    """One KV cache per layer, as the allocator hands them over.

    A page is laid out heads, then states, then content, so the allocation is
    ``(blocks, num_kv_heads, block_size, 2 * head_dim)`` and K and V share the
    last axis.
    """
    return {
        f"model.layers.{index}.self_attn": torch.zeros(
            pages,
            NUM_KV_HEADS,
            PAGE_SIZE,
            2 * HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for index in range(count)
    }


def _planes(cache):
    """The two NHD planes a dense QSA cache is read as, as the layer splits it."""
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

    key, value = cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    return (
        canonicalize_singleton_dim_strides(key),
        canonicalize_singleton_dim_strides(value),
    )


def _layer_batch(device, rows, seed):
    """What one layer's attention call takes, other than its cache."""
    generator = torch.Generator(device=device).manual_seed(seed)
    block_table = (
        torch.randperm(PAGES, device=device, generator=generator)
        .reshape(1, PAGES)
        .contiguous()
        .to(torch.int32)
    )
    q = torch.randn(
        rows,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.randn_like(q)
    route = torch.full((rows, ROUTE_WIDTH), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        route[row] = torch.randperm(NUM_SLOTS, device=device, generator=generator)[
            :ROUTE_WIDTH
        ].to(torch.int32)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    return q, gate, route, block_table, token_to_req


def _step_once(device, owners):
    """One QSA attention per owner, as a step runs them."""
    rows = 16
    for owner in owners:
        q, gate, route, block_table, token_to_req = _layer_batch(device, rows, seed=7)
        out = torch.empty(
            rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        views = qsa_flashinfer.qsa_cache_views(*_planes(owner.kv_cache), "auto")
        owner.qsa_runtime.run_attention(
            q,
            views.k_data,
            views.v_data,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gate,
            out=out,
        )
    torch.accelerator.synchronize()


def _prepared_step(device, owners, caches, randomize_caches=False):
    """Every layer's inputs, and the closure that runs the rank's step.

    Both of the tests below need the same thing set up the same way -- one
    batch and one pair of cache views per layer, then a callable that runs them
    in order -- and they differ only in what they watch while it runs.
    """
    rows = 16
    batches = []
    for index, owner in enumerate(owners):
        cache = caches[owner.layer_name]
        if randomize_caches:
            cache.normal_(generator=torch.Generator(device=device).manual_seed(index))
        q, gate, route, block_table, token_to_req = _layer_batch(device, rows, index)
        out = torch.empty(
            rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        views = qsa_flashinfer.qsa_cache_views(*_planes(cache), "auto")
        batches.append((owner, views, q, gate, route, block_table, token_to_req, out))

    def step():
        for owner, views, q, gate, route, table, token_to_req, out in batches:
            owner.qsa_runtime.run_attention(
                q,
                views.k_data,
                views.v_data,
                route=route,
                block_table=table,
                token_to_req=token_to_req,
                output_gate=gate,
                k_sf=views.k_sf,
                v_sf=views.v_sf,
                out=out,
            )

    return batches, step


@requires_cuda
def test_every_qsa_layer_of_a_rank_binds(device, worker):
    """The pass that binds them all, as the model runner runs it.

    A model has one of these per decoder block and they are bound in one pass.
    A runtime that a layer owns rather than leases makes the second layer fail,
    and every kernel test still passes, because no kernel test has two layers.
    """
    owners = _owners()
    caches = _caches(device)
    context = {owner.layer_name: owner for owner in owners}

    bind_kv_cache_to_layers(caches, context)

    for owner in owners:
        assert owner.qsa_runtime is not None
        assert owner.indexer.qsa_runtime is not None


@requires_cuda
def test_the_layers_of_one_slot_share_one_runtime(device, worker):
    """Not one each: one, shared.

    The layers of a slot run one after another inside a step, so the route, the
    mask and the padded buffers are the slot's and not the layer's. Two copies
    would be two of everything the arena holds, once per decoder block.
    """
    owners = _owners()
    bind_kv_cache_to_layers(_caches(device), {o.layer_name: o for o in owners})

    attention = {id(owner.qsa_runtime) for owner in owners}
    selection = {id(owner.indexer.qsa_runtime) for owner in owners}
    assert len(attention) == 1, f"{len(attention)} runtimes for one rank"
    assert len(selection) == 1, f"{len(selection)} selection handles for one rank"
    assert attention == selection, "the two halves are not the same object"


@requires_cuda
def test_the_profiling_cache_is_replaced_by_the_production_one(device, worker):
    """The two binds the engine does, and what has to be true between them.

    ``initialize_kv_cache`` runs twice: once for the minimal cache the memory
    profile runs and captures against, then for the real one, and the engine
    drops the profiling graphs in between. The library refuses to replan a
    runtime that has run -- it cannot see that the graphs are gone -- so the
    second cache gets a runtime of its own on the same two buffers, and the
    first one has to go with the graphs that held it.
    """
    owners = _owners()
    context = {owner.layer_name: owner for owner in owners}
    holder = owners[0].qsa
    persistent = holder.qsa_persistent.data_ptr()

    bind_kv_cache_to_layers(_caches(device, pages=2), context)
    profiling = owners[0].qsa_runtime
    assert profiling.num_slots == 2 * PAGE_SIZE
    assert all(owner.qsa_runtime is profiling for owner in owners)
    _step_once(device, owners[:1])
    dead = weakref.ref(profiling)
    del profiling

    bind_kv_cache_to_layers(_caches(device, pages=PAGES), context)
    production = owners[0].qsa_runtime
    gc.collect()

    assert dead() is None, "the profiling runtime outlived the cache it planned"
    assert production.num_slots == NUM_SLOTS
    assert all(owner.qsa_runtime is production for owner in owners)
    assert all(owner.indexer.qsa_runtime is production for owner in owners)
    assert holder.qsa_persistent.data_ptr() == persistent, (
        "the replacement took a second persistent buffer"
    )
    _step_once(device, owners)


@requires_cuda
def test_a_moved_scratch_replaces_the_runtime_even_for_the_same_cache(device, worker):
    """The cache did not change; the memory under the runtime did.

    The engine binds before it locks, so a consumer asking for more room
    between two binds reallocates the workspace. Cache geometry alone would say
    nothing happened, and the runtime would keep running out of memory the
    manager has given to somebody else.
    """
    from vllm.v1.worker.workspace import current_workspace_manager, unlock_workspace

    owners = _owners()
    context = {owner.layer_name: owner for owner in owners}
    caches = _caches(device)
    bind_kv_cache_to_layers(caches, context)
    first = owners[0].qsa_runtime
    held = first._attention._transient.data_ptr()

    # Another consumer, asking for more than QSA did, while it can still grow.
    unlock_workspace()
    manager = current_workspace_manager()
    grown = manager._current_workspaces[0].numel() * 2
    manager.get_simultaneous(((grown,), torch.uint8))
    if manager._current_workspaces[0].data_ptr() == held:
        pytest.skip("the allocator handed back the same address")
    dead = weakref.ref(first)
    del first

    # The same cache, bound again, as the second initialize_kv_cache does.
    bind_kv_cache_to_layers(caches, context)
    second = owners[0].qsa_runtime
    gc.collect()

    assert dead() is None, "the runtime on the freed scratch was kept"
    assert second._attention._transient.data_ptr() != held
    assert all(owner.qsa_runtime is second for owner in owners)
    assert all(owner.indexer.qsa_runtime is second for owner in owners)
    assert second.num_slots == NUM_SLOTS
    _step_once(device, owners)


@requires_cuda
def test_the_layers_run_one_after_another_and_replay(device, worker):
    """A step is every layer in turn, and a captured graph replays all of them.

    The shared runtime is what they take turns on, so this is where sharing
    either holds or leaves one layer reading what the next one wrote.
    """
    owners = _owners()
    caches = _caches(device)
    bind_kv_cache_to_layers(caches, {o.layer_name: o for o in owners})
    batches, step = _prepared_step(device, owners, caches, randomize_caches=True)

    step()
    torch.accelerator.synchronize()
    expected = [batch[-1].clone() for batch in batches]
    assert all(bool((one != 0).any()) for one in expected), "a layer wrote nothing"
    # Every layer's answer is its own: no two are the same tensor of numbers.
    assert len({one.sum().item() for one in expected}) == len(expected)

    for batch in batches:
        batch[-1].zero_()
    step()
    torch.accelerator.synchronize()
    for got, want in zip((batch[-1] for batch in batches), expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for batch in batches:
        batch[-1].zero_()
    graph.replay()
    torch.accelerator.synchronize()
    for got, want in zip((batch[-1] for batch in batches), expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_cuda
def test_a_step_builds_and_allocates_nothing(device, worker):
    """Whatever a forward would need had to be there before capture.

    A planner call, a JIT load or a tensor allocation inside a step is a thing
    a graph cannot contain and the memory profile cannot see.
    """
    import flashinfer.qsa_attention as attention_module
    import flashinfer.topk as topk_module

    owners = _owners()
    caches = _caches(device)
    bind_kv_cache_to_layers(caches, {o.layer_name: o for o in owners})
    _batches, step = _prepared_step(device, owners, caches)

    step()
    torch.accelerator.synchronize()

    reentered = []
    saved = {}

    def watch(owner, name, label):
        original = getattr(owner, name)
        saved[(owner, name)] = original

        def spy(*args, **kwargs):
            reentered.append(label)
            return original(*args, **kwargs)

        setattr(owner, name, spy)

    watch(flashinfer.BlockSparseAttentionWrapper, "plan", "plan")
    watch(flashinfer.BlockSparseAttentionWrapper, "query_workspace_size", "sizing")
    watch(attention_module, "BlockSparseAttentionWrapper", "wrapper")
    watch(topk_module, "get_topk_module", "topk_module")
    watch(flashinfer.QSA, "workspace_requirements", "sizing")
    watch(flashinfer.QSA, "plan_cache", "planning")
    try:
        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        baseline = torch.accelerator.memory_allocated()
        for _ in range(3):
            step()
        torch.accelerator.synchronize()
        assert not reentered, f"a step re-entered {sorted(set(reentered))}"
        # Peak, not net: a temporary freed before the next line still had to
        # come from somewhere the memory profile counted.
        assert torch.accelerator.max_memory_allocated() == baseline, "a step allocated"
    finally:
        for (owner, name), original in saved.items():
            setattr(owner, name, original)


# --- which backend, and when ----------------------------------------------


@requires_cuda
def test_the_backend_is_settled_when_the_model_is_built(device, worker):
    """Once, for the rank, and not re-decided by whether a runtime exists yet.

    These are two different questions and conflating them is how a FlashInfer
    deployment ends up quietly running a Triton kernel: at the moment the
    memory profile captures its graphs the cache has only just been bound, and
    a step that read "no runtime yet" as "use the other backend" would change
    the kernel, the route's width and the memory profile, with nothing said.
    """
    for owner in _owners():
        assert owner.qsa_backend == "flashinfer"
        assert owner.indexer.qsa_backend == "flashinfer"
        # Decided before anything is bound: the runtime is still absent here.
        assert owner.qsa_runtime is None


@requires_cuda
def test_a_step_before_the_cache_is_bound_fails_rather_than_falling_back(
    device, worker
):
    """Outside the lifecycle is an error, not a reason to run the other one."""
    owner = _owners(count=1)[0]
    assert owner.qsa_runtime is None, "nothing is bound in this test"
    rows = 16
    q, gate, route, block_table, token_to_req = _layer_batch(device, rows, seed=2)
    del route
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = _caches(device, count=1)[owner.layer_name]
    metadata = SimpleNamespace(
        num_actual_tokens=rows, block_table=block_table, max_query_len=1
    )
    owner.topk_indices_buffer[:rows].fill_(0)

    with pytest.raises(RuntimeError, match="not planned yet"):
        owner.impl.forward_qsa(
            owner,
            q,
            None,
            None,
            cache,
            metadata,
            out,
            token_to_req=token_to_req,
            use_prefill_config=False,
            output_gate=gate,
        )


@requires_cuda
def test_the_route_the_profile_runs_on_is_zero_and_not_whatever_was_there(
    device, worker, monkeypatch
):
    """The attacher's new route buffer is written, not just allocated.

    The buffer the Triton path allocated is replaced here by one at the width
    the library reads, and the CUDA graph memory profile can reach the
    attention half before the selection half has written anything into it. So
    its contents are what that step routes on. Zero is the only value that is
    both valid and the same on every run: an all -1 route is the library's
    invalid sentinel and would profile a fully masked step, and uninitialised
    memory routes somewhere different each time.

    Poisoned with a positive pattern rather than -1: an in-range leftover is
    accepted by every check and is exactly the case that used to pass.
    """
    from vllm.models.qwen4_exp.nvidia.ops import qsa as triton_qsa

    owners = [_Owner(f"model.layers.{index}.self_attn") for index in range(2)]
    for owner in owners:
        # Poisoned, and on the device: this one runs a step rather than
        # stopping at the attacher, so the route has to be where the kernel
        # reads it.
        owner.register_buffer(
            "topk_indices_buffer",
            torch.full(
                (MAX_ROWS, TOKEN_TOPK + COMPRESS_RATIO),
                0x5A5A,
                dtype=torch.int32,
                device=device,
            ),
            persistent=False,
        )
        assert int(owner.topk_indices_buffer.min()) != 0

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=PAGE_SIZE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_ROWS),
    )
    qsa_module.attach_qsa_runtime(vllm_config, [_Block(owner) for owner in owners])

    for owner in owners:
        route = owner.topk_indices_buffer
        assert route.shape == (MAX_ROWS, ROUTE_WIDTH), "the width was not replaced"
        assert int(route.abs().max()) == 0, (
            "the replacement route carries the old buffer's bytes"
        )

    # And the profiling step the engine runs next attends on it without
    # reaching the other backend.
    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()
    took_triton = []
    monkeypatch.setattr(
        triton_qsa,
        "qsa_sparse_paged_attention",
        lambda *args, **kwargs: took_triton.append(1),
    )

    context = {owner.layer_name: owner for owner in owners}
    bind_kv_cache_to_layers(_caches(device, count=2, pages=2), context)

    rows = 8
    owner = owners[0]
    q, gate, _route, block_table, token_to_req = _layer_batch(device, rows, seed=11)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    metadata = SimpleNamespace(
        num_actual_tokens=rows, block_table=block_table, max_query_len=1
    )
    owner.impl.forward_qsa(
        owner,
        q,
        None,
        None,
        owner.kv_cache,
        metadata,
        out,
        token_to_req=token_to_req,
        use_prefill_config=False,
        output_gate=gate,
    )
    torch.accelerator.synchronize()
    assert not took_triton, "the profiling step fell through to Triton"
    assert bool(torch.isfinite(out.to(torch.float32)).all())


@requires_cuda
def test_each_backend_gets_the_route_width_it_reads(device, worker):
    """One buffer per layer, at the width of the backend that will read it.

    The Triton route carries a trailing count column that its tile loop reads
    as a bound; the FlashInfer route is indices and nothing else. A buffer
    sized for one and written by the other runs off the end of it.
    """
    for owner in _owners():
        assert owner.qsa_backend == "flashinfer"
        assert owner.topk_indices_buffer.shape[1] == ROUTE_WIDTH
        assert owner.indexer.packed_output_width == ROUTE_WIDTH + 1
