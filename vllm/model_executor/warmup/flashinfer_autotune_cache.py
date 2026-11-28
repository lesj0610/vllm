# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig, replace
from vllm.config.utils import hash_config, hash_factors, normalize_value
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _normalize_cache_config_for_hash(cache_config: CacheConfig) -> CacheConfig:
    """Normalize cache config for hashing"""
    if cache_config.mamba_block_size is None and cache_config.kv_cache_layout is None:
        return cache_config
    normalized = replace(cache_config, mamba_block_size=None)
    assert normalized.kv_cache_layout is None
    return normalized


def _flashinfer_mm_encoder_autotune_hash(
    mm_config: MultiModalConfig, *, enable_chunked_prefill: bool
) -> str:
    return hash_factors(
        {
            "encoder_config": hash_config(mm_config),
            "enable_chunked_prefill": enable_chunked_prefill,
            "video_pruning_rate": mm_config.video_pruning_rate,
            "video_pruning_method": mm_config.video_pruning_method,
            "mm_processor_kwargs": normalize_value(mm_config.mm_processor_kwargs),
            "limit_per_prompt": normalize_value(mm_config.limit_per_prompt),
        }
    )


def get_flashinfer_mm_encoder_autotune_config(
    vllm_config: VllmConfig,
) -> MultiModalConfig | None:
    """Return the multimodal config when encoder autotuning is enabled."""
    model_config = vllm_config.model_config
    if (
        model_config is None
        or not model_config.supports_multimodal_inputs
        or not vllm_config.kernel_config.enable_flashinfer_mm_encoder_autotune
    ):
        return None
    mm_config = model_config.multimodal_config
    if mm_config is None or mm_config.skip_mm_profiling:
        return None
    return mm_config


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    vllm_config: VllmConfig = runner.vllm_config
    cache_config = _normalize_cache_config_for_hash(vllm_config.cache_config)
    if cache_config is not vllm_config.cache_config:
        vllm_config = replace(vllm_config, cache_config=cache_config)
    factors = vllm_config.compile_factors(include_version=False)
    mm_config = get_flashinfer_mm_encoder_autotune_config(vllm_config)
    if mm_config is not None:
        factors["flashinfer_mm_encoder_autotune"] = (
            _flashinfer_mm_encoder_autotune_hash(
                mm_config,
                enable_chunked_prefill=vllm_config.scheduler_config.enable_chunked_prefill,
            )
        )
    # DP runs with padding skip tune TRT-LLM MoE's precomputed-routing ops
    # instead of the logits ops; a cache from before lacks them, and a hit
    # skips all tuning.
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and envs.VLLM_MOE_SKIP_PADDING
        and not envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        and any(
            isinstance(m, RoutedExperts) and m.supports_precomputed_routing
            for m in runner.get_model().modules()
        )
    ):
        factors = {**factors, "moe_routed": True}
    return hash_factors(factors)


def runs_flashinfer_autotune(runner: "GPUModelRunner") -> bool:
    """FlashInfer tunes its kernels from SM90, and below only its MoE experts
    and NVFP4 GEMMs, which pick among their tactics per token count."""
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(90):
        return True

    from vllm.model_executor.kernels.linear import (
        FlashInferCudnnNvFp4LinearKernel,
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferTrtllmNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.fused_moe import MoERunner
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
        FlashInferCuteDSLExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsBase,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsBase,
    )

    flashinfer_experts = (
        FlashInferCuteDSLExperts,
        FlashInferExperts,
        TrtLlmMxfp4ExpertsBase,
        TrtLlmNvFp4ExpertsBase,
    )
    flashinfer_gemms = (
        FlashInferCudnnNvFp4LinearKernel,
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferTrtllmNvFp4LinearKernel,
    )
    # The drafter is outside the target's module tree.
    for model in (runner.get_model(), runner.get_draft_model()):
        for module in model.modules() if model is not None else ():
            for holder_name in ("quant_method", "scheme"):
                holder = getattr(module, holder_name, None)
                if isinstance(getattr(holder, "kernel", None), flashinfer_gemms):
                    return True
            if isinstance(module, MoERunner):
                kernel = getattr(module._quant_method, "moe_kernel", None)
                if kernel is not None and isinstance(
                    kernel.impl.fused_experts, flashinfer_experts
                ):
                    return True
    return False


def flashinfer_autotune_root() -> Path:
    """Root directory of the FlashInfer autotune cache (no runner needed)."""
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        return Path(override_dir).expanduser()
    from flashinfer.jit import env as flashinfer_jit_env

    flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
    return (
        Path(envs.VLLM_CACHE_ROOT)
        / "flashinfer_autotune_cache"
        / flashinfer_workspace.parent.name
        / flashinfer_workspace.name
    )


def flashinfer_autotune_filename(dp_rank: int, rank: int) -> str:
    """Per-rank cache filename, shared by the engine reader and D2D exchange."""
    return f"autotune_configs_dp{dp_rank}_rank{rank}.json"


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    output_dir = flashinfer_autotune_root() / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def resolve_rank_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """The per-rank variant of `resolve_flashinfer_autotune_file`."""
    from vllm.distributed.parallel_state import get_world_group

    path = resolve_flashinfer_autotune_file(runner)
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    return path.with_name(
        flashinfer_autotune_filename(dp_rank, get_world_group().rank_in_group)
    )


def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from vllm.distributed.parallel_state import get_node_count, get_world_group
    from vllm.utils.flashinfer import has_flashinfer_autotune_v2

    # Elastic EP transfers tuning state to joining workers with v1
    # save_configs/load_configs, which do not export managed-store winners.
    if runner.vllm_config.parallel_config.enable_elastic_ep:
        return False
    # PP stages profile independently. Keep their group-scoped v1 files until
    # managed-cache reads and publishes can be isolated across tuning groups.
    if runner.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return False
    if not has_flashinfer_autotune_v2():
        return False
    # Use the initialized topology: Ray may span nodes with config.nnodes=1.
    # A cache hit skips profiling collectives, so node-local stores with
    # different contents cannot safely participate in synchronized tuning.
    if get_node_count() != 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-node deployments; managed-cache synchronization is "
            "currently supported only within one node."
        )
        return False
    # Lift this gate after pinning a release with the fix below and validating
    # distributed managed-cache tuning: https://github.com/flashinfer-ai/flashinfer/pull/6185
    if get_world_group().world_size > 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-rank deployments until FlashInfer coordinates cache hits."
        )
        return False
    return True


def resolve_flashinfer_autotune_v2_root() -> Path | None:
    """Placement root for FlashInfer's managed autotune store.

    Unlike `resolve_flashinfer_autotune_file`, no vLLM-side identity
    (config hash, flashinfer version, arch) is folded into the path: the
    managed store keys entries per op and hashes the software/hardware
    environment itself. `None` selects FlashInfer's default root.
    """
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    return Path(override_dir).expanduser() if override_dir else None


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and runs_flashinfer_autotune(runner):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
