# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


def ple_layers_off_first_pipeline_stage(
    ple_layer_ids: Iterable[int],
    num_hidden_layers: int,
    pipeline_parallel_size: int,
) -> list[int]:
    """Return the PLE-owning decoder layers that miss the first PP stage.

    Non-first pipeline ranks never receive the raw input_ids that PLE needs, so
    the n-gram context is only correct on the first stage. That is enough as
    long as every PLE layer lives there: decoder layer ``i`` owns the PLE block
    for ``ple_layer_ids`` entry ``i + 1``.
    """
    if pipeline_parallel_size <= 1:
        return []
    _, first_stage_end = get_pp_indices(
        int(num_hidden_layers), 0, pipeline_parallel_size
    )
    return sorted(
        int(layer_id) - 1
        for layer_id in ple_layer_ids
        if not 0 <= int(layer_id) - 1 < first_stage_end
    )


def ple_pipeline_stage_error(offstage: list[int], first_stage_end: int) -> str:
    """Explain which PLE layers fall outside the first pipeline stage."""
    return (
        "N-gram PLE embedding requires every PLE layer to sit on the first "
        "pipeline stage, because only that stage receives the raw input_ids "
        f"PLE needs. Decoder layer(s) {offstage} hold PLE blocks but fall "
        f"outside the first stage's range [0, {first_stage_end}). Move the "
        "pipeline split later (VLLM_PP_LAYER_PARTITION) or run with "
        "pipeline_parallel_size=1."
    )


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count


class PLEVocabParallelEmbedding(VocabParallelEmbedding):
    """Vocab-parallel embedding that accepts checkpoint row shards."""

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if checkpoint_start is None:
            super().weight_loader(param, loaded_weight)
            return
        copy_ple_embedding_shard_(
            param,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=self.shard_indices.org_vocab_start_index,
            tp_end=self.shard_indices.org_vocab_end_index,
        )
