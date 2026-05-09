# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

import vllm.distributed.ec_transfer.ec_transfer_state as ec_transfer_state
import vllm.distributed.kv_transfer.kv_transfer_state as kv_transfer_state
from vllm import PoolingParams, SamplingParams
from vllm.distributed.parallel_state import get_pp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@contextmanager
def _disable_transfer_connectors_for_warmup() -> Iterator[None]:
    """Temporarily bypass connector-only branches for synthetic warmup.

    V1 execute_model() checks the global KV/EC connector agents directly and
    expects real scheduler metadata when they are active. Warmup scheduler
    outputs are synthetic, so disable the globals only while warming up and
    restore them even if warmup fails.
    """
    kv_connector_agent = kv_transfer_state._KV_CONNECTOR_AGENT
    ec_connector_agent = ec_transfer_state._EC_CONNECTOR_AGENT
    kv_transfer_state._KV_CONNECTOR_AGENT = None
    ec_transfer_state._EC_CONNECTOR_AGENT = None
    try:
        yield
    finally:
        kv_transfer_state._KV_CONNECTOR_AGENT = kv_connector_agent
        ec_transfer_state._EC_CONNECTOR_AGENT = ec_connector_agent


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    2 + num_spec_steps prompt tokens each. The second iteration simulates
    a decode step with all requests generating 1 + num_spec_steps tokens.
    """
    num_spec_steps = model_runner.num_speculative_steps
    # Use 1 + num_spec_steps + 1 tokens so the prefill batch's per-request
    # query length exceeds decode_query_len (= 1 + num_spec_steps), preventing
    # it from being misclassified as a uniform decode batch.
    prompt_len = 2 + num_spec_steps
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates 1 verified + num_spec_steps draft tokens.
    decode_len = prompt_len + 1 + num_spec_steps

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Compute per-request block counts for each KV cache group.
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, 1 + num_spec_steps),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # Step 1: Prefill all requests with 2 + num_spec_steps prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with 1 + num_spec_steps tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(_alloc_blocks(n) for n in decode_block_deltas) if new_block else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: 1 + num_spec_steps for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()


@torch.inference_mode()
def warmup_v1_kernels(
    model_runner: Any,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run V1 execute_model + sample_tokens to JIT compile runtime kernels.

    V1's legacy `_dummy_run()` path does not go through the real request input
    preparation path, so kernels such as slot mapping and backend-specific
    prefill/decode kernels can still JIT on the first user request. This warmup
    mirrors the V2 synthetic prefill/decode flow but uses V1 runner attributes.
    """
    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    if not kv_cache_groups:
        return

    num_spec_steps = model_runner.num_spec_tokens
    # Use a context batch that is longer than decode query len so V1 does not
    # classify the prefill step as a uniform decode batch.
    prompt_len = 2 + num_spec_steps
    prompt_token_ids = list(range(prompt_len))
    decode_len = prompt_len + 1 + num_spec_steps

    num_kv_cache_groups = len(kv_cache_groups)
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)
    if max_blocks_per_req <= 0:
        return

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, 1 + num_spec_steps),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )
    if num_reqs <= 0:
        return

    req_ids = [f"_v1_warmup_{i}_" for i in range(num_reqs)]

    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)

    with _disable_transfer_connectors_for_warmup():
        try:
            worker_execute_model(prefill_output)

            if not model_runner.is_pooling_model:
                grammar_output = None
                if get_pp_group().is_last_rank:
                    vocab_size = model_runner.model_config.get_vocab_size()
                    bitmask_width = (vocab_size + 31) // 32
                    grammar_bitmask = np.full(
                        (len(req_ids), bitmask_width),
                        fill_value=-1,
                        dtype=np.int32,
                    )
                    grammar_output = GrammarOutput(
                        structured_output_request_ids=req_ids,
                        grammar_bitmask=grammar_bitmask,
                    )

                worker_sample_tokens(grammar_output)

                cached_req_data = CachedRequestData.make_empty()
                cached_req_data.req_ids = list(req_ids)
                cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
                cached_req_data.num_output_tokens = [1] * num_reqs
                new_block = any(decode_block_deltas)
                cached_req_data.new_block_ids = [
                    tuple(_alloc_blocks(n) for n in decode_block_deltas)
                    if new_block
                    else None
                    for _ in range(num_reqs)
                ]

                decode_output = SchedulerOutput.make_empty()
                decode_output.scheduled_cached_reqs = cached_req_data
                decode_output.num_scheduled_tokens = {
                    req_id: 1 + num_spec_steps for req_id in req_ids
                }
                if num_spec_steps > 0:
                    decode_output.scheduled_spec_decode_tokens = {
                        req_id: [0] * num_spec_steps for req_id in req_ids
                    }
                decode_output.total_num_scheduled_tokens = sum(
                    decode_output.num_scheduled_tokens.values()
                )
                decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

                worker_execute_model(decode_output)
                worker_sample_tokens(None)
        finally:
            worker_execute_model(cleanup_output)

    torch.accelerator.synchronize()
