# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu import warmup as gpu_warmup


class _FakeKVConnector:
    def __init__(self) -> None:
        self.disabled_states: list[bool] = []

    def set_disabled(self, disabled: bool) -> None:
        self.disabled_states.append(disabled)


def test_warmup_kernels_includes_chunked_prefill(monkeypatch) -> None:
    monkeypatch.setattr(gpu_warmup.torch.accelerator, "synchronize", lambda: None)

    kv_connector = _FakeKVConnector()
    runner = SimpleNamespace(
        num_speculative_steps=0,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=256,
        ),
        max_num_tokens=256,
        max_model_len=4096,
        is_pooling_model=False,
        is_last_pp_rank=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 32000),
        kv_connector=kv_connector,
    )
    outputs = []
    sample_inputs = []
    dummy_runs = []

    def execute_model(scheduler_output):
        outputs.append(scheduler_output)

    def sample_tokens(grammar_output):
        sample_inputs.append(grammar_output)

    def dummy_run(**kwargs):
        dummy_runs.append(kwargs)

    runner._dummy_run = dummy_run

    gpu_warmup.warmup_kernels(runner, execute_model, sample_tokens)

    req_id = "_chunked_prefill_warmup_"
    chunked_new_outputs = [
        output
        for output in outputs
        if output.scheduled_new_reqs and output.scheduled_new_reqs[0].req_id == req_id
    ]
    chunked_cached_outputs = [
        output for output in outputs if output.scheduled_cached_reqs.req_ids == [req_id]
    ]

    assert len(chunked_new_outputs) == 1
    assert len(chunked_cached_outputs) == 1

    chunked_new = chunked_new_outputs[0]
    assert chunked_new.num_scheduled_tokens == {req_id: 64}
    assert chunked_new.total_num_scheduled_tokens == 64
    assert chunked_new.scheduled_new_reqs[0].prefill_token_ids == list(range(128))
    assert len(chunked_new.scheduled_new_reqs[0].block_ids[0]) == 4

    chunked_cached = chunked_cached_outputs[0]
    assert chunked_cached.num_scheduled_tokens == {req_id: 64}
    assert chunked_cached.total_num_scheduled_tokens == 64
    assert chunked_cached.scheduled_cached_reqs.num_computed_tokens == [64]
    assert chunked_cached.scheduled_cached_reqs.num_output_tokens == [0]
    assert chunked_cached.scheduled_cached_reqs.new_block_ids is not None
    assert len(chunked_cached.scheduled_cached_reqs.new_block_ids[0][0]) == 4

    cleanup_outputs = [
        output for output in outputs if req_id in output.finished_req_ids
    ]
    assert len(cleanup_outputs) == 1
    assert kv_connector.disabled_states == [True, False]
    assert len(sample_inputs) == 4
    assert dummy_runs == [
        {
            "num_tokens": 128,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": False,
            "profile_seq_lens": 4096,
        }
    ]
