# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config


def test_get_generation_config_from_local_model_file_parent(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    generation_config_path = tmp_path / "generation_config.json"
    generation_config_path.write_text(
        json.dumps(
            {
                "eos_token_id": [1, 2, 3],
                "stability_threshold": 1,
            }
        ),
        encoding="utf-8",
    )

    generation_config = try_get_generation_config(
        str(model_file), trust_remote_code=False
    )

    assert generation_config is not None
    diff_config = generation_config.to_diff_dict()
    assert diff_config["eos_token_id"] == [1, 2, 3]
    assert diff_config["stability_threshold"] == 1


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118
