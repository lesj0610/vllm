# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from vllm.transformers_utils.gguf_utils import get_gguf_tokenizer_special_ids

from vllm.transformers_utils.config import (
    get_config,
    maybe_override_with_speculators,
)


def test_get_config_prefers_sibling_config_json_for_local_gguf(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 32,
                "n_embd": 16,
                "n_layer": 2,
                "n_head": 2,
            }
        )
    )

    config = get_config(model_dir / "model.gguf", trust_remote_code=False)

    assert config.model_type == "gpt2"


def test_speculator_detection_prefers_sibling_config_json_for_local_gguf(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 32,
                "n_embd": 16,
                "n_layer": 2,
                "n_head": 2,
            }
        )
    )

    model, tokenizer, speculative = maybe_override_with_speculators(
        model=str(model_dir / "model.gguf"),
        tokenizer=str(model_dir),
        trust_remote_code=False,
    )

    assert model == str(model_dir / "model.gguf")
    assert tokenizer == str(model_dir)
    assert speculative is None


def test_get_gguf_tokenizer_special_ids(monkeypatch, tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"GGUF")

    fields = {
        "tokenizer.ggml.bos_token_id": SimpleNamespace(parts=[2]),
        "tokenizer.ggml.eos_token_id": SimpleNamespace(parts=[106]),
        "tokenizer.ggml.unknown_token_id": SimpleNamespace(parts=[3]),
        "tokenizer.ggml.padding_token_id": SimpleNamespace(parts=[0]),
    }

    class FakeReader:
        def __init__(self, path):
            self.path = path

        def get_field(self, name):
            return fields.get(name)

    monkeypatch.setattr("vllm.transformers_utils.gguf_utils.gguf.GGUFReader", FakeReader)
    get_gguf_tokenizer_special_ids.cache_clear()

    assert get_gguf_tokenizer_special_ids(gguf_file) == {
        "bos_token_id": 2,
        "eos_token_id": 106,
        "unknown_token_id": 3,
        "padding_token_id": 0,
    }
