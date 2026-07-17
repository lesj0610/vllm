# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)


@pytest.mark.parametrize(
    "info_cls",
    [Qwen3_5ProcessingInfo, Qwen3_5MoeProcessingInfo],
)
@pytest.mark.parametrize(
    ("processor_kwargs", "expected_kwargs"),
    [
        ({}, {}),
        ({"use_fast": True}, {"use_fast": True}),
        ({"use_fast": False}, {"use_fast": False}),
        ({"use_fast": None}, {"use_fast": None}),
        ({"max_pixels": 1024}, {"max_pixels": 1024}),
    ],
)
def test_qwen3_5_processor_does_not_force_deprecated_use_fast(
    info_cls,
    processor_kwargs: dict[str, object],
    expected_kwargs: dict[str, object],
) -> None:
    ctx = MagicMock()
    hf_processor = MagicMock(spec=Qwen3VLProcessor)
    ctx.get_hf_processor.return_value = hf_processor

    info = info_cls(ctx)

    assert info.get_hf_processor(**processor_kwargs) is hf_processor
    ctx.get_hf_processor.assert_called_once_with(Qwen3VLProcessor, **expected_kwargs)
