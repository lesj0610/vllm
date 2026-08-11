# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ATEM structural tag for MuseGlimmer required/named tool_choice.

Without the tag, opting out of the generic JSON constraint left required/named
tool_choice unconstrained: the model could answer in the user channel and call
nothing, while the serving layer still reported finish_reason="tool_calls".
These tests pin the grammar to the chat template's render_atem output, and pin
that it is no stricter than the parser accepts.
"""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SUPPORTED_STRUCTURAL_TAG_MODELS,
    get_model_structural_tag,
)

xgr = pytest.importorskip("xgrammar")
xgr_testing = pytest.importorskip("xgrammar.testing")


WEATHER = ChatCompletionToolsParam.model_validate(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    }
)
ORDER = ChatCompletionToolsParam.model_validate(
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a purchase order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["sku", "quantity"],
            },
        },
    }
)

# The generation prompt already ends with "<|start|>assistant", so a turn's
# first message starts at " to=".
CALL = (
    " to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Busan</atem:parameter>\n'
    '<atem:parameter name="unit">celsius</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eom|>"
)
REASONING_THEN_CALL = (
    " to=self<|message|>We need the weather. Use get_weather.<|eom|>"
    "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Oslo</atem:parameter>\n'
    '<atem:parameter name="unit">celsius</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eom|>"
)
# The template renders a bare tool's recipient as "<name>.*", so the model emits
# the doubled form; _normalize_name collapses it.
DOUBLED_NAME = CALL.replace("get_weather", "get_weather.get_weather")
PLAIN_ANSWER = " to=user<|message|>Hello! Nice to hear from you.<|eot|>"
BARE_CHANNEL_ANSWER = "<|message|>Hello there, how can I help?<|eot|>"
ENUM_VIOLATION = CALL.replace(">celsius<", ">kelvin<")
ORDER_CALL = (
    " to=place_order<|message|><atem:function_calls>\n"
    '<atem:invoke name="place_order">\n'
    '<atem:parameter name="sku">BR-7781</atem:parameter>\n'
    '<atem:parameter name="quantity">12</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eom|>"
)


def _tag(tool_choice, tools=None):
    return get_model_structural_tag(
        model="muse_glimmer",
        tools=tools if tools is not None else [WEATHER, ORDER],
        tool_choice=tool_choice,
        reasoning=False,
    )


def _accepts(tag, text: str) -> bool:
    grammar = xgr.Grammar.from_structural_tag(json.loads(json.dumps(tag.model_dump())))
    return xgr_testing._is_grammar_accept_string(grammar, text)


def test_registered():
    assert "muse_glimmer" in SUPPORTED_STRUCTURAL_TAG_MODELS
    assert MuseGlimmerToolParser.structural_tag_model == "muse_glimmer"
    # A parser declaring a structural tag parses required/named itself.
    assert MuseGlimmerToolParser.supports_required_and_named is False


@pytest.mark.parametrize(
    "tool_choice,expected",
    [("required", True), ("none", False), ("auto", False)],
)
def test_tag_built_only_when_constraining(tool_choice, expected):
    # "auto" is unconstrained unless a tool opts into strict mode; "none" never
    # constrains.
    assert (_tag(tool_choice) is not None) is expected


def test_no_tag_without_tools():
    assert _tag("required", tools=[]) is None


@pytest.mark.parametrize(
    "text",
    [CALL, REASONING_THEN_CALL, DOUBLED_NAME, ORDER_CALL],
    ids=["bare_recipient", "reasoning_then_call", "doubled_name", "second_tool"],
)
def test_required_accepts_valid_calls(text):
    assert _accepts(_tag("required"), text)


@pytest.mark.parametrize(
    "text",
    [PLAIN_ANSWER, BARE_CHANNEL_ANSWER, ENUM_VIOLATION],
    ids=["user_channel_answer", "bare_channel_answer", "enum_violation"],
)
def test_required_rejects(text):
    assert not _accepts(_tag("required"), text)


def test_named_choice_pins_the_tool():
    named = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "place_order"}}
    )
    tag = _tag(named)
    assert tag is not None
    assert _accepts(tag, ORDER_CALL)
    assert not _accepts(tag, CALL)
