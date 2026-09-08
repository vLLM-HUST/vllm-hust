# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.responses.api_router import (
    _convert_stream_to_sse_events,
)
from vllm.entrypoints.openai.responses.custom_tools import (
    CustomToolStreamTransformer,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

pytestmark = pytest.mark.skip_global_cleanup


def test_request_lowers_custom_tools_and_inputs_for_function_renderers() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "done",
                }
            ],
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/s",
                    },
                }
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
    )

    assert request.tools[0].type == "function"
    assert request.tools[0].parameters == {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": (
                    "Provide one complete raw input string for this custom tool. "
                    "The string must satisfy the declared custom-tool format exactly; "
                    "do not replace required control characters with their escaped "
                    "textual spelling. Format type: grammar. Grammar syntax: lark. "
                    "The complete grammar is:\nstart: /.+/s"
                ),
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }
    assert request.tools[0].description == (
        "Apply a patch Return exactly one complete input string in the required "
        "custom-tool format."
    )
    assert request.tool_choice.model_dump() == {
        "type": "function",
        "name": "apply_patch",
    }
    assert request.vllm_custom_tool_names == {"apply_patch"}
    assert request.vllm_original_tools[0]["type"] == "custom"
    assert request.input[0]["type"] == "function_call_output"


class _Event:
    def __init__(self, payload, response=None):
        self.payload = payload
        self.response = response

    def model_dump(self, **_kwargs):
        return self.payload


def test_stream_transformer_restores_custom_tool_events() -> None:
    response = SimpleNamespace(
        vllm_custom_tool_names={"apply_patch"},
        vllm_original_tools=[{"type": "custom", "name": "apply_patch"}],
        vllm_original_tool_choice="auto",
    )
    transformer = CustomToolStreamTransformer()
    transformer.transform(
        _Event(
            {"type": "response.created", "response": {"output": []}},
            response=response,
        )
    )
    added = transformer.transform(
        _Event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "item_1",
                    "call_id": "call_1",
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": "",
                },
            }
        )
    )
    delta = transformer.transform(
        _Event(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "item_1",
                "output_index": 0,
                "sequence_number": 2,
                "delta": '{"input":"*** Begin Patch"}',
            }
        )
    )

    assert added[0]["item"]["type"] == "custom_tool_call"
    assert delta[0]["type"] == "response.custom_tool_call_input.delta"
    assert delta[0]["delta"] == "*** Begin Patch"


@pytest.mark.asyncio
async def test_sse_converter_seeds_custom_context_from_request() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "Apply the patch",
            "stream": True,
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/s",
                    },
                }
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
    )

    async def events():
        yield _Event(
            {
                "type": "response.created",
                # Streaming serving serializes the response before constructing
                # this event, so excluded compatibility fields are absent here.
                "response": {"output": []},
            }
        )
        yield _Event(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "item_1",
                    "call_id": "call_1",
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": "",
                },
            }
        )
        yield _Event(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item_1",
                "output_index": 0,
                "sequence_number": 2,
                "arguments": '{"input":"*** Begin Patch"}',
            }
        )

    payloads = []
    async for block in _convert_stream_to_sse_events(events(), request):
        data_line = next(
            line for line in block.splitlines() if line.startswith("data: ")
        )
        payloads.append(json.loads(data_line.removeprefix("data: ")))

    assert payloads[1]["item"]["type"] == "custom_tool_call"
    assert payloads[2]["type"] == "response.custom_tool_call_input.delta"
    assert payloads[3]["type"] == "response.custom_tool_call_input.done"
