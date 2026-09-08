# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.openai.responses.custom_tools import (
    CustomToolStreamTransformer,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


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
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }
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
