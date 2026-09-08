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
    _convert_custom_tool_lark_to_ebnf,
    constrain_custom_tool_formats,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

pytestmark = pytest.mark.skip_global_cleanup


APPLY_PATCH_LARK = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF"""


def test_apply_patch_lark_converts_to_xgrammar_ebnf() -> None:
    ebnf = _convert_custom_tool_lark_to_ebnf(APPLY_PATCH_LARK)

    assert ebnf.startswith("root ::= begin_patch hunk+ end_patch")
    assert "filename ::= [^\\n]+" in ebnf
    assert 'add_line ::= "+" [^\\n]* LF' in ebnf
    assert 'LF ::= "\\n"' in ebnf
    assert "-> line" not in ebnf


def test_custom_grammar_replaces_qwen_xml_string_schema() -> None:
    xgrammar = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string

    structure_tag = xgrammar.StructuralTag.model_validate(
        {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "<tool_call>\n<function=apply_patch>\n",
                "content": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                    "style": "qwen_xml",
                    "any_order": False,
                },
                "end": "\n</function>\n</tool_call>",
            },
        }
    )
    request = SimpleNamespace(
        vllm_original_tools=[
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": APPLY_PATCH_LARK,
                },
            }
        ]
    )

    constrained = constrain_custom_tool_formats(structure_tag, request).model_dump()
    content = constrained["format"]["content"]
    assert content["type"] == "tag"
    assert content["begin"] == "<parameter=input>"
    assert content["end"] == "</parameter>"
    assert content["content"]["type"] == "grammar"
    assert content["content"]["grammar"].startswith("root ::=")
    grammar = xgrammar.Grammar.from_ebnf(content["content"]["grammar"])
    complete = "*** Begin Patch\n*** Add File: result.txt\n+ok\n*** End Patch\n"
    incomplete = "*** Begin Patch\n*** Add File: result.txt\n+ok\n"
    assert _is_grammar_accept_string(grammar, complete)
    assert not _is_grammar_accept_string(grammar, incomplete)


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
