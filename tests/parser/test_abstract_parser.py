# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser, structured_outputs_to_format
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_registry import ToolChoice


class TestToolChoice_Plus_ResponseFormat:
    """Note(arpera):
    Test cases for tool_choice={auto,required} + response_format
    To keep it short:
    DelegatingParser.adjust_request behavior in some corner cases is checked there

    Initial bug report:
    https://github.com/vllm-project/vllm/issues/39929
    And PR that fixed this:
    https://github.com/vllm-project/vllm/pull/56086
    """

    # ================================
    # Helper methods
    # ================================

    @staticmethod
    def _tools(strict: bool) -> list[ChatCompletionToolsParam]:
        """Single get_weather tool, optionally marked as strict"""
        function: dict[str, Any] = {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        if strict:
            function["strict"] = True
        return [ChatCompletionToolsParam(type="function", function=function)]

    @staticmethod
    def _qwen_tool_call() -> str:
        """Tool call for Qwen model that do support structural tag"""
        return (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
            "</parameter>\n</function>\n</tool_call>"
        )

    @staticmethod
    def _json_schema_response_format() -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    @staticmethod
    def _setup_request(
        tools: list[ChatCompletionToolsParam],
        tool_choice: ToolChoice,
        response_format: dict,
    ) -> ChatCompletionRequest:
        request = ChatCompletionRequest(
            messages=[],  # for our test cases it's always empty
            model="m",  # Just a placeholder, don't pay much attention
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        return request

    @staticmethod
    def _setup_abstract_parser(
        tools: list[ChatCompletionToolsParam],
    ) -> DelegatingParser:
        """Construct parser that does NOT support structural tag"""

        class TestParser(DelegatingParser):
            tool_parser_cls = ToolParser

        return TestParser(MagicMock(), tools=tools)

    @staticmethod
    def _setup_qwen_parser(
        tools: list[ChatCompletionToolsParam],
    ) -> DelegatingParser:
        """Construct parser that supports structural tag"""

        class TestParser(DelegatingParser):
            tool_parser_cls = Qwen3EngineToolParser

        return TestParser(MagicMock(), tools=tools)

    # ================================
    # Test cases
    # tool_choice=auto + response_format
    # ================================

    @pytest.mark.parametrize(
        # In this test we check that for response_format
        # resulting grammar accepts @compliant_output and rejects @non_compliant_output
        # You can add more examples here if you see some corner cases not covered
        ("response_format", "compliant_output", "non_compliant_output"),
        [
            (_json_schema_response_format(), '{"text": "hi"}', '{"foo": 1}'),
            ({"type": "json_object"}, '{"any": 1}', "[1, 2]"),
        ],
        # We test here two different response_format types:
        ids=["json_schema", "json_object"],
    )
    def test_auto_with_strict_tools(
        self,
        response_format: dict,
        compliant_output: str,
        non_compliant_output: str,
    ):
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=response_format,
        )
        parser = self._setup_qwen_parser(tools)
        out = parser.adjust_request(request)

        # Now check that request does not have response_format anymore
        # but instead has structured_outputs set as structural tag
        # And this structural tag is OR operation
        assert out.tool_choice == "auto"
        assert out.response_format is None
        assert out.structured_outputs is not None
        tag = json.loads(out.structured_outputs.structural_tag)
        assert tag["format"]["type"] == "or"
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)

        assert _is_grammar_accept_string(grammar, compliant_output)
        assert not _is_grammar_accept_string(grammar, non_compliant_output)

        # Also tool call must be accepted by grammar
        assert _is_grammar_accept_string(grammar, self._qwen_tool_call())

        # IMPORTANT(arpera): Regression test
        # If we in adjust_request implementation by mistake
        # construct structural tag using tool_choice=auto
        # then such a structural tag would allow plain text as well
        # We need to be sure that plain text is NOT accepted in our case
        assert not _is_grammar_accept_string(grammar, "Hello")

    def test_auto_without_strict_tools(self):
        tools = self._tools(strict=False)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=self._json_schema_response_format(),
        )
        parser = self._setup_qwen_parser(tools)

        # There must be a warning that tool calls are disabled
        # Consume that warning
        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    def test_when_model_does_not_have_structural_tag(self):
        """Note(arpera):
        When model does NOT have structural tag support
        we apply constraint only for response_format
        """
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=self._json_schema_response_format(),
        )
        # SIC! use parser whose model does NOT support structural tag
        parser = self._setup_abstract_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    def test_auto_with_xgrammar_unsupported_schema(self):
        """Conrer case based on Vadim's feedback in PR #56086
        https://github.com/vllm-project/vllm/pull/56086#pullrequestreview-5329203459
        """
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "integer", "multipleOf": 2},
                },
            },
        )
        parser = self._setup_qwen_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    def test_auto_with_xgrammar_unsupported_nested_schema(self):
        """Conrer case based on Vadim's feedback in PR #56086
        https://github.com/vllm-project/vllm/pull/56086#pullrequestreview-5329203459
        """
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "number", "multipleOf": 0.5}},
                        "required": ["value"],
                    },
                },
            },
        )
        parser = self._setup_qwen_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    def test_auto_with_lark_grammar(self):
        """Conrer case based on Vadim's feedback in PR #56086
        https://github.com/vllm-project/vllm/pull/56086#pullrequestreview-5329203459
        Lark grammars are converted to EBNF before being merged into the tag.
        """
        tools = self._tools(strict=True)
        request = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=tools,
            tool_choice="auto",
            structured_outputs=StructuredOutputsParams(grammar='start: "ok"'),
        )
        parser = self._setup_qwen_parser(tools)
        out = parser.adjust_request(request)

        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        assert _is_grammar_accept_string(grammar, "ok")
        assert not _is_grammar_accept_string(grammar, "bad")
        assert _is_grammar_accept_string(grammar, self._qwen_tool_call())

    # ================================
    # Test cases
    # tool_choice=required + response_format
    # ================================

    def test_required(self):
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="required",
            response_format=self._json_schema_response_format(),
        )
        parser = self._setup_qwen_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is None
        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        assert _is_grammar_accept_string(grammar, self._qwen_tool_call())
        assert not _is_grammar_accept_string(grammar, '{"text": "hi"}')
        mock_warn.assert_called_once()


class TestResponsesCustomToolStructuralTags:
    @staticmethod
    def _request(
        tool_choice: Any = "auto",
        *,
        grammar_syntax: str = "lark",
        with_output_format: bool = False,
        parallel_tool_calls: bool | None = None,
    ) -> ResponsesRequest:
        data: dict[str, Any] = {
            "model": "m",
            "input": "Use the tool",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch",
                    "format": {
                        "type": "grammar",
                        "syntax": grammar_syntax,
                        "definition": 'start: "CUSTOM"',
                    },
                }
            ],
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        if with_output_format:
            data["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            }
        return ResponsesRequest.model_validate(data)

    @staticmethod
    def _parser(request: ResponsesRequest) -> DelegatingParser:
        class TestParser(DelegatingParser):
            tool_parser_cls = Qwen3EngineToolParser

        return TestParser(MagicMock(), tools=request.tools)

    @staticmethod
    def _custom_tool_call() -> str:
        return (
            "<tool_call>\n<function=apply_patch>\n"
            "<parameter=input>CUSTOM</parameter>\n"
            "</function>\n</tool_call>"
        )

    def test_custom_grammar_constrains_only_tool_branch(self) -> None:
        request = self._request(with_output_format=True)

        out = self._parser(request).adjust_request(request)

        assert out.structured_outputs is not None
        tag = json.loads(out.structured_outputs.structural_tag)
        assert tag["format"]["type"] == "or"
        tool_branch, output_branch = tag["format"]["elements"]
        assert '"type": "grammar"' in json.dumps(tool_branch)
        assert '"type": "grammar"' not in json.dumps(output_branch)
        assert output_branch["type"] == "json_schema"

    def test_auto_with_structured_output_preserves_both_branches(self) -> None:
        request = self._request(with_output_format=True)

        out = self._parser(request).adjust_request(request)

        assert out.text is None
        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        assert _is_grammar_accept_string(grammar, self._custom_tool_call())
        assert _is_grammar_accept_string(grammar, '{"answer":"done"}')
        assert not _is_grammar_accept_string(grammar, '{"other":"done"}')

    def test_custom_grammar_and_single_call_constraint_compose(self) -> None:
        request = self._request(
            with_output_format=True,
            parallel_tool_calls=False,
        )

        out = self._parser(request).adjust_request(request)

        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        custom_call = self._custom_tool_call()
        assert _is_grammar_accept_string(grammar, custom_call)
        assert not _is_grammar_accept_string(grammar, custom_call + custom_call)
        assert _is_grammar_accept_string(grammar, '{"answer":"done"}')

    @pytest.mark.parametrize(
        "tool_choice",
        ["required", {"type": "custom", "name": "apply_patch"}],
        ids=["required", "named"],
    )
    def test_required_and_named_choices_preserve_custom_grammar(
        self, tool_choice: Any
    ) -> None:
        request = self._request(tool_choice=tool_choice)

        out = self._parser(request).adjust_request(request)

        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        assert _is_grammar_accept_string(grammar, self._custom_tool_call())
        assert not _is_grammar_accept_string(
            grammar,
            self._custom_tool_call().replace("CUSTOM", "OTHER"),
        )

    def test_unsupported_custom_grammar_fails_closed(self) -> None:
        request = self._request(grammar_syntax="regex")

        with pytest.raises(
            VLLMValidationError,
            match="Unsupported custom-tool grammar syntax",
        ):
            self._parser(request).adjust_request(request)


@pytest.mark.parametrize("field", ["json", "structural_tag"])
def test_structured_outputs_to_format_rejects_deeply_nested_string(field):
    """Parsers convert the constraint before request validation, and json.loads
    raises RecursionError on a string nested this deeply."""
    schema = '{"type": "array", "items": ' * 20_000 + "{}" + "}" * 20_000
    value = schema
    if field == "structural_tag":
        value = (
            '{"type": "structural_tag", "format": {"type": "json_schema", '
            f'"json_schema": {schema}}}}}'
        )
    with pytest.raises(VLLMValidationError, match="nested too deeply"):
        structured_outputs_to_format(StructuredOutputsParams(**{field: value}))
