# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import regex as re

from vllm.exceptions import VLLMValidationError

_LARK_RULE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<body>.*)$")
_LARK_ALIAS_RE = re.compile(r"\s+->\s+[A-Za-z_][A-Za-z0-9_]*\s*$")


def _convert_custom_tool_lark_to_ebnf(definition: str) -> str:
    """Convert the bounded Lark subset used by Responses custom tools.

    xgrammar structural tags embed EBNF, while the Responses API specifies
    custom-tool grammars as Lark.  This converter deliberately accepts the
    portable subset used by Codex-style line-oriented grammars and rejects
    unsupported constructs instead of silently weakening the constraint.
    """
    if not definition.strip():
        raise VLLMValidationError(
            "A grammar custom tool must have a non-empty definition.",
            parameter="tools",
        )

    output: list[str] = []
    imported_lf = False
    saw_start = False
    for line_number, raw_line in enumerate(definition.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line == "%import common.LF":
            imported_lf = True
            continue
        if line.startswith("%"):
            raise VLLMValidationError(
                f"Unsupported Lark directive on line {line_number}: {line!r}.",
                parameter="tools",
            )

        match = _LARK_RULE_RE.match(line)
        if match is None:
            raise VLLMValidationError(
                f"Unsupported Lark grammar line {line_number}: {line!r}.",
                parameter="tools",
            )
        name = match.group("name")
        body = _LARK_ALIAS_RE.sub("", match.group("body").strip())
        # OpenAI's apply_patch grammar uses these two line-local regexes.  They
        # have exact EBNF character-class equivalents and cannot cross LF.
        body = body.replace("/(.+)/", r"[^\n]+").replace("/(.*)/", r"[^\n]*")
        if re.search(r"/(?:\\.|[^/])+/[A-Za-z]*", body):
            raise VLLMValidationError(
                "This Lark regular expression cannot be represented exactly "
                f"by the structured-output backend (line {line_number}).",
                parameter="tools",
            )
        if name == "start":
            name = "root"
            saw_start = True
        output.append(f"{name} ::= {body}")

    if not saw_start:
        raise VLLMValidationError(
            "A grammar custom tool must define a start rule.", parameter="tools"
        )
    if imported_lf:
        output.append(r'LF ::= "\n"')
    return "\n".join(output)


def constrain_custom_tool_formats(structure_tag: Any, request: Any) -> Any:
    """Embed Responses custom-tool grammars in a parser structural tag.

    Custom tools are lowered to single-string function tools for chat-template
    compatibility.  Without this rewrite, structural decoding constrains only
    the outer JSON/XML argument object, not the declared grammar of ``input``.
    """
    original_tools = getattr(request, "vllm_original_tools", None)
    if not original_tools:
        return structure_tag

    grammars: dict[str, str] = {}
    for tool in original_tools:
        if not isinstance(tool, dict) or tool.get("type") != "custom":
            continue
        tool_format = tool.get("format")
        if not isinstance(tool_format, dict) or tool_format.get("type") != "grammar":
            continue
        syntax = str(tool_format.get("syntax") or "").lower()
        if syntax != "lark":
            raise VLLMValidationError(
                f"Unsupported custom-tool grammar syntax: {syntax!r}.",
                parameter="tools",
            )
        grammars[str(tool.get("name") or "")] = _convert_custom_tool_lark_to_ebnf(
            str(tool_format.get("definition") or "")
        )
    if not grammars:
        return structure_tag

    payload = structure_tag.model_dump()
    constrained: set[str] = set()

    def rewrite(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                rewrite(item)
            return
        if not isinstance(node, dict):
            return

        begin = node.get("begin")
        content = node.get("content")
        if node.get("type") == "tag" and isinstance(begin, str):
            for name, grammar in grammars.items():
                if f"<function={name}>" not in begin:
                    continue
                if not (
                    isinstance(content, dict)
                    and content.get("type") == "json_schema"
                    and content.get("style") == "qwen_xml"
                ):
                    continue
                schema = content.get("json_schema")
                properties = (
                    schema.get("properties") if isinstance(schema, dict) else None
                )
                input_schema = (
                    properties.get("input") if isinstance(properties, dict) else None
                )
                if not (
                    isinstance(schema, dict)
                    and schema.get("type") == "object"
                    and isinstance(properties, dict)
                    and set(properties) == {"input"}
                    and isinstance(input_schema, dict)
                    and input_schema.get("type") == "string"
                    and schema.get("required") == ["input"]
                    and schema.get("additionalProperties") is False
                ):
                    continue
                node["content"] = {
                    "type": "tag",
                    "begin": "<parameter=input>",
                    "content": {"type": "grammar", "grammar": grammar},
                    "end": "</parameter>",
                }
                constrained.add(name)
                break

        for value in node.values():
            rewrite(value)

    rewrite(payload)
    missing = set(grammars) - constrained
    if missing:
        names = ", ".join(sorted(missing))
        raise VLLMValidationError(
            "The active tool parser cannot exactly constrain the declared "
            f"custom-tool grammar for: {names}.",
            parameter="tools",
        )

    from xgrammar import StructuralTag

    return StructuralTag.model_validate(payload)


def _custom_input_description(tool: dict[str, Any]) -> str:
    """Preserve model-visible custom-tool format requirements when lowering."""
    guidance = [
        "Provide one complete raw input string for this custom tool.",
        (
            "The string must satisfy the declared custom-tool format exactly; do "
            "not replace required control characters with their escaped textual "
            "spelling."
        ),
    ]
    tool_format = tool.get("format")
    if not isinstance(tool_format, dict):
        return " ".join(guidance)

    format_type = str(tool_format.get("type") or "").strip()
    if format_type:
        guidance.append(f"Format type: {format_type}.")
    if format_type == "grammar":
        syntax = str(tool_format.get("syntax") or "").strip()
        definition = str(tool_format.get("definition") or "").strip()
        if syntax:
            guidance.append(f"Grammar syntax: {syntax}.")
        if definition:
            guidance.append(f"The complete grammar is:\n{definition}")
    return " ".join(guidance)


def lower_custom_tools(data: Any) -> Any:
    """Lower Responses custom tools to function tools for chat renderers."""
    if not isinstance(data, dict):
        return data
    tools = data.get("tools")
    if not isinstance(tools, list):
        return data

    lowered_data = copy.deepcopy(data)
    original_tools = copy.deepcopy(tools)
    existing_names = {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and tool.get("name")
    }
    custom_names: set[str] = set()
    lowered_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "custom":
            lowered_tools.append(copy.deepcopy(tool))
            continue
        name = str(tool.get("name") or "")
        if not name:
            raise VLLMValidationError(
                "A custom tool must have a non-empty name.", parameter="tools"
            )
        if name in existing_names or name in custom_names:
            raise VLLMValidationError(
                f"Tool name {name!r} is ambiguous across tool types.",
                parameter="tools",
            )
        custom_names.add(name)
        description = tool.get("description") or "Provide freeform text input."
        lowered_tools.append(
            {
                "type": "function",
                "name": name,
                "description": (
                    f"{description} Return exactly one complete input string in the "
                    "required custom-tool format."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": _custom_input_description(tool),
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )
    if not custom_names:
        return data

    lowered_data["tools"] = lowered_tools
    lowered_data["vllm_original_tools"] = original_tools
    lowered_data["vllm_custom_tool_names"] = custom_names
    original_choice = copy.deepcopy(data.get("tool_choice", "auto"))
    lowered_data["vllm_original_tool_choice"] = original_choice
    if isinstance(original_choice, dict) and original_choice.get("type") == "custom":
        name = str(original_choice.get("name") or "")
        if name not in custom_names:
            raise VLLMValidationError(
                "Custom tool choice does not match a declared custom tool.",
                parameter="tool_choice",
            )
        lowered_data["tool_choice"] = {"type": "function", "name": name}

    request_input = lowered_data.get("input")
    if isinstance(request_input, list):
        lowered_data["input"] = [
            _lower_custom_input_item(item) for item in request_input
        ]
    return lowered_data


def _lower_custom_input_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    item_type = item.get("type")
    lowered = copy.deepcopy(item)
    if item_type == "custom_tool_call":
        custom_input = lowered.pop("input", "")
        lowered["type"] = "function_call"
        lowered["arguments"] = json.dumps(
            {"input": custom_input}, ensure_ascii=False, separators=(",", ":")
        )
    elif item_type == "custom_tool_call_output":
        lowered["type"] = "function_call_output"
    return lowered


def _decode_custom_input(arguments: str) -> str | None:
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("input"), str):
        return None
    return value["input"]


def _restore_custom_call(item: dict[str, Any]) -> dict[str, Any]:
    restored = copy.deepcopy(item)
    arguments = str(restored.pop("arguments", ""))
    custom_input = _decode_custom_input(arguments)
    restored["type"] = "custom_tool_call"
    restored["input"] = custom_input if custom_input is not None else arguments
    restored.pop("status", None)
    return restored


def restore_custom_response(response: Any) -> dict[str, Any]:
    original_tools = getattr(response, "vllm_original_tools", None)
    custom_names = frozenset(getattr(response, "vllm_custom_tool_names", set()))
    payload = response.model_dump(mode="json", by_alias=True)
    if not original_tools or not custom_names:
        return payload
    payload["tools"] = copy.deepcopy(original_tools)
    payload["tool_choice"] = copy.deepcopy(
        getattr(response, "vllm_original_tool_choice", "auto")
    )
    output = payload.get("output")
    if isinstance(output, list):
        payload["output"] = [
            _restore_custom_call(item)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") in custom_names
            else item
            for item in output
        ]
    return payload


@dataclass
class CustomToolStreamTransformer:
    custom_names: frozenset[str] = frozenset()
    original_tools: list[dict[str, Any]] | None = None
    original_tool_choice: Any = "auto"
    custom_item_ids: set[str] = field(default_factory=set)
    argument_buffers: dict[str, str] = field(default_factory=dict)
    emitted_deltas: set[str] = field(default_factory=set)

    def transform(self, event: Any) -> list[dict[str, Any]]:
        response = getattr(event, "response", None)
        if response is not None:
            names = frozenset(getattr(response, "vllm_custom_tool_names", set()))
            if names:
                self.custom_names = names
                self.original_tools = copy.deepcopy(
                    getattr(response, "vllm_original_tools", None)
                )
                self.original_tool_choice = copy.deepcopy(
                    getattr(response, "vllm_original_tool_choice", "auto")
                )

        payload = event.model_dump(mode="json", by_alias=True)
        event_type = payload.get("type")
        response_payload = payload.get("response")
        if isinstance(response_payload, dict) and self.custom_names:
            if self.original_tools is not None:
                response_payload["tools"] = copy.deepcopy(self.original_tools)
                response_payload["tool_choice"] = copy.deepcopy(
                    self.original_tool_choice
                )
            output = response_payload.get("output")
            if isinstance(output, list):
                response_payload["output"] = [
                    _restore_custom_call(item)
                    if isinstance(item, dict)
                    and item.get("type") == "function_call"
                    and item.get("name") in self.custom_names
                    else item
                    for item in output
                ]

        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = payload.get("item")
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and item.get("name") in self.custom_names
            ):
                item_id = str(item.get("id") or payload.get("item_id") or "")
                if item_id:
                    self.custom_item_ids.add(item_id)
                payload["item"] = _restore_custom_call(item)

        item_id = str(payload.get("item_id") or "")
        if event_type == "response.function_call_arguments.delta" and (
            item_id in self.custom_item_ids
        ):
            self.argument_buffers[item_id] = self.argument_buffers.get(
                item_id, ""
            ) + str(payload.get("delta") or "")
            custom_input = _decode_custom_input(self.argument_buffers[item_id])
            if custom_input is None:
                return []
            self.emitted_deltas.add(item_id)
            payload["type"] = "response.custom_tool_call_input.delta"
            payload["delta"] = custom_input
            return [payload]

        if event_type == "response.function_call_arguments.done" and (
            item_id in self.custom_item_ids
        ):
            arguments = str(payload.pop("arguments", ""))
            custom_input = _decode_custom_input(arguments)
            if custom_input is None:
                custom_input = arguments
            transformed: list[dict[str, Any]] = []
            if item_id not in self.emitted_deltas:
                transformed.append(
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "delta": custom_input,
                        "item_id": item_id,
                        "output_index": payload.get("output_index", 0),
                        "sequence_number": payload.get("sequence_number", 0),
                    }
                )
            payload["type"] = "response.custom_tool_call_input.done"
            payload["input"] = custom_input
            transformed.append(payload)
            return transformed
        return [payload]
