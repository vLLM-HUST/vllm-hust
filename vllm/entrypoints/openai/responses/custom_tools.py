# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from vllm.exceptions import VLLMValidationError


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
        lowered_tools.append(
            {
                "type": "function",
                "name": name,
                "description": tool.get("description")
                or "Provide freeform text input.",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
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
