# SPDX-License-Identifier: Apache-2.0
"""Request-bound S1/S5 observations for an isolated pilot carrier.

The observer is opt-in: ordinary OpenAI requests are unchanged. Pilot
requests carry a small ``qbi_pre_action`` extra field. The API route records
that field before handing the request to the serving handler, giving the
runner a provider-side request boundary and a server timestamp. This module
does not inspect prompts, answers, or terminal outcomes.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any


MECHANISMS = (
    "native_tokens_prompt_prepared_input",
    "token_budget_output_control",
    "native_output_stream_batching",
    "repetition_detection_termination",
    # The provider request shape (tool count, input bytes, template complexity,
    # serialization work units) is a property of the parsed request, not of one
    # lifecycle.  S4's Suffix mechanism needs a provider-observed
    # ``available_tool_count``, so it is recorded through the same opt-in
    # snapshot rather than inventing a second, narrower observer.
    "native_suffix_speculation",
)

_OBSERVER = {
    mechanism: {"sequence": 0, "dropped": 0, "receipts": deque(maxlen=8192)}
    for mechanism in MECHANISMS
}
_STREAM_DELIVERY = {"sequence": 0, "dropped": 0, "receipts": deque(maxlen=8192)}

_PROVIDER_SNAPSHOT_SCHEMA = "anise.s1-s5.provider-context-snapshot.v1"
_PROVIDER_ENCODER_FIELDS = (
    "tokenizer_cache_state",
    "template_cache_state",
    "serialization_work_units",
    "preprocessor_queue_depth",
    "preprocessor_active_count",
    "preprocessor_pressure_ratio",
)


def _model_extra(request: Any) -> Mapping[str, Any]:
    extra = getattr(request, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}


def _pre_action(request: Any) -> Mapping[str, Any] | None:
    value = _model_extra(request).get("qbi_pre_action")
    return value if isinstance(value, Mapping) else None


def _request_value(request: Any, name: str) -> Any:
    """Read a protocol field without serializing the full request object."""

    value = getattr(request, name, None)
    if value is not None:
        return value
    if isinstance(request, Mapping):
        return request.get(name)
    return None


def _plain_input(value: Any) -> Any:
    """Convert protocol models to JSON-safe input shape without identities."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain_input(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_plain_input(nested) for nested in value]
    if isinstance(value, tuple):
        return [_plain_input(nested) for nested in value]
    return value


def _provider_request_shape(request: Any) -> dict[str, Any]:
    """Capture safe input structure at the provider request boundary.

    The observer sees the validated protocol object immediately before the
    serving handler.  Only messages, prompt representation and tool schema
    shape are retained; task ids, output, timing and response fields are
    never copied.  Token count is intentionally omitted here because this
    route does not own the tokenizer call.
    """

    messages = _request_value(request, "messages")
    messages = [_plain_input(item) for item in messages] if isinstance(messages, list) else []
    tools = _request_value(request, "tools")
    tools = [_plain_input(item) for item in tools] if isinstance(tools, list) else []
    prompt = _plain_input(_request_value(request, "prompt"))
    message_count = len(messages) if messages else (1 if prompt is not None else 0)
    roles = {
        str(item.get("role"))
        for item in messages
        if isinstance(item, Mapping) and item.get("role") is not None
    }
    content_bytes = 0
    content_units = 0
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if isinstance(content, str):
            content_bytes += len(content.encode("utf-8"))
            content_units += 1
        elif content is not None:
            content_bytes += len(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            content_units += len(content) if isinstance(content, (list, tuple, Mapping)) else 1
    if not messages and isinstance(prompt, str):
        content_bytes = len(prompt.encode("utf-8"))
        content_units = 1
    elif not messages and isinstance(prompt, list):
        content_bytes = len(prompt) * 4
        content_units = 1
    tool_key_count = sum(len(item) for item in tools if isinstance(item, Mapping))
    template_kwargs = _plain_input(_request_value(request, "chat_template_kwargs"))
    template_key_count = len(template_kwargs) if isinstance(template_kwargs, Mapping) else 0
    message_score = min(1.0, message_count / 8.0)
    role_score = min(1.0, len(roles) / 4.0)
    tool_score = min(1.0, len(tools) / 8.0)
    tool_shape_score = min(1.0, tool_key_count / 32.0)
    content_score = min(1.0, content_bytes / 2048.0)
    unit_score = min(1.0, content_units / 16.0)
    template_score = min(1.0, template_key_count / 8.0)
    input_bytes = len(
        json.dumps(
            {
                key: value
                for key, value in (
                    ("messages", messages),
                    ("prompt", prompt),
                    ("tools", tools),
                    ("chat_template_kwargs", template_kwargs),
                )
                if value is not None
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "schema": "anise.s1-s5.provider-request-shape.v1",
        "message_count": message_count,
        "tool_count": len(tools),
        "role_count": len(roles),
        "tool_schema_key_count": tool_key_count,
        "content_bytes": content_bytes,
        "content_units": content_units,
        "input_bytes": input_bytes,
        "template_complexity": round(
            0.25 * message_score
            + 0.15 * role_score
            + 0.20 * tool_score
            + 0.10 * tool_shape_score
            + 0.15 * content_score
            + 0.05 * unit_score
            + 0.10 * template_score,
            6,
        ),
        "serialization_work_units": input_bytes + 16 * message_count + 4 * tool_key_count,
        "prepared_input_available": isinstance(prompt, list),
    }


def _provider_renderer(provider_context: Any) -> Any:
    renderer = getattr(provider_context, "renderer", None)
    return renderer if renderer is not None else provider_context


def _provider_identity(value: Any) -> str | None:
    if value is None:
        return None
    class_name = f"{type(value).__module__}.{type(value).__qualname__}"
    name = getattr(value, "name_or_path", None) or getattr(value, "name", None)
    identity = f"{class_name}:{name}" if isinstance(name, str) and name else class_name
    return identity[:256]


def _template_identity(request: Any, provider_context: Any, renderer: Any, tokenizer: Any) -> str | None:
    template = _request_value(request, "chat_template")
    if template is None:
        template = getattr(provider_context, "chat_template", None)
    if template is None:
        template = getattr(renderer, "chat_template", None)
    if template is None:
        template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return None
    if not isinstance(template, str):
        template = json.dumps(_plain_input(template), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def _explicit_state(owner: Any, *names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = getattr(owner, name, None)
        if isinstance(value, str) and value in {"cold", "warm", "unknown"}:
            return value, f"{type(owner).__module__}.{type(owner).__qualname__}.{name}"
    return None, None


def _tokenizer_cache_state(provider_context: Any, renderer: Any, tokenizer: Any) -> tuple[str, str]:
    for owner in (provider_context, renderer, tokenizer):
        state, source = _explicit_state(owner, "qbi_tokenizer_cache_state", "tokenizer_cache_state")
        if state is not None:
            return state, source or "provider_context.tokenizer_cache_state"
    if tokenizer is None:
        return "cold", "vllm.renderers.BaseRenderer.tokenizer"
    return "warm", "vllm.renderers.BaseRenderer.tokenizer_initialized"


def _template_cache_state(provider_context: Any, renderer: Any, tokenizer: Any) -> tuple[str, str]:
    for owner in (provider_context, renderer, tokenizer):
        state, source = _explicit_state(owner, "qbi_template_cache_state", "template_cache_state")
        if state is not None:
            return state, source or "provider_context.template_cache_state"
    hf_renderer = sys.modules.get("vllm.renderers.hf")
    template_map = getattr(hf_renderer, "_PROCESSOR_CHAT_TEMPLATES", None)
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    model_config = getattr(renderer, "model_config", None)
    trust_remote_code = bool(getattr(model_config, "trust_remote_code", False))
    if isinstance(template_map, dict) and isinstance(tokenizer_name, str):
        if (tokenizer_name, trust_remote_code) in template_map:
            return "warm", "vllm.renderers.hf._PROCESSOR_CHAT_TEMPLATES"
    return "unknown", "vllm.renderers.hf.template_cache_state_unexported"


def _queue_value(queue: Any) -> int:
    qsize = getattr(queue, "qsize", None)
    if not callable(qsize):
        return 0
    try:
        value = qsize()
    except Exception:
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _preprocessor_state(provider_context: Any, renderer: Any) -> dict[str, Any]:
    async_tokenizer = getattr(renderer, "_async_tokenizer", None)
    if async_tokenizer is None:
        queue_depth = getattr(provider_context, "qbi_preprocessor_queue_depth", None)
        active_count = getattr(provider_context, "qbi_preprocessor_active_count", None)
        if isinstance(queue_depth, int) and queue_depth >= 0 and isinstance(active_count, int) and active_count >= 0:
            denominator = queue_depth + active_count
            return {
                "queue_depth": queue_depth,
                "active_count": active_count,
                "pressure_ratio": queue_depth / denominator if denominator else 0.0,
                "queue_source": "provider_context.qbi_preprocessor_queue_depth",
            }
        return {
            "queue_depth": None,
            "active_count": None,
            "pressure_ratio": None,
            "queue_source": "vllm.utils.async_utils.AsyncMicrobatchTokenizer.uninitialized",
        }
    queues = getattr(async_tokenizer, "_queues", {})
    queues = queues.values() if isinstance(queues, Mapping) else ()
    queue_depth = sum(_queue_value(queue) for queue in queues)
    tasks = getattr(async_tokenizer, "_batcher_tasks", ())
    active_count = sum(1 for task in tasks if not bool(getattr(task, "done", lambda: True)()))
    denominator = queue_depth + active_count
    return {
        "queue_depth": queue_depth,
        "active_count": active_count,
        "pressure_ratio": queue_depth / denominator if denominator else 0.0,
        "queue_source": "vllm.utils.async_utils.AsyncMicrobatchTokenizer._queues",
    }


def provider_context_snapshot(request: Any, provider_context: Any) -> dict[str, Any]:
    """Project provider-owned cache and preprocessor state before dispatch."""

    renderer = _provider_renderer(provider_context)
    tokenizer = getattr(renderer, "tokenizer", None)
    shape = _provider_request_shape(request)
    tokenizer_cache_state, tokenizer_cache_source = _tokenizer_cache_state(provider_context, renderer, tokenizer)
    template_cache_state, template_cache_source = _template_cache_state(provider_context, renderer, tokenizer)
    preprocessor = _preprocessor_state(provider_context, renderer)
    return {
        "schema": _PROVIDER_SNAPSHOT_SCHEMA,
        "tokenizer_identity": _provider_identity(tokenizer),
        "template_identity": _template_identity(request, provider_context, renderer, tokenizer),
        "tokenizer_cache_state": tokenizer_cache_state,
        "template_cache_state": template_cache_state,
        "serialization_work_units": shape["serialization_work_units"],
        "preprocessor_queue_depth": preprocessor["queue_depth"],
        "preprocessor_active_count": preprocessor["active_count"],
        "preprocessor_pressure_ratio": preprocessor["pressure_ratio"],
        "prepared_input_available": shape["prepared_input_available"],
        "source_fields": {
            "tokenizer_identity": "vllm.renderers.BaseRenderer.tokenizer",
            "template_identity": "vllm.renderers.hf.resolve_chat_template",
            "tokenizer_cache_state": tokenizer_cache_source,
            "template_cache_state": template_cache_source,
            "serialization_work_units": "vllm.entrypoints.serve.rpc.s1_s5_observer._provider_request_shape",
            "preprocessor_queue_depth": preprocessor["queue_source"],
            "preprocessor_active_count": preprocessor["queue_source"],
            "preprocessor_pressure_ratio": preprocessor["queue_source"],
        },
        "post_action_data_included": False,
    }


def _snapshot_with_binding(request: Any, provider_context: Any, body: Mapping[str, Any], observed_at_ns: int) -> dict[str, Any]:
    snapshot = provider_context_snapshot(request, provider_context)
    state = body["state"]
    bound = {
        **snapshot,
        "provider_request_id": body.get("provider_request_id") or state.get("provider_request_id") or state.get("decision_id"),
        "engine_boot_id": body.get("engine_boot_id") or state.get("engine_boot_id"),
        "config_epoch": body.get("config_epoch", state.get("config_epoch", 0)),
        "clock_id": state["clock_id"],
        "observed_at_ns": observed_at_ns,
    }
    bound["canonical_sha256"] = hashlib.sha256(
        json.dumps(bound, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return bound


def record_request_pre_action(request: Any, *, path: str, provider_context: Any | None = None) -> dict[str, Any] | None:
    """Record an opt-in request snapshot immediately before serving."""

    body = _pre_action(request)
    if body is None:
        return None
    mechanism = body.get("mechanism")
    if mechanism not in _OBSERVER:
        raise ValueError("unsupported qbi_pre_action mechanism")
    state = body.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("qbi_pre_action state is required")
    for name in ("source_receipt_sha256", "checkpoint_sha256", "clock_id", "decision_id"):
        if not isinstance(state.get(name), str) or not state[name]:
            raise ValueError(f"qbi_pre_action.{name} is required")
    record = copy.deepcopy(dict(body))
    record["provider_observed_at_ns"] = time.time_ns()
    record["provider_path"] = path
    record["provider_boundary"] = "after_protocol_validation_before_serving_handler"
    provider_request_id = state.get("provider_request_id") or state.get("decision_id")
    if not isinstance(provider_request_id, str) or not provider_request_id:
        raise ValueError("qbi_pre_action.provider_request_id is required")
    supplied_provider_request_id = body.get("provider_request_id")
    if supplied_provider_request_id is not None and supplied_provider_request_id != provider_request_id:
        raise ValueError("qbi_pre_action.provider_request_id differs from state")
    record["provider_request_id"] = provider_request_id
    record["provider_request_shape"] = _provider_request_shape(request)
    record["provider_shape_sha256"] = hashlib.sha256(
        json.dumps(record["provider_request_shape"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if provider_context is not None:
        record["provider_context_snapshot"] = _snapshot_with_binding(
            request, provider_context, body, record["provider_observed_at_ns"]
        )
    bucket = _OBSERVER[mechanism]
    if len(bucket["receipts"]) == bucket["receipts"].maxlen:
        bucket["dropped"] += 1
    bucket["sequence"] += 1
    record["sequence"] = bucket["sequence"]
    bucket["receipts"].append(record)
    return copy.deepcopy(record)


def state(mechanism: str, after_sequence: int = 0) -> dict[str, Any]:
    if mechanism not in _OBSERVER:
        raise ValueError("unsupported S1/S5 observer mechanism")
    if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
        raise ValueError("observer sequence cursor must be a non-negative integer")
    bucket = _OBSERVER[mechanism]
    receipts = [copy.deepcopy(item) for item in bucket["receipts"] if int(item["sequence"]) > after_sequence]
    oldest = int(bucket["receipts"][0]["sequence"]) if bucket["receipts"] else 0
    return {
        "schema": "anise.s1-s5.request-bound-observer-state.v1",
        "mechanism": mechanism,
        "provider_request_bound": True,
        "boundary": "after_protocol_validation_before_serving_handler",
        "capacity": bucket["receipts"].maxlen,
        "oldest_available_sequence": oldest,
        "latest_sequence": bucket["sequence"],
        "dropped_receipts": bucket["dropped"],
        "receipts": receipts,
    }


def stream_delivery_state(after_sequence: int = 0, decision_id: str | None = None) -> dict[str, Any]:
    """Read opt-in API carrier emissions, never response content or token claims."""
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("stream delivery cursor must be nonnegative")
    if decision_id is not None and (not isinstance(decision_id, str) or not decision_id):
        raise ValueError("stream delivery decision_id is invalid")
    bucket = _STREAM_DELIVERY
    return {
        "schema": "anise.request-bound-chat-stream-carrier-events.v1",
        "capacity": bucket["receipts"].maxlen,
        "oldest_available_sequence": bucket["receipts"][0]["sequence"] if bucket["receipts"] else 0,
        "latest_sequence": bucket["sequence"],
        "dropped_receipts": bucket["dropped"],
        "receipts": [copy.deepcopy(item) for item in bucket["receipts"]
                     if item["sequence"] > after_sequence
                     and (decision_id is None or item["decision_id"] == decision_id)],
    }


def _record_stream_carrier_frame(decision_id: str, interval: int, frame: bytes,
                                 frame_index: int, decoded_event_offset: int,
                                 prior_response_id: str | None,
                                 body_hasher: Any) -> tuple[int, str]:
    """Observe an SSE frame at the opt-in API carrier boundary."""
    response_ids: set[str] = set()
    decoded_events = 0
    finished = False
    for line in frame.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            finished = True
            continue
        event = json.loads(data)
        if not isinstance(event, Mapping) or not isinstance(event.get("id"), str):
            raise ValueError("opt-in Chat stream event lacks response id")
        response_ids.add(event["id"])
        decoded_events += 1
    if len(response_ids) > 1 or (not response_ids and prior_response_id is None):
        raise ValueError("opt-in Chat carrier frame lacks one response identity")
    response_id = next(iter(response_ids), prior_response_id)
    if prior_response_id is not None and response_id != prior_response_id:
        raise ValueError("opt-in Chat carrier mixed response identities")
    bucket = _STREAM_DELIVERY
    if len(bucket["receipts"]) == bucket["receipts"].maxlen:
        bucket["dropped"] += 1
    bucket["sequence"] += 1
    offset = decoded_event_offset + decoded_events
    body_hasher.update(frame)
    bucket["receipts"].append({
        "schema": "anise.request-bound-chat-stream-carrier-frame.v1",
        "sequence": bucket["sequence"], "decision_id": decision_id,
        "provider_response_id": response_id,
        "interval": interval, "frame_index": frame_index,
        "decoded_event_count": decoded_events, "decoded_event_offset": offset,
        "finished": finished, "carrier_emission_ns": time.time_ns(),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "emitted_body_sha256": body_hasher.hexdigest(),
    })
    return offset, response_id


def delivery_interval_ms(request: Any) -> float:
    body = _pre_action(request)
    if not isinstance(body, Mapping) or body.get("mechanism") != "native_output_stream_batching":
        return 0.0
    state_value = body.get("state")
    action_config = state_value.get("action_config") if isinstance(state_value, Mapping) else None
    if not isinstance(action_config, Mapping):
        return 0.0
    value = action_config.get("delivery_interval_ms", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value > 1000:
        raise ValueError("delivery_interval_ms must be in [0, 1000]")
    return float(value)


def stream_batch_size(request: Any) -> int:
    """Read the request-scoped delivery batch size, in decoded events per flush.

    LOCAL CARRIER PATCH (see
    ``work/replay_validity/FIX_STREAM_BATCHING_HOOK_20260922_V21.json``).  The
    native mechanism this hook exposes is the engine's own output batching
    (``SchedulerConfig.stream_interval``): emit an output frame once
    ``stream_interval`` decoded events have accumulated.  The request-scoped
    action uses the same unit and the same field name.  ``delivery_interval_ms``
    stays accepted as an alias so an already-recorded pre-action state keeps its
    meaning (1 ms of allowed delivery window = 1 event per flush).

    The consumer's declared delivery window is enforced here the same way the S5
    causal contract enforces it: when the requested batch exceeds
    ``allowed_delivery_interval_ms`` the action is inadmissible and the request is
    delivered event-by-event instead of being rejected, so an inadmissible action
    is observable as "no effect" rather than as a failed request.
    """

    body = _pre_action(request)
    if not isinstance(body, Mapping) or body.get("mechanism") != "native_output_stream_batching":
        return 1
    state_value = body.get("state")
    action_config = state_value.get("action_config") if isinstance(state_value, Mapping) else None
    if not isinstance(action_config, Mapping):
        return 1
    value = action_config.get("stream_interval")
    if value is None:
        legacy = action_config.get("delivery_interval_ms")
        if isinstance(legacy, bool) or not isinstance(legacy, (int, float)):
            return 1
        value = max(1, int(round(float(legacy))))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("stream_interval must be a positive integer")
    if value > 1000:
        raise ValueError("stream_interval is out of the supported range")
    if isinstance(state_value, Mapping):
        allowed = state_value.get("allowed_delivery_interval_ms")
        if allowed is not None:
            if isinstance(allowed, bool) or not isinstance(allowed, (int, float)) or allowed < 0:
                raise ValueError("allowed_delivery_interval_ms must be non-negative")
            if float(value) > float(allowed):
                # Same predicate as vllm_quality_bounded_inference.s5_causal_contract:
                # the consumer does not permit this delivery window.
                return 1
    return value


async def batch_sse(generator: AsyncIterable[Any], batch_size: int,
                    *, decision_id: str | None = None) -> AsyncIterator[bytes]:
    """Coalesce decoded events into one SSE flush per ``batch_size`` events.

    LOCAL CARRIER PATCH (see
    ``work/replay_validity/FIX_STREAM_BATCHING_HOOK_20260922_V21.json``).  Two
    defects made the original hook unable to serve as the mechanism's action
    plane: its body emitted on a wall-clock window and slept one whole interval
    after every emit (which cannot reduce the flush count at the carrier's
    decode cadence and only adds delay), and no route ever called it.  This
    version implements the engine's own semantics -- always emit the first
    event, then flush every ``batch_size`` accumulated events, with no inserted
    sleep -- so the flush count falls for long streams while the time of the
    first and last byte is unchanged.
    """

    if batch_size <= 1 and decision_id is None:
        async for chunk in generator:
            yield chunk if isinstance(chunk, bytes) else str(chunk).encode()
        return
    pending: list[bytes] = []
    sent_events = 0
    frame_index = 0
    decoded_event_offset = 0
    response_id: str | None = None
    body_hasher = hashlib.sha256()
    async for chunk in generator:
        pending.append(chunk if isinstance(chunk, bytes) else str(chunk).encode())
        if sent_events == 0 or len(pending) >= batch_size:
            frame = b"".join(pending)
            if decision_id is not None:
                decoded_event_offset, response_id = _record_stream_carrier_frame(
                    decision_id, batch_size, frame, frame_index, decoded_event_offset,
                    response_id, body_hasher)
            frame_index += 1
            yield frame
            sent_events += len(pending)
            pending.clear()
    if pending:
        frame = b"".join(pending)
        if decision_id is not None:
            _record_stream_carrier_frame(
                decision_id, batch_size, frame, frame_index, decoded_event_offset,
                response_id, body_hasher)
        yield frame


__all__ = [
    "MECHANISMS",
    "provider_context_snapshot",
    "record_request_pre_action",
    "state",
    "delivery_interval_ms",
    "stream_batch_size",
    "batch_sse",
]
