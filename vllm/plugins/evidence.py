# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional delivery of host-observed events from existing plugin paths.

General Python plugins are trusted in-process extensions. These observations
describe loader control flow for non-adversarial plugins; they do not provide
integrity against a malicious plugin running in the same interpreter.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import socket
import time
from collections.abc import Callable
from typing import Any, Literal

logger = logging.getLogger(__name__)
SCHEMA = "vllm-hust-plugin-evidence/0.1"
SinkState = Literal["unconfigured", "ready", "failed"]

_sink: Callable[[dict[str, Any]], None] | None = None
_sink_state: SinkState = "unconfigured"
_sink_error: Exception | None = None
_sink_failures = 0
_delivery_attempts = 0
_delivery_count = 0
_delivered: set[tuple[Any, ...]] = set()
_owner_pid = os.getpid()


class EvidenceConfigurationError(RuntimeError):
    """Evidence configuration or identity is invalid."""


def _strict() -> bool:
    value = os.getenv("VLLM_ECPA_EVIDENCE_STRICT", "0")
    if value not in {"0", "1"}:
        raise EvidenceConfigurationError("VLLM_ECPA_EVIDENCE_STRICT must be 0 or 1")
    return value == "1"


def _start_identity() -> str:
    """Read Linux field 22 without splitting the parenthesized comm field."""
    try:
        with open("/proc/self/stat") as stat_file:
            raw = stat_file.read()
        close = raw.rfind(")")
        if close < 0:
            raise ValueError("missing comm terminator")
        fields_after_comm = raw[close + 1 :].split()
        start_ticks = fields_after_comm[19]
        int(start_ticks)
    except (OSError, IndexError, ValueError) as exc:
        raise EvidenceConfigurationError(
            "process start identity is unavailable"
        ) from exc
    return f"pid:{os.getpid()}:start_ticks:{start_ticks}"


def _identity() -> dict[str, Any]:
    try:
        start_identity = _start_identity()
        host = socket.gethostname()
        if not host:
            raise ValueError("empty hostname")
        role = os.getenv("VLLM_ECPA_PROCESS_ROLE", "unknown")
        ordinal = int(os.getenv("VLLM_ECPA_PROCESS_ORDINAL", "0"))
        default_epoch = int(
            hashlib.sha256(start_identity.encode()).hexdigest()[:12], 16
        )
        epoch = int(os.getenv("VLLM_ECPA_PROCESS_EPOCH", str(default_epoch)))
        if ordinal < 0 or epoch < 0 or not role:
            raise ValueError("negative epoch/ordinal or empty role")
    except (OSError, ValueError) as exc:
        raise EvidenceConfigurationError("invalid process identity") from exc
    return {
        "host": host,
        "role": role,
        "ordinal": ordinal,
        "pid": os.getpid(),
        "start_identity": start_identity,
        "process_epoch": epoch,
    }


def _reset_after_fork() -> None:
    global _owner_pid, _sink, _sink_state, _sink_error
    current_pid = os.getpid()
    if current_pid == _owner_pid:
        return
    _owner_pid = current_pid
    _sink = None
    _sink_state = "unconfigured"
    _sink_error = None
    _delivered.clear()


def _record_failure(message: str, exc: Exception, *args: Any) -> None:
    global _sink_failures
    _sink_failures += 1
    logger.error(message, *args, _sink_failures, exc_info=exc)


def _load_sink(strict: bool) -> Callable[[dict[str, Any]], None] | None:
    global _sink, _sink_state, _sink_error
    _reset_after_fork()
    if _sink_state == "ready":
        return _sink
    if _sink_state == "failed":
        assert _sink_error is not None
        raise EvidenceConfigurationError("evidence sink is failed") from _sink_error
    target = os.getenv("VLLM_ECPA_EVIDENCE_SINK")
    if not target:
        return None
    try:
        module_name, attribute = target.split(":", 1)
        if not module_name or not attribute:
            raise ValueError("expected module:callable")
        candidate = getattr(importlib.import_module(module_name), attribute)
        if not callable(candidate):
            raise TypeError("sink is not callable")
    except Exception as exc:
        _record_failure(
            "ECPA_EVIDENCE_SINK_LOAD_FAILED target=%s failures=%d", exc, target
        )
        if strict:
            _sink_state, _sink_error = "failed", exc
            raise EvidenceConfigurationError("evidence sink load failed") from exc
        # Compatibility mode retries configuration on the next event.
        _sink, _sink_state = None, "unconfigured"
        return None
    _sink, _sink_state, _sink_error = candidate, "ready", None
    return candidate


def _event_key(
    identity: dict[str, Any],
    event: str,
    group: str,
    name: str,
    value: str,
    detail: str | None,
) -> tuple[Any, ...]:
    return (
        identity["host"],
        identity["pid"],
        identity["start_identity"],
        identity["process_epoch"],
        identity["role"],
        identity["ordinal"],
        group,
        name,
        value,
        event,
        detail,
    )


def _emit_host_event(
    event: str,
    group: str,
    name: str,
    value: str,
    *,
    detail: str | None = None,
) -> bool:
    """Attempt delivery; return true only after the configured sink accepts."""
    global _delivery_attempts, _delivery_count, _sink_state, _sink_error
    try:
        strict = _strict()
    except EvidenceConfigurationError as exc:
        _record_failure("ECPA_EVIDENCE_CONFIG_FAILED failures=%d", exc)
        _sink_state, _sink_error = "failed", exc
        raise
    try:
        sink = _load_sink(strict)
        if sink is None:
            return False
        identity = _identity()
    except EvidenceConfigurationError as exc:
        if not (_sink_state == "failed" and _sink_error is not None):
            _record_failure("ECPA_EVIDENCE_IDENTITY_FAILED failures=%d", exc)
        if strict:
            _sink_state, _sink_error = "failed", exc
            raise EvidenceConfigurationError("evidence emission failed") from exc
        return False

    key = _event_key(identity, event, group, name, value, detail)
    if key in _delivered:
        return True
    _delivery_attempts += 1
    observed_at_ns = time.time_ns()
    event_material = json.dumps(
        [*key, _delivery_attempts, observed_at_ns], separators=(",", ":")
    ).encode()
    payload = {
        "schema": SCHEMA,
        "event_id": hashlib.sha256(event_material).hexdigest(),
        "event": event,
        "entry_point": {"group": group, "name": name, "value": value},
        "process": identity,
        "observed_at_ns": observed_at_ns,
        "delivery_attempt": _delivery_attempts,
        "plan_id": os.getenv("VLLM_ECPA_PLAN_ID"),
        "launch_id": os.getenv("VLLM_ECPA_LAUNCH_ID"),
        "plugin_id": None,
        "artifact_digest": None,
        "identity_status": "absent; bind from the ECPA Plan at trusted ingestion",
        "detail": detail,
    }
    try:
        sink(payload)
    except Exception as exc:
        _record_failure(
            "ECPA_EVIDENCE_SINK_WRITE_FAILED event=%s group=%s name=%s failures=%d",
            exc,
            event,
            group,
            name,
        )
        if strict:
            _sink_state, _sink_error = "failed", exc
            raise EvidenceConfigurationError("evidence sink write failed") from exc
        # Compatibility mode keeps a ready sink and retries this event later.
        return False
    _delivered.add(key)
    _delivery_count += 1
    return True


def reset_for_tests() -> None:
    global _sink, _sink_state, _sink_error, _sink_failures
    global _delivery_attempts, _delivery_count, _owner_pid
    _sink, _sink_state, _sink_error = None, "unconfigured", None
    _sink_failures, _delivery_attempts, _delivery_count = 0, 0, 0
    _owner_pid = os.getpid()
    _delivered.clear()
