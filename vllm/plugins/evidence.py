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
import threading
import time
from collections.abc import Callable
from typing import Any, Literal, TypedDict

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
_state_lock = threading.RLock()
_controller_sequence = 0
_invocation_sequence = 0
_process_identity_binding: tuple[str, int] | None = None


class EvidenceConfigurationError(RuntimeError):
    """Evidence configuration or identity is invalid."""


class EvidenceScope(TypedDict):
    """Process identity and launch binding frozen at component initialization."""

    process: dict[str, Any]
    plan_id: str | None
    launch_id: str | None
    binding_status: Literal["bound", "unbound"]


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
    _ensure_process_state()
    try:
        start_identity = _start_identity()
        host = socket.gethostname()
        if not host:
            raise ValueError("empty hostname")
        default_epoch = int(
            hashlib.sha256(start_identity.encode()).hexdigest()[:12], 16
        )
        if _process_identity_binding is None:
            role = os.getenv("VLLM_ECPA_PROCESS_ROLE", "unknown")
            ordinal = int(os.getenv("VLLM_ECPA_PROCESS_ORDINAL", "0"))
            epoch = int(os.getenv("VLLM_ECPA_PROCESS_EPOCH", str(default_epoch)))
            assignment_source = "environment"
        else:
            role, ordinal = _process_identity_binding
            epoch = default_epoch
            assignment_source = "host"
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
        "assignment_source": assignment_source,
    }


def bind_process_identity(role: str, ordinal: int) -> None:
    """Freeze the host-assigned role and ordinal for this process instance."""
    global _process_identity_binding
    _ensure_process_state()
    if (
        not isinstance(role, str)
        or not role
        or role.strip() != role
        or not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or ordinal < 0
    ):
        raise EvidenceConfigurationError("invalid host process identity")
    binding = (role, ordinal)
    with _state_lock:
        if _process_identity_binding not in (None, binding):
            raise EvidenceConfigurationError("host process identity is already bound")
        _process_identity_binding = binding


def capture_scope(*, expected_role: str | None = None) -> EvidenceScope:
    """Capture identity and launch binding once for a long-lived component."""
    identity = _identity()
    if expected_role is not None and identity["role"] != expected_role:
        raise EvidenceConfigurationError(
            f"evidence role must be {expected_role!r}, got {identity['role']!r}"
        )
    plan_id = os.getenv("VLLM_ECPA_PLAN_ID") or None
    launch_id = os.getenv("VLLM_ECPA_LAUNCH_ID") or None
    return {
        "process": identity,
        "plan_id": plan_id,
        "launch_id": launch_id,
        "binding_status": (
            "bound" if plan_id is not None and launch_id is not None else "unbound"
        ),
    }


def observer_configured() -> bool:
    """Return whether this process has an explicit evidence destination."""
    _ensure_process_state()
    with _state_lock:
        return _sink_state == "ready" or bool(os.getenv("VLLM_ECPA_EVIDENCE_SINK"))


def _ensure_process_state() -> None:
    """Discard inherited mutable state before a child process can use it."""
    global _owner_pid, _sink, _sink_state, _sink_error, _state_lock
    global _controller_sequence, _invocation_sequence, _process_identity_binding
    current_pid = os.getpid()
    if current_pid == _owner_pid:
        return
    # A lock may be inherited while held by a vanished parent thread.
    _state_lock = threading.RLock()
    _owner_pid = current_pid
    _sink = None
    _sink_state = "unconfigured"
    _sink_error = None
    _controller_sequence = 0
    _invocation_sequence = 0
    _process_identity_binding = None
    _delivered.clear()


def scope_is_current(scope: EvidenceScope) -> bool:
    """Return whether a frozen scope belongs to this exact process instance."""
    return bool(
        scope["process"]["pid"] == os.getpid()
        and scope["process"]["start_identity"] == _start_identity()
    )


def _require_current_scope(scope: EvidenceScope) -> None:
    _ensure_process_state()
    if not scope_is_current(scope):
        raise EvidenceConfigurationError("evidence scope belongs to another process")


def allocate_controller_instance(scope: EvidenceScope) -> str:
    """Allocate a process-wide controller identity under the frozen launch."""
    global _controller_sequence
    _require_current_scope(scope)
    with _state_lock:
        _controller_sequence += 1
        material = json.dumps(
            [
                scope["process"]["host"],
                scope["process"]["start_identity"],
                scope["plan_id"],
                scope["launch_id"],
                _controller_sequence,
            ],
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(material).hexdigest()


def allocate_invocation_sequence(scope: EvidenceScope) -> int:
    """Allocate a process-wide dispatch sequence, shared by all controllers."""
    global _invocation_sequence
    _require_current_scope(scope)
    with _state_lock:
        _invocation_sequence += 1
        return _invocation_sequence


def _record_failure(message: str, exc: Exception, *args: Any) -> None:
    global _sink_failures
    _sink_failures += 1
    logger.error(message, *args, _sink_failures, exc_info=exc)


def _load_sink(strict: bool) -> Callable[[dict[str, Any]], None] | None:
    global _sink, _sink_state, _sink_error
    _ensure_process_state()
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
    occurrence_id: int | str | None,
    plan_id: str | None,
    launch_id: str | None,
    observation_kind: str,
    controller_instance_id: str | None,
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
        occurrence_id,
        plan_id,
        launch_id,
        observation_kind,
        controller_instance_id,
    )


def _dispatch_id(
    identity: dict[str, Any],
    plan_id: str | None,
    launch_id: str | None,
    controller_instance_id: str | None,
    occurrence_id: int | str | None,
) -> str | None:
    if not isinstance(occurrence_id, int):
        return None
    material = json.dumps(
        [
            identity["host"],
            identity["start_identity"],
            identity["process_epoch"],
            plan_id,
            launch_id,
            controller_instance_id,
            occurrence_id,
        ],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _emit_host_event_unlocked(
    event: str,
    group: str,
    name: str,
    value: str,
    *,
    detail: str | None = None,
    scope: EvidenceScope | None = None,
    occurrence_id: int | str | None = None,
    observation_kind: str = "loader_lifecycle",
    controller_instance_id: str | None = None,
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
        if scope is not None:
            _require_current_scope(scope)
        identity = _identity() if scope is None else scope["process"]
    except EvidenceConfigurationError as exc:
        if not (_sink_state == "failed" and _sink_error is not None):
            _record_failure("ECPA_EVIDENCE_IDENTITY_FAILED failures=%d", exc)
        if strict:
            _sink_state, _sink_error = "failed", exc
            raise EvidenceConfigurationError("evidence emission failed") from exc
        return False

    plan_id = os.getenv("VLLM_ECPA_PLAN_ID") if scope is None else scope["plan_id"]
    launch_id = (
        os.getenv("VLLM_ECPA_LAUNCH_ID") if scope is None else scope["launch_id"]
    )
    binding_status = (
        "bound"
        if plan_id is not None and launch_id is not None
        else "unbound"
        if scope is None
        else scope["binding_status"]
    )
    key = _event_key(
        identity,
        event,
        group,
        name,
        value,
        detail,
        occurrence_id,
        plan_id,
        launch_id,
        observation_kind,
        controller_instance_id,
    )
    delivery_key = (*key, identity["assignment_source"])
    if delivery_key in _delivered:
        return True
    _delivery_attempts += 1
    observed_at_ns = time.time_ns()
    event_material = json.dumps(
        [
            *key,
            _delivery_attempts,
            observed_at_ns,
            identity["assignment_source"],
        ],
        separators=(",", ":"),
    ).encode()
    payload = {
        "schema": SCHEMA,
        "event_id": hashlib.sha256(event_material).hexdigest(),
        "event": event,
        "observation_kind": observation_kind,
        "entry_point": {"group": group, "name": name, "value": value},
        "process": identity,
        "observed_at_ns": observed_at_ns,
        "delivery_attempt": _delivery_attempts,
        "plan_id": plan_id,
        "launch_id": launch_id,
        "binding_status": binding_status,
        "occurrence_id": occurrence_id,
        "controller_instance_id": controller_instance_id,
        "invocation_seq": occurrence_id if isinstance(occurrence_id, int) else None,
        "dispatch_id": _dispatch_id(
            identity,
            plan_id,
            launch_id,
            controller_instance_id,
            occurrence_id,
        ),
        "plugin_id": None,
        "artifact_digest": None,
        "identity_status": (
            "launch-bound"
            if binding_status == "bound"
            else "unbound; not eligible for translated ECPA evidence"
        ),
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
        _sink_state, _sink_error = "failed", exc
        if strict:
            raise EvidenceConfigurationError("evidence sink write failed") from exc
        # A broken hot-path sink remains failed; later events short-circuit.
        return False
    _delivered.add(delivery_key)
    _delivery_count += 1
    return True


def _emit_host_event(
    event: str,
    group: str,
    name: str,
    value: str,
    *,
    detail: str | None = None,
    scope: EvidenceScope | None = None,
    occurrence_id: int | str | None = None,
    observation_kind: str = "loader_lifecycle",
    controller_instance_id: str | None = None,
) -> bool:
    """Serialize sink, delivery, and dedupe state across observer threads."""
    _ensure_process_state()
    with _state_lock:
        return _emit_host_event_unlocked(
            event,
            group,
            name,
            value,
            detail=detail,
            scope=scope,
            occurrence_id=occurrence_id,
            observation_kind=observation_kind,
            controller_instance_id=controller_instance_id,
        )


def reset_for_tests() -> None:
    global _sink, _sink_state, _sink_error, _sink_failures
    global _delivery_attempts, _delivery_count, _owner_pid, _state_lock
    global _controller_sequence, _invocation_sequence, _process_identity_binding
    _sink, _sink_state, _sink_error = None, "unconfigured", None
    _sink_failures, _delivery_attempts, _delivery_count = 0, 0, 0
    _owner_pid = os.getpid()
    _state_lock = threading.RLock()
    _controller_sequence, _invocation_sequence = 0, 0
    _process_identity_binding = None
    _delivered.clear()
