# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional host-owned lifecycle evidence for existing plugin paths.

Disabled by default. This module observes the existing loader; it is not a
loader and never lets a plugin self-assert invocation.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import socket
import time
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)
SCHEMA = "vllm-hust-plugin-evidence/0.1"
_sink: Callable[[dict[str, Any]], None] | None = None
_sink_loaded = False
_sink_failures = 0
_emitted: set[tuple[str, str, str, str]] = set()


def _start_identity() -> str:
    try:
        start_ticks = open("/proc/self/stat").read().split()[21]  # noqa: SIM115
    except (OSError, IndexError):
        start_ticks = "unknown"
    return f"pid:{os.getpid()}:start_ticks:{start_ticks}"


def _identity() -> dict[str, Any]:
    start_identity = _start_identity()
    default_epoch = int(hashlib.sha256(start_identity.encode()).hexdigest()[:12], 16)
    return {
        "host": socket.gethostname(),
        "role": os.getenv("VLLM_ECPA_PROCESS_ROLE", "unknown"),
        "ordinal": int(os.getenv("VLLM_ECPA_PROCESS_ORDINAL", "0")),
        "pid": os.getpid(),
        "start_identity": start_identity,
        "process_epoch": int(os.getenv("VLLM_ECPA_PROCESS_EPOCH", str(default_epoch))),
    }


def _load_sink() -> Callable[[dict[str, Any]], None] | None:
    global _sink, _sink_loaded, _sink_failures
    if _sink_loaded:
        return _sink
    _sink_loaded = True
    target = os.getenv("VLLM_ECPA_EVIDENCE_SINK")
    if not target:
        return None
    try:
        module_name, attribute = target.split(":", 1)
        candidate = getattr(importlib.import_module(module_name), attribute)
        if not callable(candidate):
            raise TypeError("sink is not callable")
        _sink = candidate
    except Exception:
        _sink_failures += 1
        logger.exception("ECPA_EVIDENCE_SINK_LOAD_FAILED target=%s", target)
        if os.getenv("VLLM_ECPA_EVIDENCE_STRICT") == "1":
            raise
    return _sink


def emit(
    event: str,
    group: str,
    name: str,
    value: str,
    *,
    detail: str | None = None,
) -> None:
    """Emit one host-observed event; sink failures are visible and counted."""
    global _sink_failures
    sink = _load_sink()
    if sink is None:
        return
    identity = _identity()
    dedupe_key = (identity["process_epoch"], event, group, name)
    if dedupe_key in _emitted:
        return
    _emitted.add(dedupe_key)
    observed_at = time.time_ns()
    payload = {
        "schema": SCHEMA,
        "event_id": str(uuid.uuid4()),
        "event": event,
        "entry_point": {"group": group, "name": name, "value": value},
        "process": identity,
        "observed_at_ns": observed_at,
        "plan_id": os.getenv("VLLM_ECPA_PLAN_ID"),
        "launch_id": os.getenv("VLLM_ECPA_LAUNCH_ID"),
        "plugin_id": None,
        "artifact_digest": None,
        "identity_status": "absent; bind from the ECPA Plan at trusted ingestion",
        "detail": detail,
    }
    try:
        sink(payload)
    except Exception:
        _sink_failures += 1
        logger.exception(
            "ECPA_EVIDENCE_SINK_WRITE_FAILED event=%s group=%s name=%s failures=%d",
            event,
            group,
            name,
            _sink_failures,
        )
        if os.getenv("VLLM_ECPA_EVIDENCE_STRICT") == "1":
            raise


def reset_for_tests() -> None:
    global _sink, _sink_loaded, _sink_failures
    _sink, _sink_loaded, _sink_failures = None, False, 0
    _emitted.clear()
