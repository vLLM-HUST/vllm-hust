"""Prospective source-owned worker drain; candidate only, never formal admission."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

SCHEMA = "ecpa-native-worker-request-drained/v1"
ENV = "ECPA_NATIVE_PHASE_DRAIN"
_records = {}
_closed = set()
_lock = threading.RLock()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def record_ack(runtime_request_id, ack):
    """Record an actual synchronous intake ACK, never an emitted Boolean."""
    if os.environ.get(ENV) != "1":
        return
    if runtime_request_id in _closed:
        raise ValueError("native fact after source-owned request drain")
    fields = ("sequence", "event_sha256", "raw_sha256")
    if not isinstance(ack, dict) or any(k not in ack for k in fields):
        raise ValueError("native drain requires actual authenticated intake ACK")
    item = {k: ack[k] for k in fields}
    if type(item["sequence"]) is not int or item["sequence"] < 0:
        raise ValueError("native drain ACK sequence is unsupported")
    if any(type(item[k]) is not str or len(item[k]) != 64 for k in fields[1:]):
        raise ValueError("native drain ACK digests are unsupported")
    records = _records.setdefault(runtime_request_id, [])
    if records and item["sequence"] <= records[-1]["sequence"]:
        raise ValueError("native drain ACK order/replay differs")
    records.append(item)


def acknowledged_publication(runtime_request_id, publish):
    """Serialize actual pending transports with source closure, not a timer."""
    if os.environ.get(ENV) != "1":
        return publish()
    with _lock:
        if runtime_request_id in _closed:
            raise ValueError("native publication after source-owned request drain")
        ack = publish()
        record_ack(runtime_request_id, ack)
        return ack


def callback_with_drain(callback):
    def wrapped(fact):
        if (
            fact.get("process", {}).get("role") == "worker"
            and fact.get("schema") != SCHEMA
        ):
            return acknowledged_publication(
                fact["runtime_request_id"], lambda: callback(fact)
            )
        return callback(fact)

    return wrapped


def _after_finished_request_removal(runner, scheduler_output, *, scope, publish):
    """Called after pinned Host removal, before the zero-token early return.

    Worker execution and all three receipt transports are synchronous. No quiet
    interval, owner-provided count, response-derived token fence or guessed row.
    Resubmitted IDs cannot drain. Removal may follow finish or abort; this
    marker proves transport closure, while strict HTTP/task success is separate.
    """
    if os.environ.get(ENV) != "1":
        return
    for runtime in sorted(scheduler_output.finished_req_ids):
        if runtime not in _records:
            continue  # warmup / requests without a candidate source registration
        if runtime in _closed:
            raise ValueError("native request drain is replayed")
        if (
            runtime in runner.requests
            or runtime in runner.input_batch.req_id_to_index
            or runtime in scheduler_output.num_scheduled_tokens
            or any(x.req_id == runtime for x in scheduler_output.scheduled_new_reqs)
        ):
            raise ValueError(
                "actual native request has not been removed or was resubmitted"
            )
        records = _records[runtime]
        fact = {
            "schema": SCHEMA,
            **scope("worker"),
            "runtime_request_id": runtime,
            "source_ack_count": len(records),
            "source_ack_chain_sha256": digest(records),
            "source_last_ack": records[-1],
            "native_requests_contains": False,
            "native_input_batch_contains": False,
            "scheduled_finished": True,
            "source_drain_completed_monotonic_ns": time.monotonic_ns(),
            "formal_producer_registration": False,
        }
        # The publication callback must synchronously verify the collector ACK.
        ack = publish(fact)
        if not isinstance(ack, dict) or ack.get("raw_sha256") != digest(fact):
            raise ValueError("native source drain publication lacks actual raw ACK")
        _closed.add(runtime)


def after_finished_request_removal(runner, scheduler_output, *, scope, publish):
    with _lock:
        return _after_finished_request_removal(
            runner, scheduler_output, scope=scope, publish=publish
        )
