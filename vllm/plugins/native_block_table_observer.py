"""Candidate Host-native facts. Opt-in; upload observation is not KV equivalence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

_observer = None


def register_observer(observer):
    """Register one source-pinned trusted observer in this native process."""
    global _observer
    if _observer is not None:
        raise RuntimeError("native observer already registered")
    if not callable(observer):
        raise TypeError("observer must be callable")
    from .native_phase_drain import callback_with_drain
    _observer = callback_with_drain(observer)


def _scope(role):
    from vllm.plugins.evidence import _identity

    identity = _identity()
    if identity["assignment_source"] != "host" or identity["role"] != role:
        raise RuntimeError("native Host-assigned observer role required")
    launch = os.environ.get("VLLM_ECPA_LAUNCH_ID")
    plan = os.environ.get("VLLM_ECPA_PLAN_ID")
    if not launch or not plan:
        raise RuntimeError("native plan/launch binding missing")
    return {
        "process": {
            **identity,
            "session": os.getsid(0),
            "start_ticks": int(identity["start_identity"].rsplit(":", 1)[1]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
        "launch_id": launch,
        "plan_id": plan,
        "monotonic_ns": time.monotonic_ns(),
    }


def observe_request_id_mapping(request):
    if _observer is None:
        from .native_semantic_bootstrap import ensure_registered
        ensure_registered(register_observer)
    if _observer is None:
        return
    if not request.external_req_id or not request.request_id:
        raise RuntimeError("native external/internal request IDs missing")
    _observer(
        {
            "schema": "ecpa-native-request-id-mapping/v1",
            **_scope("api-server"),
            "external_request_id": request.external_req_id,
            "runtime_request_id": request.request_id,
        }
    )


def observe_block_table_upload(runner, scheduler_output):
    if _observer is None:
        from .native_semantic_bootstrap import ensure_registered
        ensure_registered(register_observer)
    if _observer is None:
        return
    # Imported only for an opted-in observation. No global device sync, dummy
    # graph interception or model/KV-tensor correctness assertion.
    import torch

    batch = runner.input_batch
    tables = batch.block_table.block_tables
    if not tables:
        raise RuntimeError("native block-table groups missing")
    devices = {str(table.block_table.gpu.device) for table in tables}
    if len(devices) != 1 or tables[0].block_table.gpu.device.type != "npu":
        raise RuntimeError("real same-device NPU block tables required")
    scope = _scope("worker")
    # commit_block_table queues nonblocking H2D copies on this stream. This
    # deliberately costly qualification barrier is not a production fast path.
    torch.npu.current_stream(tables[0].block_table.gpu.device).synchronize()
    scheduled = scheduler_output.num_scheduled_tokens
    for row, request_id in enumerate(batch.req_ids[: batch.num_reqs]):
        if scheduled.get(request_id, 0) <= 0:
            continue
        observed = []
        for group_id, table in enumerate(tables):
            width = int(table.num_blocks_per_row[row])
            if width <= 0 or width > table.block_table.gpu.shape[1]:
                raise RuntimeError("invalid active native block-table prefix")
            cpu = table.block_table.np[row, :width].tolist()
            device = table.block_table.gpu[row, :width]
            gpu = device.to(device="cpu", non_blocking=False).tolist()
            if cpu != gpu:
                raise RuntimeError("uploaded NPU block-table prefix differs")
            observed.append({"group": group_id, "row": row, "blocks": gpu})
        payload = json.dumps(observed, sort_keys=True, separators=(",", ":"))
        _observer(
            {
                "schema": "ecpa-native-block-table-upload/v1",
                **scope,
                "observed_after_barrier_monotonic_ns": time.monotonic_ns(),
                "runtime_request_id": request_id,
                "device": next(iter(devices)),
                "uploaded_rows": observed,
                "uploaded_rows_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "scope": "NPU-block-table-prefix-readback-only",
                "kv_tensor_equivalence": None,
                "attention_consumed_observed_tensor": None,
                "full_generation_quality": None,
            }
        )


def emit_native_fact(factory):
    """Lazy source-owned extension fact; authentication/ACK failures propagate."""
    if _observer is None:
        from .native_semantic_bootstrap import ensure_registered

        ensure_registered(register_observer)
    if _observer is not None:
        return _observer(factory())
