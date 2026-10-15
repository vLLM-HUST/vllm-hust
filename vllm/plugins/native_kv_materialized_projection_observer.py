"""Read the actual materialized Host row after its native scheduler update."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .native_kv_projection_observer import OPERATION, SCOPE

SCHEMA = "ecpa-native-kv-blocktable-projection/v2"


@dataclass(frozen=True)
class ProjectionMarker:
    pending: object
    appended_block_ids: tuple[int, ...]
    metadata_effect_monotonic_ns: int


def capture_marker(provider, scheduler_output, request_id, pending):
    """Called only after actual Request/active mutation, before native append."""
    cached = scheduler_output.scheduled_cached_reqs
    slots = [i for i, value in enumerate(cached.req_ids) if value == request_id]
    if len(slots) != 1 or request_id not in scheduler_output.num_scheduled_tokens:
        raise ValueError("applied projection lacks exact native cached request")
    group = provider.attention_group_index
    new = cached.new_block_ids[slots[0]]
    appended = tuple(new[group]) if new is not None else ()
    request = provider.runner.requests[request_id]
    active = provider.active[request_id]
    if (
        list(request.block_ids[group]) != list(pending.block_ids)
        or active.semantic_anchor != pending.semantic_anchor
        or active.physical_anchor != pending.physical_anchor
        or any(type(x) is not int or x < 0 for x in appended)
    ):
        raise ValueError("marker does not follow actual applied metadata")
    return ProjectionMarker(pending, appended, time.monotonic_ns())


def projection_fact(provider, scheduler_output, request_id, marker, native_scope):
    if request_id not in scheduler_output.num_scheduled_tokens:
        raise ValueError("materialized projection request is not scheduled")
    runner = provider.runner
    row = runner.input_batch.req_id_to_index.get(request_id)
    if row is None:
        raise ValueError("actual Host CPU row is not materialized")
    pending = marker.pending
    group = provider.attention_group_index
    table = runner.input_batch.block_table.block_tables[group]
    width = int(table.num_blocks_per_row[row])
    observed = list(map(int, table.get_numpy_array()[row, :width]))
    destination = list(pending.block_ids)
    appended = list(marker.appended_block_ids)
    expected_full = destination + appended
    request_full = list(runner.requests[request_id].block_ids[group])
    active = provider.active[request_id]
    block_size = table.block_size
    threshold = provider.runtime_spec.compression_threshold_tokens
    if (
        not destination
        or width != len(expected_full)
        or observed != expected_full
        or request_full != observed
        or len(set(expected_full)) != len(expected_full)
        or any(type(x) is not int or x < 0 for x in expected_full)
        or active.semantic_anchor != pending.semantic_anchor
        or active.physical_anchor != pending.physical_anchor
        or type(block_size) is not int
        or block_size <= 0
        or pending.physical_anchor != len(destination) * block_size
        or type(threshold) is not int
        or not 0 < pending.physical_anchor < threshold <= pending.semantic_anchor
        or not marker.metadata_effect_monotonic_ns <= native_scope["monotonic_ns"]
    ):
        raise ValueError(
            "materialized Host row differs from actual destination/append/anchors"
        )
    transaction = {
        "runtime_request_id": request_id,
        "attention_group": group,
        "semantic_anchor": pending.semantic_anchor,
        "physical_anchor": pending.physical_anchor,
        "destination_block_ids": destination,
        "host_cpu_block_ids": observed,
        "allocation_block_size": block_size,
        "compression_threshold_tokens": threshold,
        "scheduled_appended_block_ids": appended,
        "metadata_effect_monotonic_ns": marker.metadata_effect_monotonic_ns,
    }
    transaction_id = hashlib.sha256(
        json.dumps(
            transaction, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        **native_scope,
        **transaction,
        "metadata_row": row,
        "transaction_id": transaction_id,
        "operation": OPERATION,
        "scope": SCOPE,
        "host_mapping": "identity-host-block-id-projection-with-scheduled-append",
        "kv_tensor_equivalence": None,
        "attention_consumed_observed_tensor": None,
        "full_generation_quality": None,
    }


def observe_materialized_projection(provider, scheduler_output, request_id, marker):
    from . import native_block_table_observer as native

    native.emit_native_fact(
        lambda: projection_fact(
            provider, scheduler_output, request_id, marker, native._scope("worker")
        )
    )
