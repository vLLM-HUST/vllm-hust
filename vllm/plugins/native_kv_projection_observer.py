# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source candidate for actual KVCompress Host metadata, not tensor equivalence."""

from __future__ import annotations

import hashlib
import json

SCHEMA = "ecpa-native-kv-blocktable-projection/v1"
OPERATION = "KVCompress-worker-host-blocktable-projection-applied"
SCOPE = "compression-host-metadata-and-uploaded-destination-binding-only"


def projection_fact(provider, scheduler_output, request_id, pending, native_scope):
    """Read after the acknowledged barrier and actual metadata mutations.

    The pinned Host table uses identity block IDs; an expanded/unknown mapping
    is unsupported. No successful-effect flag or caller-supplied observed row.
    """
    if request_id not in scheduler_output.num_scheduled_tokens:
        raise ValueError("projection request lacks acknowledged scheduled barrier")
    runner = provider.runner
    request = runner.requests[request_id]
    group = provider.attention_group_index
    row = runner.input_batch.req_id_to_index[request_id]
    table = runner.input_batch.block_table.block_tables[group]
    destination = list(pending.block_ids)
    width = int(table.num_blocks_per_row[row])
    observed = list(map(int, table.get_numpy_array()[row, :width]))
    active = provider.active[request_id]
    block_size = table.block_size
    threshold = provider.runtime_spec.compression_threshold_tokens
    if (
        not destination
        or width != len(destination)
        or list(request.block_ids[group]) != destination
        or observed != destination
        or active.semantic_anchor != pending.semantic_anchor
        or active.physical_anchor != pending.physical_anchor
        or type(block_size) is not int
        or block_size <= 0
        or pending.physical_anchor != len(destination) * block_size
        or type(threshold) is not int
        or not 0 < pending.physical_anchor < threshold <= pending.semantic_anchor
    ):
        raise ValueError("actual Host projection/destination/anchors are inconsistent")
    transaction = {
        "runtime_request_id": request_id,
        "attention_group": group,
        "semantic_anchor": pending.semantic_anchor,
        "physical_anchor": pending.physical_anchor,
        "destination_block_ids": destination,
        "host_cpu_block_ids": observed,
        "allocation_block_size": block_size,
        "compression_threshold_tokens": threshold,
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
        "host_mapping": "identity-host-block-id-projection",
        "kv_tensor_equivalence": None,
        "attention_consumed_observed_tensor": None,
        "full_generation_quality": None,
    }


def observe_kv_projection(provider, scheduler_output, request_id, pending):
    """New clean Host candidate hook; absent native observer emits no proof."""
    from . import native_block_table_observer as native

    # The public interface installs a source-owned bootstrap and throws on a
    # rejected/authentication/ACK failure. Never return a success boolean.
    native.emit_native_fact(
        lambda: projection_fact(
            provider, scheduler_output, request_id, pending, native._scope("worker")
        )
    )
