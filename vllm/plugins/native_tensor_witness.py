# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selected-source byte witness candidate. No source registration or GPU result."""

from __future__ import annotations

import hashlib
import time
from typing import Any, cast

import pybase64 as base64

SCHEMA = "ecpa-selected-KV-materialization-source-candidate/v1"
_sink: Any = None


def register_candidate_sink(callback):
    """CPU/source candidate only: callback return is never a durable/formal ACK."""
    global _sink
    if _sink is not None or not callable(callback):
        raise ValueError("candidate sink must be a single callable")
    _sink = callback


def enabled():
    return _sink is not None


def check(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def slots(blocks, indices, block_size):
    check(type(block_size) is int and block_size > 0, "invalid block size")
    check(
        blocks
        and all(type(x) is int and x >= 0 for x in blocks)
        and len(set(blocks)) == len(blocks),
        "invalid or aliased blocks",
    )
    check(
        indices
        and all(type(x) is int and 0 <= x < len(blocks) * block_size for x in indices),
        "invalid token indices",
    )
    return [blocks[x // block_size] * block_size + x % block_size for x in indices]


def verify(fact):
    check(
        fact["schema"] == SCHEMA
        and fact["classification"] == "CPU_SOURCE_CANDIDATE_ONLY"
        and fact["formal_registration"] is False
        and fact["GPU_qualification"] is None
        and fact["attention_consumption"] is None
        and fact["full_model_quality"] is None,
        "candidate scope cannot become GPU/attention/formal evidence",
    )
    binding = fact["binding"]
    check(
        isinstance(binding["runtime_request_id"], str)
        and binding["runtime_request_id"]
        and isinstance(binding["layer_name"], str)
        and binding["layer_name"]
        and type(binding["layer_index"]) is int
        and binding["layer_index"] >= 0
        and type(binding["tensor_parallel_size"]) is int
        and binding["tensor_parallel_size"] > 0
        and type(binding["rank"]) is int
        and 0 <= binding["rank"] < binding["tensor_parallel_size"]
        and isinstance(binding["expected_layer_indices"], list)
        and binding["expected_layer_indices"]
        and all(type(x) is int and x >= 0 for x in binding["expected_layer_indices"])
        and binding["layer_index"] in binding["expected_layer_indices"]
        and len(set(binding["expected_layer_indices"]))
        == len(binding["expected_layer_indices"]),
        "actual request/layer binding missing",
    )
    keep = fact["keep_indices"]
    check(
        type(binding["physical_num_tokens"]) is int
        and binding["physical_num_tokens"] > 0
        and type(binding["semantic_num_tokens"]) is int
        and binding["semantic_num_tokens"] >= binding["physical_num_tokens"],
        "actual physical/semantic context missing",
    )
    check(
        keep
        and all(
            type(x) is int and 0 <= x < binding["physical_num_tokens"] for x in keep
        )
        and keep == sorted(set(keep))
        and fact["source_slots"]
        == slots(fact["source_blocks"], keep, fact["block_size"])
        and fact["destination_slots"]
        == slots(fact["destination_blocks"], list(range(len(keep))), fact["block_size"])
        and not set(fact["source_blocks"]) & set(fact["destination_blocks"]),
        "actual keep/source/private-destination slot mapping differs",
    )
    clocks = [
        fact[k]
        for k in [
            "source_before_gather_ns",
            "reference_after_gather_ns",
            "write_completed_barrier_ns",
            "destination_readback_completed_ns",
        ]
    ]
    check(
        all(type(x) is int and x > 0 for x in clocks) and clocks == sorted(clocks),
        "actual byte/barrier chronology differs",
    )
    for kind in ["K", "V"]:
        view = fact["cache_views"][kind]
        before, after = view["before"], view["after"]
        check(
            before == after
            and before["dtype"] in {"torch.float16", "torch.bfloat16"}
            and len(before["shape"]) == len(before["stride"]) == 4
            and before["shape"][1] == fact["block_size"]
            and before["stride"][1:]
            == [before["shape"][2] * before["shape"][3], before["shape"][3], 1]
            and before["stride"][0] >= fact["block_size"] * before["stride"][1]
            and all(
                type(before[k]) is int and before[k] > 0
                for k in ["data_ptr", "storage_ptr"]
            )
            and type(before["storage_offset"]) is int
            and before["storage_offset"] >= 0,
            "actual cache dtype/layout/pointer binding differs",
        )
        size = len(keep) * before["shape"][2] * before["shape"][3] * 2
        check(0 < size <= 4_194_304, "bounded selected byte payload required")
        check(
            all(type(x) is int and x > 0 for x in before["shape"] + before["stride"])
            and max(fact["source_blocks"] + fact["destination_blocks"])
            < before["shape"][0]
            and (
                (before["device"] == "cpu" and before["npu_format"] is None)
                or (
                    before["device"].startswith("npu:")
                    and type(before["npu_format"]) is int
                    and before["npu_format"] >= 0
                )
            ),
            "actual cache extent/device/format binding differs",
        )
        token_bytes = before["shape"][2] * before["shape"][3] * 2
        block_bytes = {
            int(block): base64.b64decode(value, validate=True)
            for block, value in fact["exported_source_block_bytes"][kind].items()
        }
        needed = {fact["source_blocks"][i // fact["block_size"]] for i in keep}
        source_after = {
            int(block): base64.b64decode(value, validate=True)
            for block, value in fact["source_after_write_block_bytes"][kind].items()
        }
        check(
            source_after == block_bytes,
            "actual source cache changed during private-destination write",
        )
        check(
            set(block_bytes) == needed
            and all(
                len(value) == fact["block_size"] * token_bytes
                for value in block_bytes.values()
            ),
            "bounded actual source block exports differ",
        )
        independent = b"".join(
            block_bytes[fact["source_blocks"][i // fact["block_size"]]][
                (i % fact["block_size"]) * token_bytes : (i % fact["block_size"] + 1)
                * token_bytes
            ]
            for i in keep
        )
        source = base64.b64decode(fact["selected_source_bytes"][kind], validate=True)
        destination = base64.b64decode(
            fact["destination_readback_bytes"][kind], validate=True
        )
        check(
            len(source) == len(destination) == size
            and source == destination == independent
            and source
            == base64.b64decode(fact["gathered_workspace_bytes"][kind], validate=True)
            and digest(source) == fact["selected_source_sha256"][kind]
            and digest(destination) == fact["destination_readback_sha256"][kind],
            "actual selected-source/destination byte predicate differs",
        )
    return {
        "selected_KV_bytes_match": True,
        "formal_result": False,
        "GPU_qualification": None,
        "attention_consumption": None,
        "full_model_quality": None,
    }


def _sync(tensor):
    import torch

    if tensor.device.type == "npu":
        cast(Any, torch).npu.current_stream(tensor.device).synchronize()
    elif tensor.device.type != "cpu":
        raise ValueError("candidate supports only explicit CPU/NPU stream")
    return time.monotonic_ns()


def _raw(tensor):
    import torch

    # Reinterpret, never round through a float/JSON value. Preserves bf16/f16 bits.
    return bytes(
        tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()
    )


def _view(tensor):
    native_format = None
    if tensor.device.type == "npu":
        import torch_npu

        native_format = int(torch_npu.get_npu_format(tensor))
    return {
        "npu_format": native_format,
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_offset": tensor.storage_offset(),
    }


def capture_before_actual_gather(
    k_cache, v_cache, binding, source_slots, destination_slots
):
    if not enabled():
        return None
    _sync(k_cache)
    keep = binding["keep_indices"].detach().cpu().tolist()
    source_blocks = list(binding["source_blocks"])
    block_size = k_cache.shape[1]
    expected_slots = slots(source_blocks, keep, block_size)
    check(
        source_slots.detach().cpu().tolist() == expected_slots,
        "actual device source slots differ from source blocks/keep indices",
    )
    check(
        keep[-1] < binding["physical_num_tokens"],
        "selected position exceeds actual physical context",
    )
    selected = {}
    exports = {}
    for kind, cache in [("K", k_cache), ("V", v_cache)]:
        token_bytes = cache.shape[2] * cache.shape[3] * 2
        # Export only needed actual physical blocks, before workspace gathers.
        needed = {source_blocks[index // block_size] for index in keep}
        blocks = {block: _raw(cache[block]) for block in needed}
        exports[kind] = blocks
        selected[kind] = b"".join(
            blocks[source_blocks[index // block_size]][
                (index % block_size) * token_bytes : (index % block_size + 1)
                * token_bytes
            ]
            for index in keep
        )
    return {
        "binding": {k: v for k, v in binding.items() if k != "keep_indices"},
        "keep_indices": keep,
        "source_slots": expected_slots,
        "destination_slots": destination_slots.detach().cpu().tolist(),
        "cache_before": {"K": _view(k_cache), "V": _view(v_cache)},
        "selected_source_raw": selected,
        "source_block_exports": exports,
        "source_before_gather_ns": time.monotonic_ns(),
    }


def observe_actual_workspace(reference, k_workspace, v_workspace):
    if reference is None:
        return
    _sync(k_workspace)
    observed = {"K": _raw(k_workspace), "V": _raw(v_workspace)}
    check(
        observed == reference["selected_source_raw"],
        "actual gather workspace differs from independent source-block reference",
    )
    reference["workspace_raw"] = observed
    reference["reference_after_gather_ns"] = time.monotonic_ns()


def observe_after_actual_write(reference, k_cache, v_cache, destination_slots, gather):
    if reference is None:
        return
    barrier = _sync(k_cache)
    # Real destination reads after real writes; independent source fixed before gather.
    del gather  # Never reuse the plugin gather primitive as the destination oracle.
    actual_slots = destination_slots.detach().cpu().tolist()
    raw = {}
    source_after = {}
    for kind, cache in [("K", k_cache), ("V", v_cache)]:
        source_after[kind] = {
            block: _raw(cache[block])
            for block in reference["source_block_exports"][kind]
        }
        block_size = cache.shape[1]
        token_bytes = cache.shape[2] * cache.shape[3] * 2
        actual_blocks = {slot // block_size for slot in actual_slots}
        blocks = {block: _raw(cache[block]) for block in actual_blocks}
        raw[kind] = b"".join(
            blocks[slot // block_size][
                (slot % block_size) * token_bytes : (slot % block_size + 1)
                * token_bytes
            ]
            for slot in actual_slots
        )
    _sync(k_cache)
    binding = dict(reference["binding"])
    fact = {
        "schema": SCHEMA,
        "classification": "CPU_SOURCE_CANDIDATE_ONLY",
        "formal_registration": False,
        "GPU_qualification": None,
        "attention_consumption": None,
        "full_model_quality": None,
        "binding": binding,
        "keep_indices": reference["keep_indices"],
        "source_blocks": binding.pop("source_blocks"),
        "destination_blocks": binding.pop("destination_blocks"),
        "block_size": binding.pop("block_size"),
        "source_slots": reference["source_slots"],
        "destination_slots": reference["destination_slots"],
        "source_before_gather_ns": reference["source_before_gather_ns"],
        "reference_after_gather_ns": reference["reference_after_gather_ns"],
        "write_completed_barrier_ns": barrier,
        "destination_readback_completed_ns": time.monotonic_ns(),
        "cache_views": {
            "K": {"before": reference["cache_before"]["K"], "after": _view(k_cache)},
            "V": {"before": reference["cache_before"]["V"], "after": _view(v_cache)},
        },
        "source_after_write_block_bytes": {
            kind: {
                str(block): base64.b64encode(value).decode()
                for block, value in blocks.items()
            }
            for kind, blocks in source_after.items()
        },
        "exported_source_block_bytes": {
            kind: {
                str(block): base64.b64encode(value).decode()
                for block, value in blocks.items()
            }
            for kind, blocks in reference["source_block_exports"].items()
        },
        "selected_source_bytes": {
            kind: base64.b64encode(value).decode()
            for kind, value in reference["selected_source_raw"].items()
        },
        "gathered_workspace_bytes": {
            kind: base64.b64encode(value).decode()
            for kind, value in reference["workspace_raw"].items()
        },
        "destination_readback_bytes": {
            kind: base64.b64encode(value).decode() for kind, value in raw.items()
        },
        "selected_source_sha256": {
            kind: digest(value)
            for kind, value in reference["selected_source_raw"].items()
        },
        "destination_readback_sha256": {
            kind: digest(value) for kind, value in raw.items()
        },
    }
    verify(fact)
    # No callback Boolean or local PublicationAck becomes an authenticated ACK.
    _sink(fact)
