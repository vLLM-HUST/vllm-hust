"""Explicit real-runtime-unqualified contract; source bytes are not qualification."""

from __future__ import annotations

import re

if __package__ and __package__.startswith("vllm."):
    from . import native_tensor_witness as witness
else:
    from experiments.native_kv_tensor_candidate import witness

SCHEMA = "ecpa-selected-KV-materialization-runtime-unqualified/v1"
CLASSIFICATION = "REAL_RUNTIME_UNQUALIFIED"


def verify(fact, process):
    witness.check(
        fact.get("schema") == SCHEMA
        and fact.get("classification") == CLASSIFICATION
        and fact.get("native_producer") == process,
        "real runtime tensor schema/source role missing",
    )
    witness.check(
        process.get("role") == "worker"
        and process.get("assignment_source") == "host"
        and type(process.get("pid")) is int
        and process["pid"] > 0
        and type(process.get("ordinal")) is int
        and process["ordinal"] == fact["binding"]["rank"]
        and type(process.get("start_ticks")) is int
        and process["start_ticks"] > 0
        and type(process.get("session")) is int
        and process["session"] > 0
        and isinstance(process.get("boot_id"), str)
        and process["boot_id"],
        "actual native runtime worker identity missing",
    )
    devices = set()
    for view in fact["cache_views"].values():
        for stage in ["before", "after"]:
            actual = view[stage]
            witness.check(
                isinstance(actual["device"], str)
                and re.fullmatch(r"npu:[0-9]+", actual["device"])
                and type(actual["npu_format"]) is int
                and actual["npu_format"] >= 0,
                "real runtime contract rejects CPU or unobserved NPU cache format",
            )
            devices.add(actual["device"])
    witness.check(len(devices) == 1, "actual K/V readbacks span different devices")
    # Explicitly reuse only the original byte/layout/slot predicate. This local
    # adapter is not an old CPU archive, an admission, or a hardware result.
    selected = {k: v for k, v in fact.items() if k != "native_producer"}
    selected["schema"] = witness.SCHEMA
    selected["classification"] = "CPU_SOURCE_CANDIDATE_ONLY"
    result = witness.verify(selected)
    return result | {
        "classification": CLASSIFICATION,
        "actual_device": next(iter(devices)),
    }


def from_actual_native_readback(fact, process):
    """Source callback encodes fresh actual NPU readback; CPU cannot be upgraded."""
    witness.verify(fact)
    encoded = fact | {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "native_producer": dict(process),
    }
    verify(encoded, process)
    return encoded
