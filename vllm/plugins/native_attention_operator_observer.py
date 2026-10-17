# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in source contacts; callback return is not attention/formal qualification."""

from __future__ import annotations

import hashlib
import json
import math
import time
from enum import IntEnum
from typing import Any

SCHEMA = "ecpa-native-cached-attention-operator-source-candidate/v1"
_publisher: Any = None
_scope: Any = None
_current_stream: Any = None
_get_format: Any = None
_read_prefix: Any = None
_materialized: dict[Any, Any] = {}
_metadata: dict[Any, Any] = {}
_observed: set[Any] = set()
_materialization_inventory: Any = None
_registered_phase_layers: set[Any] = set()
_inventory_tokens: dict[int, Any] = {}


class _RegisteredMaterializationInventory:
    phases: dict[str, dict[str, Any]]
    rank: int
    process: dict[str, Any]
    capacity: int
    sha256: str

    __slots__ = ("phases", "rank", "process", "capacity", "sha256")


def registered_materialization_inventory(spec_raw, policy_raw, process):
    """Derive a finite scope from the private hash-bound registration, not env.

    The bootstrap has independently checked its private context, source pins,
    actual worker and complete transport policy before calling this helper.
    Returning a token never certifies materialization: every entry still needs
    its original authenticated source ACK and exact registered phase/layer.
    """
    require(
        isinstance(spec_raw, bytes) and isinstance(policy_raw, bytes),
        "raw registered spec/policy required",
    )
    require(
        0 < len(spec_raw) <= 65536 and 0 < len(policy_raw) <= 65536,
        "registered private reader bound exceeded",
    )
    spec, policy = json.loads(spec_raw), json.loads(policy_raw)
    require(
        spec["schema"] == "ecpa-native-kv-semantic-collector-prereg/v2"
        and spec["classification"] == "CPU_REVIEW_CANDIDATE_ONLY"
        and spec["formal_producer_registration"] is False,
        "original runtime native spec domain required",
    )
    require(
        policy["native_spec_sha256"] == sha(spec_raw),
        "materialization inventory is not bound to native spec",
    )
    require(
        policy["schema"] == "ecpa-selected-KV-private-artifact-runtime-policy/v1"
        and policy["classification"] == "REAL_RUNTIME_UNQUALIFIED"
        and policy["formal_registration"] is False,
        "runtime registration required for extended materialization inventory",
    )
    require(
        process["role"] == "worker"
        and type(process["ordinal"]) is int
        and process["ordinal"] in (0, 1),
        "registered worker rank required",
    )
    require(
        sum(canonical(s["process"]) == canonical(process) for s in spec["senders"])
        == 1,
        "inventory worker absent or repeated in sender registration",
    )
    phases = {}
    clients, externals, nonces = set(), set(), set()
    for phase in spec["phases"]:
        require(
            set(phase)
            == {
                "phase_id",
                "client_request_id",
                "challenge_nonce",
                "external_request_id",
                "expected_ranks",
            },
            "registered phase fields differ",
        )
        require(
            all(
                type(phase[k]) is str and phase[k]
                for k in (
                    "phase_id",
                    "client_request_id",
                    "challenge_nonce",
                    "external_request_id",
                )
            )
            and canonical(phase["expected_ranks"]) == canonical([0, 1]),
            "registered phase identity/ranks differ",
        )
        pid = phase["phase_id"]
        require(
            pid not in phases
            and phase["client_request_id"] not in clients
            and phase["external_request_id"] not in externals
            and phase["challenge_nonce"] not in nonces,
            "registered phases/requests/nonces alias",
        )
        req = policy["requirements"][pid]
        layers = req["layer_indices"]
        require(
            type(layers) is list
            and layers
            and all(type(x) is int and x >= 0 for x in layers)
            and len(set(layers)) == len(layers),
            "registered layer inventory is absent, repeated or untyped",
        )
        projection = spec["kv_requirements"][pid]
        require(
            all(
                type(req[k]) is int and req[k] > 0 and req[k] == projection[k]
                for k in ("semantic_anchor", "physical_anchor")
            )
            and req["physical_anchor"] < req["semantic_anchor"],
            "registered tensor/projection anchors differ",
        )
        phases[pid] = {"phase": phase, "layers": layers}
        clients.add(phase["client_request_id"])
        externals.add(phase["external_request_id"])
        nonces.add(phase["challenge_nonce"])
    require(
        phases
        and set(policy["requirements"]) == set(phases)
        and set(spec["kv_requirements"]) == set(phases),
        "registered phase/layer coverage differs",
    )
    token = _RegisteredMaterializationInventory()
    # Detached canonical bytes retain exact scalar types; callers cannot mutate
    # the parsed source registration through the returned object.
    token.phases = json.loads(canonical(phases))
    token.process = json.loads(canonical(process))
    token.rank = process["ordinal"]
    token.capacity = max(64, sum(len(p["layers"]) for p in phases.values()))
    token.sha256 = sha(
        canonical(
            {
                "phases": token.phases,
                "process": token.process,
                "rank": token.rank,
                "capacity": token.capacity,
            }
        )
    )
    _inventory_tokens[id(token)] = (
        token,
        canonical(
            {
                "phases": token.phases,
                "process": token.process,
                "rank": token.rank,
                "capacity": token.capacity,
            }
        ),
    )
    return token


def _inventory_current(token):
    record = _inventory_tokens.get(id(token))
    require(
        type(token) is _RegisteredMaterializationInventory
        and record is not None
        and record[0] is token,
        "source-derived inventory token required; caller capacity rejected",
    )
    assert record is not None
    # The trust anchor lives in the private issuer registry, independently of
    # every caller-visible mutable field (including the informative checksum).
    require(
        canonical(
            {
                "phases": token.phases,
                "process": token.process,
                "rank": token.rank,
                "capacity": token.capacity,
            }
        )
        == record[1]
        and token.sha256 == sha(record[1]),
        "registered materialization inventory changed",
    )
    return token


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def enabled():
    return _publisher is not None


def register_source_candidate(
    publisher,
    *,
    scope,
    current_stream,
    get_format,
    read_prefix=None,
    materialization_inventory=None,
):
    """Explicit new source-only opt-in. No existing source producer is upgraded."""
    global _publisher, _scope, _current_stream, _get_format, _read_prefix
    global _materialization_inventory
    require(not enabled(), "attention source callback already registered")
    require(
        all(callable(f) for f in (publisher, scope, current_stream, get_format)),
        "attention source callbacks missing",
    )
    if materialization_inventory is not None:
        token = _inventory_current(materialization_inventory)
        require(
            canonical(scope()["process"]) == canonical(token.process),
            "registered inventory belongs to another actual worker",
        )
        _materialization_inventory = token
    _publisher, _scope = publisher, scope
    _current_stream, _get_format = current_stream, get_format
    _read_prefix = read_prefix or read_destination_prefix


def remember_acknowledged_materialization(fact, ack):
    """Called ONLY after actual artifact publish validates its Unix source ACK."""
    if not enabled():
        return
    scope = _scope()
    require(
        fact["native_producer"] == scope["process"], "materialization worker changed"
    )
    require(
        fact["classification"] == "REAL_RUNTIME_UNQUALIFIED",
        "source candidate requires actual native runtime materialization",
    )
    require(
        ack["schema"] == "ecpa-native-semantic-collector-ack/v1"
        and ack["formal_producer_registration"] is False
        and ack["ack_sha256"]
        == sha(canonical({k: v for k, v in ack.items() if k != "ack_sha256"})),
        "actual source ACK is absent or corrupt",
    )
    require(ack["phase"]["expected_ranks"] == [0, 1], "source ACK rank scope changed")
    binding = fact["binding"]
    key = (binding["runtime_request_id"], binding["layer_name"])
    limit = 64
    phase_layer = None
    if _materialization_inventory is not None:
        token = _inventory_current(_materialization_inventory)
        require(
            canonical(scope["process"]) == canonical(token.process),
            "registered materialization worker changed",
        )
        phase = ack["phase"]
        registered = token.phases.get(phase["phase_id"])
        require(
            registered is not None
            and canonical(phase) == canonical(registered["phase"])
            and type(binding["rank"]) is int
            and binding["rank"] == token.rank
            and canonical(binding["expected_layer_indices"])
            == canonical(registered["layers"])
            and type(binding["layer_index"]) is int
            and binding["layer_index"] in registered["layers"],
            "ACK materialization is outside exact registered phase/rank/layers",
        )
        phase_layer = (phase["phase_id"], binding["layer_index"])
        require(
            phase_layer not in _registered_phase_layers,
            "registered phase/layer materialization repeated",
        )
        limit = token.capacity
    require(len(_materialized) < limit, "bounded fresh materialization scope exceeded")
    require(key not in _materialized, "ambiguous materialization for request/layer")
    # Actual ACK raw digest binds the transport envelope, whereas this digest
    # binds its privately archived typed payload; they are intentionally distinct.
    _materialized[key] = {
        "fact": fact,
        "fact_sha256": sha(canonical(fact)),
        "ack": ack,
        "registered_monotonic_ns": time.monotonic_ns(),
    }
    if phase_layer is not None:
        _registered_phase_layers.add(phase_layer)


def bind_actual_input_batch(runner, metadata, *, eager):
    """Source-owned IDs and row mapping after real attention metadata construction."""
    if not enabled():
        return
    _metadata.clear()
    # These are explicit unsupported paths, not inferred successful consumption.
    if not eager or isinstance(metadata, list):
        return
    require(isinstance(metadata, dict), "per-layer metadata mapping unsupported")
    req_ids = runner.input_batch.req_ids[: runner.input_batch.num_reqs]
    if len(req_ids) != 1:
        return
    request_id = req_ids[0]
    from .native_sampled_selection import runtime_request_selected

    if not runtime_request_selected(request_id):
        return  # before metadata/readback/OP construction, no fake native receipt
    require(
        runner.input_batch.req_id_to_index[request_id] == 0,
        "actual InputBatch request row differs",
    )
    require(runner.requests[request_id] is not None, "actual runner request absent")
    for layer_name, value in metadata.items():
        if (request_id, layer_name) not in _materialized:
            continue
        state = getattr(getattr(value, "attn_state", None), "name", None)
        if state != "DecodeOnly":
            continue
        require(
            value.num_prefills == 0 and value.num_decodes == 1,
            "cached decode metadata includes another request",
        )
        require(value.actual_seq_lengths_q == [1], "decode is not one actual token")
        tables = runner.input_batch.block_table.block_tables
        require(len(tables) == 1, "only actual single attention group supported")
        table = tables[0]
        width = int(table.num_blocks_per_row[0])
        require(0 < width <= table.block_table.gpu.shape[1], "actual row width invalid")
        cpu_blocks = table.block_table.np[0, :width].tolist()
        require(
            cpu_blocks == list(runner.requests[request_id].block_ids[0]),
            "actual CPU row and request metadata differ",
        )
        lengths = value.seq_lens.detach().cpu().tolist()
        require(
            lengths == list(value.seq_lens_list) and len(lengths) == 1,
            "actual native decode sequence metadata fields diverged",
        )
        _metadata[(id(value), layer_name)] = {
            "object": value,
            "runtime_request_id": request_id,
            "layer_name": layer_name,
            "metadata_row": 0,
            "actual_cpu_block_ids": cpu_blocks,
            "metadata_seq_lengths_kv": lengths,
            "metadata_actual_seq_lengths_q": list(value.actual_seq_lengths_q),
            "bound_monotonic_ns": time.monotonic_ns(),
        }


# Explicit exact vendor codes, excluding UNDEFINED. New/unknown formats are not
# silently mapped to a known storage layout by integer coercion.
_VENDOR_FORMAT_SOURCE_SHA256 = (
    "8abe7bea6817f25d2360aee783afae16fe7fe2aaa46a04e7b7b865e7f2039488"
)
_NPU_FORMAT_CODES = frozenset(
    {0, 1, 2, 3, 4, 12, 16, 27, 29, 30, 32, 33, 35, 47, 50, 51, 52, 53, 54}
)


class SourceObservationRejected(ValueError):
    """Bounded real rejected descriptors, never an OP receipt or source ACK."""

    def __init__(self, reason, fields, *, stage):
        self.diagnostic = {
            "schema": "ecpa-native-attention-source-rejection/v1",
            "classification": "SOURCE_REJECTION_ONLY",
            "stage": stage,
            "reason": str(reason)[:256],
            "actual_fields": fields,
            "OP_receipt": False,
            "formal_registration": False,
        }
        encoded = canonical(self.diagnostic)
        require(len(encoded) <= 8192, "source rejection diagnostic exceeds bound")
        super().__init__(str(reason)[:256] + "; diagnostic=" + encoded.decode())


def _field_descriptor(value):
    return {
        "python_type": (type(value).__module__ + "." + type(value).__qualname__)[:80],
        "value_repr": repr(value)[:80],
    }


def canonical_native_format(value):
    """Only a plain int or this actual vendor Format is a wire integer."""
    if type(value) is int:
        code = value
    else:
        if not isinstance(value, IntEnum):
            raise SourceObservationRejected(
                "native NPU format type unsupported",
                {"npu_format": _field_descriptor(value)},
                stage="cache-view-format",
            )
        try:
            from pathlib import Path

            import torch_npu.npu._format as vendor_format

            Format = vendor_format.Format
            actual_vendor_source = Path(vendor_format.__file__).read_bytes()
            actual_source_sha = sha(actual_vendor_source)
            if actual_source_sha != _VENDOR_FORMAT_SOURCE_SHA256:
                raise SourceObservationRejected(
                    "native vendor format source bytes changed",
                    {
                        "actual_source_sha256": _field_descriptor(actual_source_sha),
                        "expected_source_sha256": _field_descriptor(
                            _VENDOR_FORMAT_SOURCE_SHA256
                        ),
                        "npu_format": _field_descriptor(value),
                    },
                    stage="cache-view-format",
                )
        except (ImportError, OSError) as exc:
            raise SourceObservationRejected(
                "native vendor format source unavailable",
                {"npu_format": _field_descriptor(value)},
                stage="cache-view-format",
            ) from exc
        if Format is None or type(value) is not Format:
            raise SourceObservationRejected(
                "native NPU format type unsupported",
                {"npu_format": _field_descriptor(value)},
                stage="cache-view-format",
            )
        code = int(value)
    if code not in _NPU_FORMAT_CODES:
        raise SourceObservationRejected(
            "native NPU format code unsupported",
            {"npu_format": _field_descriptor(value)},
            stage="cache-view-format",
        )
    return code


def _cache_rejection_fields(observed, materialization):
    fields = {}
    for label, rows in (
        ("actual", observed["native_arguments"]["cache_views"]),
        (
            "reference",
            {k: v["after"] for k, v in materialization["cache_views"].items()},
        ),
    ):
        for kind, view in rows.items():
            fields[label + "." + kind] = {
                name: _field_descriptor(value)
                for name, value in view.items()
                if name
                in {
                    "device",
                    "dtype",
                    "shape",
                    "stride",
                    "data_ptr",
                    "storage_ptr",
                    "storage_offset",
                    "npu_format",
                }
            }
    fields["runtime_request_id"] = _field_descriptor(observed["runtime_request_id"])
    fields["layer_name"] = _field_descriptor(observed["layer_name"])
    for name in ("block_table_rows", "sequence_lengths"):
        fields["native_arguments." + name] = _field_descriptor(
            observed["native_arguments"][name]
        )
    for name in (
        "actual_cpu_block_ids",
        "metadata_seq_lengths_kv",
        "metadata_bound_monotonic_ns",
        "native_entry_readback_started_ns",
        "native_entry_readback_completed_ns",
        "native_call_started_ns",
        "native_call_returned_ns",
        "native_stream_completed_ns",
        "monotonic_ns",
    ):
        fields[name] = _field_descriptor(observed[name])
    return fields


def cache_view(tensor):
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "npu_format": canonical_native_format(_get_format(tensor))
        if str(tensor.device).startswith("npu:")
        else None,
    }


def read_destination_prefix(cache, fact, kind):
    """Actual logical block exports at OP entry; independent of KV gather helper."""
    import torch

    reference_shape = fact["cache_views"][kind]["after"]["shape"]
    reference = fact["cache_views"][kind]["after"]
    actual_shape, actual_stride = list(cache.shape), list(cache.stride())
    require(
        (actual_shape == reference_shape and actual_stride == reference["stride"])
        or (
            actual_shape
            == [
                reference_shape[0],
                reference_shape[1],
                reference_shape[2] * reference_shape[3],
            ]
            and actual_stride == [reference["stride"][0], reference["stride"][1], 1]
        ),
        "native cache layout is not actual typed 4D or exact flatten",
    )
    require(
        str(cache.dtype) in {"torch.float16", "torch.bfloat16"}
        and str(cache.dtype) == reference["dtype"],
        "native prefix reader supports actual binary16 cache dtype only",
    )
    logical = cache.view(reference_shape)
    remaining = len(fact["keep_indices"])
    block_size = fact["block_size"]
    digest = hashlib.sha256()
    for block in fact["destination_blocks"]:
        if remaining <= 0:
            break
        count = min(remaining, block_size)
        # Copy the actual logical block through framework device-to-CPU. No
        # compression workspace or previous digest supplies these bytes.
        raw = logical[block].detach().to(device="cpu", non_blocking=False).contiguous()
        raw = raw.view(torch.uint8).numpy().tobytes()
        token_bytes = reference_shape[2] * reference_shape[3] * 2
        require(
            len(raw) == block_size * token_bytes, "native cache block extent differs"
        )
        digest.update(raw[: count * token_bytes])
        remaining -= count
    require(remaining == 0, "native destination prefix allocation incomplete")
    return digest.hexdigest()


def argument_descriptor(value):
    if hasattr(value, "data_ptr") and hasattr(value, "untyped_storage"):
        return {"kind": "tensor", "view": cache_view(value)}
    if value is None or type(value) in (str, bool, int):
        return {"kind": "scalar", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple)) and len(value) <= 4096:
        return {"kind": "sequence", "value": [argument_descriptor(x) for x in value]}
    raise ValueError("actual native argument type unsupported")


def verify_observation(value, materialization):
    """Pure bounded source predicate; cannot authenticate sender or prove quality."""
    require(
        value["schema"] == SCHEMA
        and value["classification"] == "REAL_RUNTIME_UNQUALIFIED"
        and value["formal_registration"] is False
        and all(
            value[k] is None
            for k in (
                "GPU_qualification",
                "attention_consumption",
                "full_model_quality",
            )
        ),
        "attention source candidate scope promoted",
    )
    require(
        value["operator"]
        in {
            "torch_npu._npu_paged_attention",
            "torch_npu.npu_fused_infer_attention_score",
        }
        and value["attn_state"] == "DecodeOnly"
        and value["execution_mode"] == "eager-single-request-cached-decode",
        "unsupported actual attention branch",
    )
    descriptors = value["native_argument_descriptors"]
    require(
        value["native_argument_descriptors_sha256"] == sha(canonical(descriptors)),
        "native argument descriptor bytes changed",
    )
    query = descriptors["query"]
    require(
        query["kind"] == "tensor"
        and query["view"]["device"]
        == materialization["cache_views"]["K"]["after"]["device"]
        and query["view"]["dtype"]
        == materialization["cache_views"]["K"]["after"]["dtype"],
        "native query is not actual matching-device tensor",
    )
    binding = materialization["binding"]
    require(
        value["runtime_request_id"] == binding["runtime_request_id"]
        and value["layer_name"] == binding["layer_name"]
        and value["layer_index"] == binding["layer_index"]
        and value["process"] == materialization["native_producer"],
        "attention request/actual layer/worker does not match materialization",
    )
    transaction = sha(
        canonical(
            {
                "runtime_request_id": binding["runtime_request_id"],
                "compression_plan": binding["compression_plan"],
            }
        )
    )
    require(
        value["materialization_transaction_id"] == transaction,
        "actual source compression plan transaction differs",
    )
    require(
        value["source_materialization_sha256"] == sha(canonical(materialization)),
        "attention reference is not actual archived materialization",
    )
    clocks = [
        materialization["destination_readback_completed_ns"],
        value["materialization_ack_registered_monotonic_ns"],
        value["metadata_bound_monotonic_ns"],
        value["native_entry_readback_started_ns"],
        value["native_entry_readback_completed_ns"],
        value["native_call_started_ns"],
        value["native_call_returned_ns"],
        value["native_stream_completed_ns"],
        value["monotonic_ns"],
    ]
    require(
        all(type(x) is int and x > 0 for x in clocks) and clocks == sorted(clocks),
        "actual native completion chronology differs",
    )
    require(
        type(value["stream"]["npu_stream"]) is int
        and value["stream"]["npu_stream"] >= 0
        and value["stream_before"] == value["stream_after"] == value["stream"],
        "native operation changed completion stream",
    )
    rows = value["native_arguments"]["block_table_rows"]
    seq = value["native_arguments"]["sequence_lengths"]
    require(
        isinstance(rows, list)
        and len(rows) == 1
        and all(type(x) is int and x >= 0 for x in rows[0])
        and len(rows[0]) <= 4096
        and isinstance(seq, list)
        and len(seq) == 1
        and type(seq[0]) is int
        and seq[0] >= len(materialization["keep_indices"]),
        "native block table or sequence extent is unsupported",
    )
    require(
        value["actual_cpu_block_ids"]
        and rows[0][: len(value["actual_cpu_block_ids"])]
        == value["actual_cpu_block_ids"],
        "native OP block table differs from source-owned actual InputBatch row",
    )
    require(
        seq == value["metadata_seq_lengths_kv"]
        and value["metadata_actual_seq_lengths_q"] == [1],
        "actual native sequence lengths differ from bound source metadata",
    )
    destinations = materialization["destination_blocks"]
    require(
        rows[0][: len(destinations)] == destinations,
        "actual native block table does not address materialized destinations",
    )
    require(
        len(value["actual_cpu_block_ids"]) * materialization["block_size"] >= seq[0],
        "native sequence extends beyond actual block table",
    )
    require(
        value["stream"]["device"]
        == materialization["cache_views"]["K"]["after"]["device"],
        "native completion stream is on another device",
    )
    ack = value["materialization_ack"]
    require(
        ack["schema"] == "ecpa-native-semantic-collector-ack/v1"
        and ack["formal_producer_registration"] is False
        and ack["ack_sha256"]
        == sha(canonical({k: v for k, v in ack.items() if k != "ack_sha256"})),
        "materialization acknowledgement provenance changed",
    )
    require(
        value["native_selected_destination_sha256"]
        == materialization["destination_readback_sha256"],
        "actual selected destination bytes changed before native OP entry",
    )
    for kind in ("K", "V"):
        native_name = (
            ("key_cache" if kind == "K" else "value_cache")
            if (value["operator"] == "torch_npu._npu_paged_attention")
            else ("key" if kind == "K" else "value")
        )
        require(
            descriptors[native_name]["kind"] == "tensor"
            and descriptors[native_name]["view"]
            == value["native_arguments"]["cache_views"][kind],
            "native actual argument and interpreted cache view differ",
        )
        actual = value["native_arguments"]["cache_views"][kind]
        ref = materialization["cache_views"][kind]["after"]
        require(
            set(actual)
            == {
                "device",
                "dtype",
                "shape",
                "stride",
                "data_ptr",
                "storage_ptr",
                "storage_offset",
                "npu_format",
            }
            and all(
                type(actual[k]) is int and actual[k] > 0
                for k in ("data_ptr", "storage_ptr")
            )
            and type(actual["storage_offset"]) is int
            and actual["storage_offset"] >= 0
            and type(actual["npu_format"]) is int
            and actual["npu_format"] in _NPU_FORMAT_CODES,
            "native cache view fields/types unsupported",
        )
        require(
            actual["device"].startswith("npu:")
            and all(
                actual[k] == ref[k]
                for k in (
                    "device",
                    "dtype",
                    "storage_ptr",
                    "storage_offset",
                    "data_ptr",
                    "npu_format",
                )
            ),
            "native OP uses another cache allocation/view/format",
        )
        require(
            (actual["shape"] == ref["shape"] and actual["stride"] == ref["stride"])
            or (
                value["operator"] == "torch_npu.npu_fused_infer_attention_score"
                and actual["shape"]
                == [ref["shape"][0], ref["shape"][1], ref["shape"][2] * ref["shape"][3]]
                and actual["stride"] == [ref["stride"][0], ref["stride"][1], 1]
            ),
            "native OP view is not actual cache or supported exact flatten",
        )
    require(len(canonical(value)) <= 49152, "bounded native operator fact required")
    return {
        "source_argument_completion_relation": True,
        "formal_registration": False,
        "attention_consumption": None,
        "GPU_qualification": None,
        "full_model_quality": None,
    }


def observe_native_call(operator, native_call, arguments, *, metadata, layer_name):
    """Wrap actual final native call; default/unsupported paths produce no witness."""
    if not enabled():
        return native_call(**arguments)
    metadata_key = (id(metadata), layer_name)
    if metadata_key not in _metadata:
        # Native groups share one metadata object across many layers. Each
        # layer must retain its own source-owned runtime/materialization edge.
        # A known object with an unbound layer (or vice versa) is not an
        # unsupported branch that can silently bypass the source checks.
        require(
            not any(
                bound["object"] is metadata or bound["layer_name"] == layer_name
                for bound in _metadata.values()
            ),
            "native metadata/layer source binding changed",
        )
        return native_call(**arguments)
    bound = _metadata[metadata_key]
    require(
        bound["object"] is metadata and bound["layer_name"] == layer_name,
        "native metadata/layer source binding changed",
    )
    from .native_observer_rebind import request_unavailable

    if request_unavailable(bound["runtime_request_id"]):
        return native_call(**arguments)  # Execute real OP; emit no witness.
    key_id = (bound["runtime_request_id"], layer_name)
    if key_id in _observed:
        return native_call(**arguments)
    registered = _materialized[key_id]
    fact = registered["fact"]
    require(
        registered["fact_sha256"] == sha(canonical(fact)),
        "registered materialization changed",
    )
    if operator == "torch_npu._npu_paged_attention":
        key, val = arguments["key_cache"], arguments["value_cache"]
        seq = arguments["context_lens"].detach().cpu().tolist()
    else:
        key, val = arguments["key"], arguments["value"]
        seq = list(arguments["actual_seq_lengths_kv"])
    table = arguments["block_table"].detach().cpu().tolist()
    entry_readback_started = time.monotonic_ns()
    entry_digests = {
        "K": _read_prefix(key, fact, "K"),
        "V": _read_prefix(val, fact, "V"),
    }
    entry_readback_completed = time.monotonic_ns()
    descriptors = {
        name: argument_descriptor(value) for name, value in arguments.items()
    }
    stream = _current_stream(key.device)
    stream_id = {"npu_stream": stream.npu_stream, "device": str(stream.device)}
    observed = {
        "schema": SCHEMA,
        "classification": "REAL_RUNTIME_UNQUALIFIED",
        "formal_registration": False,
        **_scope(),
        "operator": operator,
        "execution_mode": "eager-single-request-cached-decode",
        "attn_state": "DecodeOnly",
        "runtime_request_id": bound["runtime_request_id"],
        "layer_name": layer_name,
        "layer_index": fact["binding"]["layer_index"],
        "metadata_row": 0,
        "actual_cpu_block_ids": bound["actual_cpu_block_ids"],
        "metadata_seq_lengths_kv": bound["metadata_seq_lengths_kv"],
        "metadata_actual_seq_lengths_q": bound["metadata_actual_seq_lengths_q"],
        "source_materialization_sha256": registered["fact_sha256"],
        "materialization_transaction_id": sha(
            canonical(
                {
                    "runtime_request_id": bound["runtime_request_id"],
                    "compression_plan": fact["binding"]["compression_plan"],
                }
            )
        ),
        "materialization_ack": registered["ack"],
        "materialization_ack_registered_monotonic_ns": registered[
            "registered_monotonic_ns"
        ],
        "metadata_bound_monotonic_ns": bound["bound_monotonic_ns"],
        "native_selected_destination_sha256": entry_digests,
        "native_entry_readback_started_ns": entry_readback_started,
        "native_entry_readback_completed_ns": entry_readback_completed,
        "native_argument_descriptors": descriptors,
        "native_argument_descriptors_sha256": sha(canonical(descriptors)),
        "native_arguments": {
            "cache_views": {"K": cache_view(key), "V": cache_view(val)},
            "block_table_rows": table,
            "sequence_lengths": seq,
        },
        "stream": stream_id,
        "stream_before": stream_id,
        "native_call_started_ns": time.monotonic_ns(),
        "GPU_qualification": None,
        "attention_consumption": None,
        "full_model_quality": None,
    }
    result = native_call(**arguments)
    observed["native_call_returned_ns"] = time.monotonic_ns()
    stream.synchronize()
    observed["native_stream_completed_ns"] = time.monotonic_ns()
    after = _current_stream(key.device)
    observed["stream_after"] = {
        "npu_stream": after.npu_stream,
        "device": str(after.device),
    }
    observed["monotonic_ns"] = time.monotonic_ns()
    try:
        verify_observation(observed, fact)
    except ValueError as exc:
        raise SourceObservationRejected(
            str(exc),
            _cache_rejection_fields(observed, fact),
            stage="after-native-completion-validation",
        ) from exc
    _publisher(observed)  # Failure propagates; no return is an authority ACK claim.
    _observed.add(key_id)
    return result
