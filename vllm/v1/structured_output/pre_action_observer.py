# SPDX-License-Identifier: Apache-2.0
"""Prompt-free structured-output observations available before admission.

The completion endpoint only gives vLLM the schema after the client opens the
request.  This module therefore exposes the facts that can be obtained from a
caller supplied schema before that request: canonical schema bytes, schema
count, tool count and requested output budget.  Compiler state is deliberately
reported as missing until a backend supplies a real pre-action snapshot; it is
never inferred from the JSON document or from a later mask event.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


SCHEMA = "qbi.schema-pre-action-observer-state.v1"
PROVIDER_SCHEMA = "qbi.schema-pre-action-observer-state.v2"
PREDICTOR_PRE_ACTION_SCHEMA = "qbi.schema-predictor-pre-action-state.v1"
PREFIX_PROVIDER_SCHEMA = "qbi.prefix-cache-pre-action-observer-state.v2"


def _field(value: Any, *, source: str, observed: bool = True) -> dict[str, Any]:
    result = {"status": "observed" if observed else "missing", "source": source}
    if observed:
        result["value"] = value
    return result


def observe_schema_pre_action(
    *,
    schema_document: Mapping[str, Any] | None,
    tool_count: int | None,
    requested_max_tokens: int | None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Return only source-bound fields known before a completion request.

    ``backend`` is accepted as a capability label only.  It does not make a
    compiler state available: compilation happens in ``grammar_init`` after
    the EngineCore receives the request.
    """

    if not isinstance(schema_document, Mapping) or not schema_document:
        raise ValueError("schema_document is required for pre-action observation")
    if type(tool_count) is not int or tool_count < 0:
        raise ValueError("tool_count must be a non-negative integer")
    if type(requested_max_tokens) is not int or requested_max_tokens < 0:
        raise ValueError("requested_max_tokens must be a non-negative integer")
    canonical = json.dumps(
        schema_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    fields = {
        # This is explicitly the caller schema document size.  It is not
        # presented as the size of a compiled backend grammar.
        "schema_size_bytes": _field(len(canonical), source="request.schema_document.canonical_json"),
        "grammar_size_bytes": _field(len(canonical), source="request.schema_document.canonical_json"),
        "schema_sha256": _field(hashlib.sha256(canonical).hexdigest(), source="request.schema_document.canonical_json"),
        "schema_count": _field(1, source="request.schema_document"),
        "tool_count": _field(tool_count, source="request.tools"),
        "requested_max_tokens": _field(requested_max_tokens, source="request.max_tokens"),
        "grammar_state_count": _field("backend_snapshot_required", source="structured_output_backend", observed=False),
        "grammar_branching_factor": _field("backend_snapshot_required", source="structured_output_backend", observed=False),
        "compile_ready": _field("post_request_only", source="structured_output_manager.grammar_init", observed=False),
    }
    missing = [name for name, item in fields.items() if item["status"] == "missing"]
    return {
        "schema": SCHEMA,
        "observation_point": "before_completion_request",
        "backend": backend,
        "fields": fields,
        "missing_fields": missing,
        "status": "NOT_READY" if missing else "READY",
        "send_allowed": not missing,
        "model_calls": 0,
        "remote": False,
    }


def observe_schema_predictor_pre_action_state(
    *,
    schema_document: Mapping[str, Any],
    tool_count: int,
    requested_max_tokens: int,
    request_sha256: str,
    request_id: str | None,
    engine_boot_id: str,
    config_epoch: int,
    clock_id: str,
    observed_at_ns: int,
    decision_at_ns: int,
    backend_initialized_before_prepare: bool,
    compile_cache_state_before_prepare: Mapping[str, Any] | None,
    runtime_snapshot: Mapping[str, Any] | None = None,
    require_compile_cache_state: bool = False,
) -> dict[str, Any]:
    """Capture only read-only Schema state available before native prepare.

    This path never calls ``compile_grammar`` and deliberately does not emit
    ``compile_ready`` or ``compile_cache_hit``. Those describe a completed
    compile/query and belong to the action-outcome receipt. Cache readiness,
    occupancy, and capacity are separate provider state observations.
    """
    if not isinstance(schema_document, Mapping) or not schema_document:
        raise ValueError("schema_document is required for Schema pre-action state")
    if type(tool_count) is not int or tool_count < 0:
        raise ValueError("tool_count must be a non-negative integer")
    if type(requested_max_tokens) is not int or requested_max_tokens < 0:
        raise ValueError("requested_max_tokens must be a non-negative integer")
    if (
        not isinstance(request_sha256, str)
        or len(request_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in request_sha256)
    ):
        raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
    if type(backend_initialized_before_prepare) is not bool:
        raise ValueError("backend initialization state must be a pre-action bool")

    canonical = json.dumps(
        schema_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    request_id = request_id or "schema-observer:" + request_sha256[:24]
    binding, missing = _binding(
        request_id=request_id,
        request_sha256=request_sha256,
        engine_boot_id=engine_boot_id,
        config_epoch=config_epoch,
        clock_id=clock_id,
        observed_at_ns=observed_at_ns,
        decision_at_ns=decision_at_ns,
    )

    def observed(value: Any, source: str) -> dict[str, Any]:
        return {
            "status": "observed",
            "value": value,
            "source_field": source,
            "observed_at_ns": observed_at_ns,
            "clock_id": clock_id,
        }

    def absent(source: str) -> dict[str, Any]:
        return {
            "status": "missing",
            "source_field": source,
            "observed_at_ns": observed_at_ns,
            "clock_id": clock_id,
        }

    cache = compile_cache_state_before_prepare
    cache = cache if isinstance(cache, Mapping) else {}
    cache_ready = cache.get("cache_ready")
    cache_size = cache.get("cache_size_bytes")
    cache_limit = cache.get("cache_limit_bytes")
    if cache_ready is not None and type(cache_ready) is not bool:
        raise ValueError("compile cache readiness must be a provider bool")
    for name, value in (("cache_size_bytes", cache_size), ("cache_limit_bytes", cache_limit)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"compile cache {name} must be a non-negative integer")
    cache_state_complete = all(
        value is not None for value in (cache_ready, cache_size, cache_limit)
    )
    # An uninitialized backend has a real, observed False initialization state;
    # its cache fields are correctly typed missing. Once initialized, however,
    # a required cache snapshot must contain readiness, occupancy, and limit.
    if (
        require_compile_cache_state
        and backend_initialized_before_prepare
        and not cache_state_complete
    ):
        missing.append("pre_action_fields.compile_cache_state")

    pre_action_fields = {
        "backend_initialized_before_prepare": observed(
            backend_initialized_before_prepare,
            "engine_core.structured_output_manager.backend",
        ),
        "compile_cache_ready_before_prepare": (
            observed(
                cache_ready,
                str(cache.get("cache_ready_source_field") or "structured_output_backend.compiler.cache_enabled"),
            )
            if cache_ready is not None
            else absent(str(cache.get("cache_ready_source_field") or "structured_output_backend.compiler.cache_enabled"))
        ),
        "compile_cache_size_before_prepare_bytes": (
            observed(
                cache_size,
                str(cache.get("cache_size_source_field") or cache.get("source_field") or "structured_output_backend.compiler.get_cache_size_bytes"),
            )
            if cache_size is not None
            else absent(str(cache.get("cache_size_source_field") or cache.get("source_field") or "structured_output_backend.compiler.get_cache_size_bytes"))
        ),
        "compile_cache_limit_before_prepare_bytes": (
            observed(
                cache_limit,
                str(cache.get("cache_limit_source_field") or "structured_output_backend.compiler.cache_limit_bytes"),
            )
            if cache_limit is not None
            else absent(str(cache.get("cache_limit_source_field") or "structured_output_backend.compiler.cache_limit_bytes"))
        ),
    }
    request_fields = {
        "schema_size_bytes": observed(len(canonical), "request.schema_document.canonical_json"),
        "schema_sha256": observed(hashlib.sha256(canonical).hexdigest(), "request.schema_document.canonical_json"),
        "schema_count": observed(1, "request.schema_document"),
        "tool_count": observed(tool_count, "request.tools"),
        "requested_max_tokens": observed(requested_max_tokens, "request.max_tokens"),
    }
    return {
        "schema": PREDICTOR_PRE_ACTION_SCHEMA,
        "observation_point": "before_native_prepare",
        "predictor_input_allowed": True,
        "engine_boot_id": engine_boot_id,
        "config_epoch": config_epoch,
        "request_fields": request_fields,
        "pre_action_fields": pre_action_fields,
        "binding": binding,
        "runtime_snapshot": dict(runtime_snapshot or {}),
        "missing_fields": sorted(set(missing)),
        "status": "READY" if not missing else "NOT_READY",
        "send_allowed": not missing,
        "model_calls": 0,
        "remote": False,
    }
def _digest(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _binding(
    *,
    request_id: str | None,
    request_sha256: str,
    engine_boot_id: str | None,
    config_epoch: int | None,
    clock_id: str | None,
    observed_at_ns: int | None,
    decision_at_ns: int | None,
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    normalized_request_id = str(request_id or "").strip()
    if not normalized_request_id:
        missing.append("binding.request_id")
    if not isinstance(engine_boot_id, str) or not engine_boot_id.strip():
        missing.append("binding.engine_boot_id")
    if type(config_epoch) is not int or config_epoch < 0:
        missing.append("binding.config_epoch")
    if not isinstance(clock_id, str) or not clock_id.strip():
        missing.append("binding.clock_id")
    if type(observed_at_ns) is not int or observed_at_ns < 0:
        missing.append("binding.observed_at_ns")
    if type(decision_at_ns) is not int or decision_at_ns < 0:
        missing.append("binding.decision_at_ns")
    if (
        type(observed_at_ns) is int
        and type(decision_at_ns) is int
        and decision_at_ns < observed_at_ns
    ):
        missing.append("binding.decision_at_ns_before_observation")
    binding = {
        "request_id": normalized_request_id or None,
        # Keep the provider terminology alongside the legacy request_id key.
        # The latter remains part of the v2 compatibility surface, while the
        # former makes it explicit that this identity is assigned/bound by the
        # provider before admission.
        "provider_request_id": normalized_request_id or None,
        "request_sha256": request_sha256,
        "engine_boot_id": engine_boot_id,
        "config_epoch": config_epoch,
        "clock_id": clock_id,
        "observed_at_ns": observed_at_ns,
        "decision_at_ns": decision_at_ns,
        "boundary": "pre_action",
    }
    return binding, missing


def _observed(value: Any, *, source: str) -> dict[str, Any]:
    return {"status": "observed", "value": value, "source": source}


def _missing(source: str) -> dict[str, Any]:
    return {"status": "missing", "source": source}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _normalize_compile_cache_evidence(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize manager/compiler cache evidence without inventing hit state."""

    source = "structured_output_backend.compiler.cache_state"
    if not isinstance(raw, Mapping) or raw.get("status") != "observed":
        return {
            "status": "missing",
            "compile_cache_hit": None,
            "compile_cache_hit_status": "unknown",
            "compile_cache_queries": None,
            "compile_cache_query_count": None,
            "compile_cache_hits": None,
            "compile_cache_hit_ratio": None,
            "preexisting_compile_key": None,
            "cache_ready": None,
            "cache_ready_status": "unknown",
            "cache_size_before_bytes": None,
            "cache_size_after_bytes": None,
            "cache_size_delta_bytes": None,
            "cache_limit_bytes": None,
            "canonical_grammar_key_sha256": None,
            "authority": "none",
            "source_field": source,
        }, []

    evidence = dict(raw)
    missing: list[str] = []
    status = evidence.get("compile_cache_hit_status")
    hit = evidence.get("compile_cache_hit")
    if status is None and type(hit) is bool:
        status = "hit" if hit else "miss"
    if status not in {"hit", "miss", "unknown"}:
        missing.append("compiler_snapshot.compile_cache_evidence.hit_status")
        status = "unknown"
    if status == "unknown":
        if hit is not None:
            missing.append("compiler_snapshot.compile_cache_evidence.hit_unknown_value")
        hit = None
    elif type(hit) is not bool:
        missing.append("compiler_snapshot.compile_cache_evidence.compile_cache_hit")

    queries = evidence.get("compile_cache_queries")
    query_count = evidence.get("compile_cache_query_count")
    if queries is not None and not _non_negative_int(queries):
        missing.append("compiler_snapshot.compile_cache_evidence.compile_cache_queries")
        queries = None
    if query_count is not None and not _non_negative_int(query_count):
        missing.append("compiler_snapshot.compile_cache_evidence.compile_cache_query_count")
        query_count = None
    if queries is None:
        queries = query_count

    hits = evidence.get("compile_cache_hits")
    if hits is not None and not _non_negative_int(hits):
        missing.append("compiler_snapshot.compile_cache_evidence.compile_cache_hits")
        hits = None
    ratio = evidence.get("compile_cache_hit_ratio")
    if ratio is not None and (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0 <= float(ratio) <= 1
    ):
        missing.append("compiler_snapshot.compile_cache_evidence.compile_cache_hit_ratio")
        ratio = None
    if hits is not None and queries is not None and hits > queries:
        missing.append("compiler_snapshot.compile_cache_evidence.hit_count")
    if status == "unknown" and (hits is not None or ratio is not None):
        missing.append("compiler_snapshot.compile_cache_evidence.unknown_hit_counters")

    for name in ("preexisting_compile_key", "cache_ready"):
        value = evidence.get(name)
        if value is not None and type(value) is not bool:
            missing.append(f"compiler_snapshot.compile_cache_evidence.{name}")
    for name in ("cache_size_before_bytes", "cache_size_after_bytes", "cache_limit_bytes"):
        value = evidence.get(name)
        if value is not None and not _non_negative_int(value):
            missing.append(f"compiler_snapshot.compile_cache_evidence.{name}")
    delta = evidence.get("cache_size_delta_bytes")
    if delta is not None and (isinstance(delta, bool) or not isinstance(delta, int)):
        missing.append("compiler_snapshot.compile_cache_evidence.cache_size_delta_bytes")
    key_sha = evidence.get("canonical_grammar_key_sha256")
    if key_sha is not None and (
        not isinstance(key_sha, str)
        or len(key_sha) != 64
        or any(char not in "0123456789abcdef" for char in key_sha)
    ):
        missing.append("compiler_snapshot.compile_cache_evidence.canonical_grammar_key_sha256")

    normalized = {
        "status": "observed",
        "compile_cache_hit": hit,
        "compile_cache_hit_status": status,
        "compile_cache_queries": queries,
        "compile_cache_query_count": query_count,
        "compile_cache_hits": hits,
        "compile_cache_hit_ratio": float(ratio) if ratio is not None else None,
        "preexisting_compile_key": evidence.get("preexisting_compile_key"),
        "preexisting_compile_key_source": str(
            evidence.get("preexisting_compile_key_source", "manager_compile_registry")
        ),
        "cache_ready": evidence.get("cache_ready"),
        "cache_ready_status": str(
            evidence.get(
                "cache_ready_status",
                "observed" if type(evidence.get("cache_ready")) is bool else "unknown",
            )
        ),
        "cache_size_before_bytes": evidence.get("cache_size_before_bytes"),
        "cache_size_after_bytes": evidence.get("cache_size_after_bytes"),
        "cache_size_delta_bytes": evidence.get("cache_size_delta_bytes"),
        "cache_limit_bytes": evidence.get("cache_limit_bytes"),
        "canonical_grammar_key_sha256": key_sha,
        "request_id": evidence.get("request_id"),
        "authority": str(evidence.get("authority", "unknown")),
        "source_field": str(evidence.get("source_field", source)),
    }
    return normalized, sorted(set(missing))


def compiler_snapshot_from_backend(
    *,
    backend: Any,
    grammar: Any,
    request_id: str,
    engine_boot_id: str,
    config_epoch: int,
) -> dict[str, Any]:
    """Extract compiler and mask facts from one compiled backend object.

    The values in this receipt are read from the backend's compiled grammar
    and a freshly filled token bitmask.  No schema-size, configured limit, or
    plan-derived value is used as a compiler substitute.
    """

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id is required for compiler snapshot")
    if not isinstance(engine_boot_id, str) or not engine_boot_id.strip():
        raise ValueError("engine_boot_id is required for compiler snapshot")
    if type(config_epoch) is not int or config_epoch < 0:
        raise ValueError("config_epoch is required for compiler snapshot")

    context = getattr(grammar, "ctx", None)
    serialize_json = getattr(context, "serialize_json", None)
    memory_size_bytes = getattr(context, "memory_size_bytes", None)
    if not callable(serialize_json) or type(memory_size_bytes) is not int:
        raise ValueError("compiled backend context does not expose runtime metadata")
    serialized = serialize_json()
    if not isinstance(serialized, str) or not serialized:
        raise ValueError("compiled backend context serialization is missing")
    try:
        serialized_payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError("compiled backend context serialization is invalid") from error
    grammar_payload = (
        serialized_payload.get("grammar")
        if isinstance(serialized_payload, Mapping)
        else None
    )
    if not isinstance(grammar_payload, Mapping):
        raise ValueError("compiled backend context has no grammar payload")
    rules = grammar_payload.get("rules")
    expression_indptr = grammar_payload.get("grammar_expr_indptr")
    if not isinstance(rules, list) or not rules:
        raise ValueError("compiled backend grammar rules are missing")
    if not isinstance(expression_indptr, list) or not expression_indptr:
        raise ValueError("compiled backend grammar states are missing")

    backend_name = type(backend).__name__
    if not backend_name:
        raise ValueError("compiled backend name is missing")
    vocab_size = getattr(backend, "vocab_size", None)
    allocate_token_bitmask = getattr(backend, "allocate_token_bitmask", None)
    matcher = getattr(grammar, "matcher", None)
    fill_next_token_bitmask = getattr(matcher, "fill_next_token_bitmask", None)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("compiled backend vocabulary size is missing")
    if not callable(allocate_token_bitmask) or not callable(fill_next_token_bitmask):
        raise ValueError("compiled backend mask operations are missing")

    bitmask = allocate_token_bitmask(1)
    shape = getattr(bitmask, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != 1:
        raise ValueError("compiled backend returned an invalid mask shape")
    need_apply = fill_next_token_bitmask(bitmask, 0)
    if type(need_apply) is not bool:
        raise ValueError("compiled backend mask operation returned no readiness flag")
    try:
        values = bitmask.detach().cpu().numpy()
    except AttributeError:
        try:
            values = bitmask.cpu().numpy()
        except AttributeError:
            values = bitmask
    try:
        row = values[0]
        allowed_token_count = 0
        for token_id in range(vocab_size):
            word = int(row[token_id // 32]) & 0xFFFFFFFF
            allowed_token_count += (word >> (token_id % 32)) & 1
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("compiled backend mask contents are unavailable") from error

    cache_evidence = getattr(grammar, "compile_cache_evidence", None)
    if not isinstance(cache_evidence, Mapping):
        cache_evidence = {
            "status": "missing",
            "compile_cache_hit": None,
            "compile_cache_hit_status": "unknown",
            "compile_cache_queries": None,
            "compile_cache_query_count": None,
            "compile_cache_hits": None,
            "compile_cache_hit_ratio": None,
            "preexisting_compile_key": None,
            "cache_ready": None,
            "cache_ready_status": "unknown",
            "cache_size_before_bytes": None,
            "cache_size_after_bytes": None,
            "cache_size_delta_bytes": None,
            "cache_limit_bytes": None,
            "canonical_grammar_key_sha256": None,
            "authority": "none",
            "source_field": "structured_output_backend.compiler.cache_state",
        }

    return {
        "grammar_size_bytes": int(memory_size_bytes),
        "grammar_serialized_size_bytes": len(serialized.encode("utf-8")),
        "grammar_state_count": len(expression_indptr),
        "grammar_rule_count": len(rules),
        "grammar_branching_factor": allowed_token_count / vocab_size,
        "compile_ready": True,
        "backend_name": backend_name,
        "backend_effective": True,
        "grammar_compiled": True,
        "mask_ready": True,
        "mask_applied": False,
        "mask_rows": int(shape[0]),
        "compile_cache_evidence": dict(cache_evidence),
        "engine_boot_id": engine_boot_id,
        "config_epoch": config_epoch,
        "request_id": request_id,
        "source_fields": {
            "grammar_size_bytes": "structured_output_backend.compiled_context.memory_size_bytes",
            "grammar_serialized_size_bytes": "structured_output_backend.compiled_context.serialize_json",
            "grammar_state_count": "structured_output_backend.compiled_context.grammar.grammar_expr_indptr",
            "grammar_rule_count": "structured_output_backend.compiled_context.grammar.rules",
            "grammar_branching_factor": "structured_output_backend.grammar_matcher.fill_next_token_bitmask",
            "compile_ready": "structured_output_backend.compile_grammar.return",
            "backend_name": "structured_output_backend.instance",
            "backend_effective": "structured_output_backend.compile_grammar.return",
            "grammar_compiled": "structured_output_backend.compile_grammar.return",
            "mask_ready": "structured_output_backend.grammar_matcher.fill_next_token_bitmask",
            "mask_applied": "pre_action_before_model_logits",
            "mask_rows": "structured_output_backend.allocate_token_bitmask.shape",
        },
    }


def observe_provider_bound_schema_pre_action(
    *,
    schema_document: Mapping[str, Any],
    tool_count: int,
    requested_max_tokens: int,
    request_id: str | None,
    engine_boot_id: str | None,
    config_epoch: int | None,
    clock_id: str | None,
    observed_at_ns: int | None,
    decision_at_ns: int | None,
    compiler_snapshot: Mapping[str, Any] | None = None,
    runtime_snapshot: Mapping[str, Any] | None = None,
    require_compile_cache_state: bool = False,
) -> dict[str, Any]:
    """Build a provider-bound Schema pre-action receipt.

    Caller-visible schema facts are kept separate from compiler facts.  The
    latter are accepted only from an engine-owned snapshot and are never
    inferred from the JSON document or from a later mask event.
    """

    if not isinstance(schema_document, Mapping) or not schema_document:
        raise ValueError("schema_document is required for provider pre-action observation")
    if type(tool_count) is not int or tool_count < 0:
        raise ValueError("tool_count must be a non-negative integer")
    if type(requested_max_tokens) is not int or requested_max_tokens < 0:
        raise ValueError("requested_max_tokens must be a non-negative integer")
    canonical = json.dumps(
        schema_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    request_sha256 = _digest(
        {
            "schema_document": schema_document,
            "tool_count": tool_count,
            "requested_max_tokens": requested_max_tokens,
        }
    )
    # A caller may omit an opaque ID on this prompt-free route.  Bind the
    # deterministic request fingerprint instead of admitting an unbound row.
    request_id = request_id or "schema-observer:" + request_sha256[:24]
    binding, missing = _binding(
        request_id=request_id,
        request_sha256=request_sha256,
        engine_boot_id=engine_boot_id,
        config_epoch=config_epoch,
        clock_id=clock_id,
        observed_at_ns=observed_at_ns,
        decision_at_ns=decision_at_ns,
    )
    fields: dict[str, dict[str, Any]] = {
        "schema_size_bytes": _observed(
            len(canonical), source="request.schema_document.canonical_json"
        ),
        "schema_sha256": _observed(
            hashlib.sha256(canonical).hexdigest(),
            source="request.schema_document.canonical_json",
        ),
        "schema_count": _observed(1, source="request.schema_document"),
        "tool_count": _observed(tool_count, source="request.tools"),
        "requested_max_tokens": _observed(
            requested_max_tokens, source="request.max_tokens"
        ),
    }
    snapshot = compiler_snapshot if isinstance(compiler_snapshot, Mapping) else {}
    if (
        snapshot.get("engine_boot_id") is not None
        and snapshot.get("engine_boot_id") != engine_boot_id
    ):
        raise ValueError("compiler snapshot boot differs from EngineCore")
    if (
        snapshot.get("config_epoch") is not None
        and snapshot.get("config_epoch") != config_epoch
    ):
        raise ValueError("compiler snapshot epoch differs from EngineCore")
    compiler_sources = snapshot.get("source_fields", {})
    if not isinstance(compiler_sources, Mapping):
        compiler_sources = {}
    for name, default_source in (
        ("grammar_size_bytes", "structured_output_backend.pre_action_snapshot"),
        ("grammar_state_count", "structured_output_backend.pre_action_snapshot"),
        ("grammar_rule_count", "structured_output_backend.pre_action_snapshot"),
        ("grammar_branching_factor", "structured_output_backend.pre_action_snapshot"),
        ("compile_ready", "structured_output_backend.pre_action_snapshot"),
    ):
        source = str(compiler_sources.get(name, default_source))
        fields[name] = (
            _observed(snapshot[name], source=source)
            if name in snapshot
            else _missing(source)
        )
        if name not in snapshot:
            missing.append(f"compiler_snapshot.{name}")
        elif name in {
            "grammar_size_bytes",
            "grammar_state_count",
            "grammar_rule_count",
        } and not _non_negative_int(snapshot[name]):
            missing.append(f"compiler_snapshot.{name}")
        elif name == "grammar_branching_factor" and (
            not _finite_number(snapshot[name]) or float(snapshot[name]) < 0
        ):
            missing.append(f"compiler_snapshot.{name}")
        elif name == "compile_ready" and type(snapshot[name]) is not bool:
            missing.append(f"compiler_snapshot.{name}")

    evidence: dict[str, Any] = {}
    for name, default_source in (
        ("backend_name", "structured_output_backend.pre_action_snapshot"),
        ("backend_effective", "structured_output_backend.pre_action_snapshot"),
        ("grammar_compiled", "structured_output_backend.pre_action_snapshot"),
        ("mask_ready", "structured_output_backend.pre_action_snapshot"),
        ("mask_applied", "structured_output_backend.pre_action_snapshot"),
    ):
        source = str(compiler_sources.get(name, default_source))
        evidence[name] = (
            _observed(snapshot[name], source=source)
            if name in snapshot
            else _missing(source)
        )
        if name not in snapshot:
            missing.append(f"compiler_snapshot.{name}")
        elif name in {"backend_effective", "grammar_compiled", "mask_ready", "mask_applied"} and type(snapshot[name]) is not bool:
            missing.append(f"compiler_snapshot.{name}")
        elif name == "backend_name" and (
            not isinstance(snapshot[name], str) or not snapshot[name].strip()
        ):
            missing.append(f"compiler_snapshot.{name}")
    mask_rows_source = str(
        compiler_sources.get("mask_rows", "structured_output_backend.pre_action_snapshot")
    )
    if "mask_rows" in snapshot:
        evidence["mask_rows"] = _observed(
            snapshot["mask_rows"], source=mask_rows_source
        )
    else:
        evidence["mask_rows"] = _missing(mask_rows_source)
    if "mask_rows" in snapshot and not _non_negative_int(snapshot["mask_rows"]):
        missing.append("compiler_snapshot.mask_rows")
    if (
        "backend_effective" in snapshot
        and snapshot["backend_effective"] is not True
    ):
        missing.append("compiler_snapshot.backend_effective")
    if "grammar_compiled" in snapshot and snapshot["grammar_compiled"] is not True:
        missing.append("compiler_snapshot.grammar_compiled")
    if "mask_ready" in snapshot and snapshot["mask_ready"] is not True:
        missing.append("compiler_snapshot.mask_ready")
    if "compile_ready" in snapshot and snapshot["compile_ready"] is not True:
        missing.append("compiler_snapshot.compile_ready")

    # Keep compiler-only migration fields in the compiler evidence section.
    # The unified overlay intentionally does not search the generic request
    # fields for ``grammar_rule_count`` after this point.
    grammar_rule_field = fields.pop("grammar_rule_count", None)
    evidence["grammar_rule_count"] = (
        grammar_rule_field
        if isinstance(grammar_rule_field, Mapping)
        else _missing(
            str(
                compiler_sources.get(
                    "grammar_rule_count",
                    "structured_output_backend.pre_action_snapshot",
                )
            )
        )
    )

    raw_cache = snapshot.get("compile_cache_evidence")
    cache_evidence, cache_missing = _normalize_compile_cache_evidence(raw_cache)
    if cache_missing:
        missing.extend(cache_missing)
    if (
        cache_evidence.get("request_id") is not None
        and cache_evidence.get("request_id") != request_id
    ):
        missing.append("compiler_snapshot.compile_cache_evidence.request_id")
    if require_compile_cache_state and cache_evidence.get("status") != "observed":
        missing.append("compiler_snapshot.compile_cache_evidence")

    runtime = runtime_snapshot if isinstance(runtime_snapshot, Mapping) else {}
    runtime_sources = runtime.get("source_fields", {})
    if not isinstance(runtime_sources, Mapping):
        runtime_sources = {}
    runtime_evidence: dict[str, Any] = {}
    for name, default_source in (
        ("active_request_count", "engine_core.scheduler.running"),
        ("context_tokens", "engine_core.scheduler.requests"),
        ("runtime_pressure_ratio", "engine_core.scheduler.kv_cache_manager"),
    ):
        source = str(runtime_sources.get(name, default_source))
        if name not in runtime:
            runtime_evidence[name] = _missing(source)
            missing.append(f"runtime_snapshot.{name}")
            continue
        value = runtime[name]
        runtime_evidence[name] = _observed(value, source=source)
        if name in {"active_request_count", "context_tokens"}:
            if not _non_negative_int(value):
                missing.append(f"runtime_snapshot.{name}")
        elif name == "runtime_pressure_ratio" and (
            not _finite_number(value) or not 0 <= float(value) <= 1
        ):
            missing.append(f"runtime_snapshot.{name}")

    unique_missing = sorted(set(missing))
    return {
        "schema": PROVIDER_SCHEMA,
        "observation_point": "before_completion_request",
        "binding": binding,
        "fields": fields,
        "compiler_evidence": evidence,
        "compile_cache_evidence": cache_evidence,
        "runtime_evidence": runtime_evidence,
        "missing_fields": unique_missing,
        "status": "READY" if not unique_missing else "NOT_READY",
        "send_allowed": not unique_missing,
        "model_calls": 0,
        "remote": False,
    }


def observe_prefix_pre_action(
    *,
    engine_snapshot: Mapping[str, Any],
    session_plan: Mapping[str, Any] | None,
    request_id: str | None,
    engine_boot_id: str | None,
    config_epoch: int | None,
    clock_id: str | None,
    observed_at_ns: int | None,
    decision_at_ns: int | None,
) -> dict[str, Any]:
    """Join engine KV pressure with a request/session plan before admission."""

    if not isinstance(engine_snapshot, Mapping):
        raise ValueError("engine_snapshot is required")
    pressure = engine_snapshot.get(
        "cache_pressure_ratio", engine_snapshot.get("cache_usage_ratio")
    )
    capacity = engine_snapshot.get("cache_capacity_ratio")
    request_sha256 = _digest(
        {
            "request_id": request_id,
            "session_plan": dict(session_plan or {}),
        }
    )
    binding, missing = _binding(
        request_id=request_id,
        request_sha256=request_sha256,
        engine_boot_id=engine_boot_id,
        config_epoch=config_epoch,
        clock_id=clock_id,
        observed_at_ns=observed_at_ns,
        decision_at_ns=decision_at_ns,
    )
    fields: dict[str, dict[str, Any]] = {
        "cache_pressure_ratio": (
            _observed(pressure, source="kv_cache_manager.block_pool.usage")
            if pressure is not None
            else _missing("kv_cache_manager.block_pool.usage")
        ),
        "cache_capacity_ratio": (
            _observed(capacity, source="kv_cache_manager.block_pool.free_blocks")
            if capacity is not None
            else _missing("kv_cache_manager.block_pool.free_blocks")
        ),
        "cached_block_hash_count": (
            _observed(
                engine_snapshot["cached_block_hash_count"],
                source="kv_cache_manager.block_pool.cached_block_hash_to_block",
            )
            if engine_snapshot.get("cached_block_hash_count") is not None
            else _missing("kv_cache_manager.block_pool.cached_block_hash_to_block")
        ),
    }
    if pressure is None:
        missing.append("engine_snapshot.cache_pressure_ratio")
    if capacity is None:
        missing.append("engine_snapshot.cache_capacity_ratio")
    if "cached_block_hash_count" not in engine_snapshot:
        missing.append("engine_snapshot.cached_block_hash_count")

    required_plan = (
        "planned_reuse_ratio",
        "prefix_tokens",
        "session_invocation_count",
        "kv_bytes",
        "namespace",
        "persistence_mode",
        "arrival_gap_ms",
    )
    plan = session_plan if isinstance(session_plan, Mapping) else {}
    for name in required_plan:
        if name not in plan or plan[name] in (None, ""):
            fields[name] = _missing("caller_session_plan." + name)
            missing.append("session_plan." + name)
        else:
            fields[name] = _observed(plan[name], source="caller_session_plan." + name)
            if name == "planned_reuse_ratio":
                if not _finite_number(plan[name]) or not 0 <= float(plan[name]) <= 1:
                    missing.append("session_plan." + name)
            elif name == "arrival_gap_ms":
                if not _finite_number(plan[name]) or float(plan[name]) < 0:
                    missing.append("session_plan." + name)
            elif name in {"prefix_tokens", "session_invocation_count", "kv_bytes"}:
                if not _non_negative_int(plan[name]):
                    missing.append("session_plan." + name)
            elif name in {"namespace", "persistence_mode"} and (
                not isinstance(plan[name], str) or not plan[name].strip()
            ):
                missing.append("session_plan." + name)

    if engine_snapshot.get("prefix_caching_enabled") is not True:
        missing.append("engine_snapshot.prefix_caching_enabled")
    fields["prefix_caching_enabled"] = (
        _observed(True, source="kv_cache_manager.enable_caching")
        if engine_snapshot.get("prefix_caching_enabled") is True
        else _missing("kv_cache_manager.enable_caching")
    )
    fields["prefix_caching_hash_algo"] = (
        _observed(
            engine_snapshot["prefix_caching_hash_algo"],
            source="vllm_config.cache_config.prefix_caching_hash_algo",
        )
        if isinstance(engine_snapshot.get("prefix_caching_hash_algo"), str)
        and engine_snapshot["prefix_caching_hash_algo"].strip()
        else _missing("vllm_config.cache_config.prefix_caching_hash_algo")
    )
    if fields["prefix_caching_hash_algo"]["status"] == "missing":
        missing.append("engine_snapshot.prefix_caching_hash_algo")
    fields["free_gpu_blocks"] = (
        _observed(engine_snapshot["free_gpu_blocks"], source="kv_cache_manager.block_pool")
        if "free_gpu_blocks" in engine_snapshot
        else _missing("kv_cache_manager.block_pool")
    )
    fields["capacity_blocks"] = (
        _observed(engine_snapshot["capacity_blocks"], source="kv_cache_manager.block_pool")
        if engine_snapshot.get("capacity_blocks") is not None
        else _missing("kv_cache_manager.block_pool")
    )
    if "free_gpu_blocks" in engine_snapshot and not _non_negative_int(
        engine_snapshot["free_gpu_blocks"]
    ):
        missing.append("engine_snapshot.free_gpu_blocks")
    if "capacity_blocks" in engine_snapshot and not _non_negative_int(
        engine_snapshot["capacity_blocks"]
    ):
        missing.append("engine_snapshot.capacity_blocks")
    if "cached_block_hash_count" in engine_snapshot and not _non_negative_int(
        engine_snapshot["cached_block_hash_count"]
    ):
        missing.append("engine_snapshot.cached_block_hash_count")
    if pressure is not None and (
        not _finite_number(pressure) or not 0 <= float(pressure) <= 1
    ):
        missing.append("engine_snapshot.cache_pressure_ratio")
    if capacity is not None and (
        not _finite_number(capacity) or not 0 <= float(capacity) <= 1
    ):
        missing.append("engine_snapshot.cache_capacity_ratio")
    unique_missing = sorted(set(missing))
    return {
        "schema": PREFIX_PROVIDER_SCHEMA,
        "observation_point": "before_completion_request",
        "binding": binding,
        # Flat aliases are intentional: provider-shaped GET/POST consumers can
        # inspect the identity without knowing the nested binding layout.  The
        # canonical runner still reads the nested fields and therefore keeps a
        # single source of truth.
        "provider_request_id": binding["provider_request_id"],
        "request_id": binding["request_id"],
        "request_sha256": binding["request_sha256"],
        "engine_boot_id": binding["engine_boot_id"],
        "config_epoch": binding["config_epoch"],
        "clock_id": binding["clock_id"],
        "observed_at_ns": binding["observed_at_ns"],
        "decision_at_ns": binding["decision_at_ns"],
        "runtime_receipt_contract": engine_snapshot.get("runtime_receipt_contract"),
        "fields": fields,
        "missing_fields": unique_missing,
        "status": "READY" if not unique_missing else "NOT_READY",
        "send_allowed": not unique_missing,
        "model_calls": 0,
        "remote": False,
    }


__all__ = [
    "SCHEMA",
    "PROVIDER_SCHEMA",
    "PREDICTOR_PRE_ACTION_SCHEMA",
    "PREFIX_PROVIDER_SCHEMA",
    "compiler_snapshot_from_backend",
    "observe_schema_pre_action",
    "observe_provider_bound_schema_pre_action",
    "observe_schema_predictor_pre_action_state",
    "observe_prefix_pre_action",
]
