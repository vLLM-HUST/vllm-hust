# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed host contract for request-scoped KV materialization control."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from vllm.logger import init_logger

logger = init_logger(__name__)

KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION = "1.0"
KV_MATERIALIZATION_RUNTIME_CONTROL_KEY = "kv_materialization_runtime_control"

MaterializationDecision = Literal["full_reuse", "partial_reuse", "recompute"]
MaterializationRuntimeObserver = Callable[[Mapping[str, object]], None]

_CONTROL_FIELDS = {
    "schema_version",
    "observed_decision",
    "effective_decision",
    "control_path",
    "cache_salt",
    "decision_supported",
    "support_tier",
    "fallback_reason",
    "target_reuse_tokens",
    "target_tail_tokens",
    "segmented_tail_cache_salt",
    "requires_segmented_materialization",
}


@dataclass(frozen=True, slots=True)
class KVMaterializationRuntimeControl:
    """Validated request-scoped policy result consumed by prefix caching."""

    schema_version: int
    observed_decision: MaterializationDecision
    effective_decision: MaterializationDecision
    control_path: str
    cache_salt: str | None
    decision_supported: bool
    support_tier: str
    fallback_reason: str | None
    target_reuse_tokens: int
    target_tail_tokens: int
    segmented_tail_cache_salt: str | None
    requires_segmented_materialization: bool

    def reuse_boundary(self, hash_block_size: int) -> int | None:
        """Return the maximum reusable prefix, aligned for the host cache."""

        if hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        if self.effective_decision == "recompute":
            return 0
        if self.effective_decision != "partial_reuse":
            return None
        return (self.target_reuse_tokens // hash_block_size) * hash_block_size


_runtime_observers: dict[str, MaterializationRuntimeObserver] = {}


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def parse_kv_materialization_runtime_control(
    value: Any,
    *,
    prompt_tokens: int,
) -> KVMaterializationRuntimeControl | None:
    """Validate and parse plugin metadata before prefix-cache use."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{KV_MATERIALIZATION_RUNTIME_CONTROL_KEY} must be an object")
    fields = set(value)
    if any(not isinstance(field, str) for field in fields):
        raise ValueError("KV materialization runtime-control keys must be strings")
    unknown = fields - _CONTROL_FIELDS
    missing = _CONTROL_FIELDS - fields
    if unknown or missing:
        raise ValueError(
            f"{KV_MATERIALIZATION_RUNTIME_CONTROL_KEY} has "
            f"unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported KV materialization runtime-control schema")

    decisions = {"full_reuse", "partial_reuse", "recompute"}
    observed = value["observed_decision"]
    effective = value["effective_decision"]
    if (
        not isinstance(observed, str)
        or not isinstance(effective, str)
        or observed not in decisions
        or effective not in decisions
    ):
        raise ValueError("KV materialization decisions are invalid")

    target_reuse = value["target_reuse_tokens"]
    target_tail = value["target_tail_tokens"]
    if type(target_reuse) is not int or type(target_tail) is not int:
        raise ValueError("KV materialization token boundaries must be integers")
    if target_reuse < 0 or target_tail < 0:
        raise ValueError("KV materialization token boundaries must be non-negative")
    if target_reuse + target_tail != prompt_tokens:
        raise ValueError(
            "KV materialization token boundaries must cover the rendered prompt"
        )
    if effective == "recompute" and target_reuse != 0:
        raise ValueError("recompute requires target_reuse_tokens=0")
    if effective == "partial_reuse" and not 0 < target_reuse < prompt_tokens:
        raise ValueError("partial_reuse requires an interior reuse boundary")

    tail_salt = value["segmented_tail_cache_salt"]
    if tail_salt is not None and not isinstance(tail_salt, str):
        raise ValueError("segmented_tail_cache_salt must be a string or null")
    if effective == "partial_reuse" and not tail_salt:
        raise ValueError("partial_reuse requires segmented_tail_cache_salt")

    fallback_reason = value["fallback_reason"]
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise ValueError("fallback_reason must be a string or null")
    if (
        type(value["decision_supported"]) is not bool
        or type(value["requires_segmented_materialization"]) is not bool
    ):
        raise ValueError("KV materialization flags must be booleans")
    cache_salt = value["cache_salt"]
    if cache_salt is not None and (not isinstance(cache_salt, str) or not cache_salt):
        raise ValueError("cache_salt must be a non-empty string or null")

    return KVMaterializationRuntimeControl(
        schema_version=1,
        observed_decision=cast(MaterializationDecision, observed),
        effective_decision=cast(MaterializationDecision, effective),
        control_path=_require_string(value["control_path"], "control_path"),
        cache_salt=cache_salt,
        decision_supported=value["decision_supported"],
        support_tier=_require_string(value["support_tier"], "support_tier"),
        fallback_reason=fallback_reason,
        target_reuse_tokens=target_reuse,
        target_tail_tokens=target_tail,
        segmented_tail_cache_salt=tail_salt,
        requires_segmented_materialization=value["requires_segmented_materialization"],
    )


def register_kv_materialization_runtime_observer(
    name: str,
    observer: MaterializationRuntimeObserver,
) -> None:
    """Register an idempotent engine-owned runtime receipt observer."""

    if not isinstance(name, str) or not name.strip() or not callable(observer):
        raise ValueError("KV materialization observer requires a name and callable")
    existing = _runtime_observers.get(name)
    if existing is not None and existing is not observer:
        raise ValueError(f"KV materialization observer {name!r} is already registered")
    _runtime_observers[name] = observer


def emit_kv_materialization_runtime_event(payload: Mapping[str, object]) -> None:
    """Notify observers without making observability failures fatal."""

    for name, observer in _runtime_observers.items():
        try:
            observer(payload)
        except Exception:
            logger.exception("KV materialization observer %r failed", name)


__all__ = [
    "KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION",
    "KV_MATERIALIZATION_RUNTIME_CONTROL_KEY",
    "KVMaterializationRuntimeControl",
    "MaterializationDecision",
    "MaterializationRuntimeObserver",
    "emit_kv_materialization_runtime_event",
    "parse_kv_materialization_runtime_control",
    "register_kv_materialization_runtime_observer",
]
