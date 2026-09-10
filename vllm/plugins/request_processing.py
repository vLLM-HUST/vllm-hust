# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public request-processing hooks for trusted in-process plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

REQUEST_PROCESSING_HOOK_API_VERSION = "1.0"

RequestEndpoint = Literal["chat", "completion", "responses"]
RequestProcessor = Callable[["RequestProcessingContext"], Mapping[str, Any] | None]

_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authorization",
        "x-api-key",
    }
)


@dataclass(frozen=True, slots=True)
class RequestProcessingContext:
    """Stable, minimal request metadata exposed before engine submission."""

    endpoint: RequestEndpoint
    request_id: str
    prompt_tokens: int
    max_tokens: int
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _RegisteredRequestProcessor:
    processor: RequestProcessor
    header_names: frozenset[str]


_request_processors: dict[str, _RegisteredRequestProcessor] = {}


def register_request_processor(
    name: str,
    processor: RequestProcessor,
    *,
    header_names: Iterable[str] = (),
) -> None:
    """Register one idempotent processor during general-plugin loading.

    Processors receive only explicitly requested, non-sensitive HTTP headers.
    Header names are normalized to lowercase before registration and delivery.
    """

    if not isinstance(name, str) or not name.strip() or not callable(processor):
        raise ValueError("request processor requires a name and callable")
    requested_headers = tuple(header_names)
    if any(not isinstance(header, str) for header in requested_headers):
        raise ValueError("request processor header names must be strings")
    normalized_headers = frozenset(
        header.strip().lower() for header in requested_headers
    )
    if "" in normalized_headers:
        raise ValueError("request processor header names must be non-empty")
    sensitive = normalized_headers & _SENSITIVE_HEADERS
    if sensitive:
        raise ValueError(
            f"request processor cannot access sensitive headers: {sorted(sensitive)}"
        )

    registration = _RegisteredRequestProcessor(processor, normalized_headers)
    existing = _request_processors.get(name)
    if existing is not None and existing != registration:
        raise ValueError(f"request processor {name!r} is already registered")
    _request_processors[name] = registration


def apply_request_processors(
    context: RequestProcessingContext,
    extra_args: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge processor-owned engine metadata without silent overwrites."""

    merged = dict(extra_args or {})
    normalized_headers = {key.lower(): value for key, value in context.headers.items()}
    for name, registration in _request_processors.items():
        allowed_headers = MappingProxyType(
            {
                key: normalized_headers[key]
                for key in registration.header_names
                if key in normalized_headers
            }
        )
        updates = registration.processor(replace(context, headers=allowed_headers))
        if updates is None:
            continue
        if not isinstance(updates, Mapping):
            raise TypeError(f"request processor {name!r} must return a mapping or None")
        if any(not isinstance(key, str) for key in updates):
            raise TypeError(f"request processor {name!r} returned a non-string key")
        conflicts = {
            key
            for key, value in updates.items()
            if key in merged and merged[key] != value
        }
        if conflicts:
            raise ValueError(
                f"request processor {name!r} conflicts with extra_args keys: "
                f"{sorted(conflicts)}"
            )
        merged.update(updates)
    return merged or None


__all__ = [
    "REQUEST_PROCESSING_HOOK_API_VERSION",
    "RequestEndpoint",
    "RequestProcessingContext",
    "RequestProcessor",
    "apply_request_processors",
    "register_request_processor",
]
