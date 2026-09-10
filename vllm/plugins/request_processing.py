# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public request-processing hooks for trusted in-process plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

REQUEST_PROCESSING_HOOK_API_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class RequestProcessingContext:
    """Stable request metadata exposed before engine submission."""

    endpoint: Literal["chat", "completion"]
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    headers: Mapping[str, str]

    @property
    def prompt_tokens(self) -> int:
        """Return the number of rendered prompt tokens."""

        return len(self.prompt_token_ids)


RequestProcessor = Callable[[RequestProcessingContext], Mapping[str, Any] | None]

_request_processors: dict[str, RequestProcessor] = {}


def register_request_processor(name: str, processor: RequestProcessor) -> None:
    """Register one idempotent processor during general-plugin loading."""

    if not name or not callable(processor):
        raise ValueError("request processor requires a name and callable")
    existing = _request_processors.get(name)
    if existing is not None and existing is not processor:
        raise ValueError(f"request processor {name!r} is already registered")
    _request_processors[name] = processor


def apply_request_processors(
    context: RequestProcessingContext,
    extra_args: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge processor-owned engine metadata without silent overwrites."""

    merged = dict(extra_args or {})
    for name, processor in _request_processors.items():
        updates = processor(context)
        if updates is None:
            continue
        if not isinstance(updates, Mapping):
            raise TypeError(f"request processor {name!r} must return a mapping or None")
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
