# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


class SharedExecutionMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    tenant: str | None = None
    canonical_prefix_key: str | None = None
    retrieval_signature: tuple[str, ...] = ()
    tool_signature: tuple[str, ...] = ()
    stage_id: str | None = None
    priority_hint: int | None = None

    def has_canonical_prefix(self) -> bool:
        return bool(self.canonical_prefix_key)

    def shares_canonical_prefix_with(
        self,
        other: "SharedExecutionMetadata | None",
    ) -> bool:
        return (
            other is not None
            and self.canonical_prefix_key is not None
            and self.canonical_prefix_key == other.canonical_prefix_key
        )


def _normalize_signature(
    value: Sequence[str] | None,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"shared_execution.{field_name} must contain only strings, "
                f"got {type(item).__name__}"
            )
        text = item.strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _get_extra_payload(
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
) -> Any:
    if sampling_params is not None and sampling_params.extra_args is not None:
        return sampling_params.extra_args.get("shared_execution")
    if pooling_params is not None and pooling_params.extra_kwargs is not None:
        return pooling_params.extra_kwargs.get("shared_execution")
    return None


def extract_shared_execution_metadata(
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
) -> SharedExecutionMetadata | None:
    payload = _get_extra_payload(sampling_params, pooling_params)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(
            "shared_execution metadata must be a mapping passed via "
            "SamplingParams.extra_args['shared_execution'] or "
            "PoolingParams.extra_kwargs['shared_execution']"
        )

    tenant = payload.get("tenant")
    if tenant is not None and not isinstance(tenant, str):
        raise ValueError("shared_execution.tenant must be a string")

    canonical_prefix_key = payload.get("canonical_prefix_key")
    if canonical_prefix_key is None:
        canonical_prefix_key = payload.get("prefix_key")
    if canonical_prefix_key is not None and not isinstance(canonical_prefix_key, str):
        raise ValueError("shared_execution.canonical_prefix_key must be a string")

    stage_id = payload.get("stage_id")
    if stage_id is not None and not isinstance(stage_id, str):
        raise ValueError("shared_execution.stage_id must be a string")

    priority_hint = payload.get("priority_hint")
    if priority_hint is not None and not isinstance(priority_hint, int):
        raise ValueError("shared_execution.priority_hint must be an int")

    retrieval_signature = _normalize_signature(
        payload.get("retrieval_signature") or payload.get("retrieval_keys"),
        field_name="retrieval_signature",
    )
    tool_signature = _normalize_signature(
        payload.get("tool_signature") or payload.get("tool_calls"),
        field_name="tool_signature",
    )

    metadata = SharedExecutionMetadata(
        tenant=tenant,
        canonical_prefix_key=canonical_prefix_key.strip()
        if isinstance(canonical_prefix_key, str) and canonical_prefix_key.strip()
        else None,
        retrieval_signature=retrieval_signature,
        tool_signature=tool_signature,
        stage_id=stage_id.strip() if isinstance(stage_id, str) and stage_id.strip() else None,
        priority_hint=priority_hint,
    )

    if (
        metadata.tenant is None
        and metadata.canonical_prefix_key is None
        and not metadata.retrieval_signature
        and not metadata.tool_signature
        and metadata.stage_id is None
        and metadata.priority_hint is None
    ):
        return None
    return metadata