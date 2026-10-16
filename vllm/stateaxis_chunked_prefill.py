# SPDX-License-Identifier: Apache-2.0
"""Default-off StateAxis contention-aware chunked-prefill contract."""

from dataclasses import dataclass
from typing import Any


MOD_ID = "org.vllm-hust.stateaxis-chunked-prefill"


@dataclass(frozen=True, slots=True)
class ChunkedPrefillExperiment:
    chunk_tokens: int
    contention_only: bool
    inplace_continuation: bool


@dataclass(frozen=True, slots=True)
class ChunkedPrefillDecision:
    num_new_tokens: int
    chunks_applied: int = 0
    tokens_deferred: int = 0
    bypassed_no_contention: int = 0


def parse_chunked_prefill_experiment(
    additional_config: dict[str, Any] | object,
    *,
    device_type: str,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
) -> ChunkedPrefillExperiment | None:
    """Validate the ECPA-bound experimental policy, or return OFF."""

    if not isinstance(additional_config, dict):
        return None
    raw = additional_config.get("stateaxis_chunked_prefill")
    if raw is None:
        return None
    binding = additional_config.get("stateaxis_mod")
    if not isinstance(binding, dict) or binding.get("mod_id") != MOD_ID:
        raise ValueError(
            "stateaxis_chunked_prefill requires its matching StateAxis MOD binding"
        )
    if additional_config.get("experiment_mode") is not True:
        raise ValueError("stateaxis_chunked_prefill requires experiment_mode")
    manifest_sha256 = binding.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError("stateaxis_chunked_prefill requires a SHA-256 manifest binding")
    if not isinstance(raw, dict) or set(raw) != {
        "chunk_tokens",
        "contention_only",
        "inplace_continuation",
    }:
        raise ValueError("stateaxis_chunked_prefill has unexpected or missing fields")
    chunk_tokens = raw["chunk_tokens"]
    if (
        not isinstance(chunk_tokens, int)
        or isinstance(chunk_tokens, bool)
        or not 0 < chunk_tokens <= max_num_batched_tokens
    ):
        raise ValueError(
            "stateaxis_chunked_prefill.chunk_tokens must be a positive integer "
            "within max_num_batched_tokens"
        )
    if raw["contention_only"] is not True:
        raise ValueError("stateaxis_chunked_prefill currently requires contention_only")
    if raw["inplace_continuation"] is not True:
        raise ValueError(
            "stateaxis_chunked_prefill currently requires inplace_continuation"
        )
    if device_type != "npu":
        raise ValueError("stateaxis_chunked_prefill currently supports Ascend NPU")
    if not enable_chunked_prefill:
        raise ValueError("stateaxis_chunked_prefill requires chunked prefill support")
    return ChunkedPrefillExperiment(
        chunk_tokens=chunk_tokens,
        contention_only=True,
        inplace_continuation=True,
    )


def apply_chunked_prefill_limit(
    experiment: ChunkedPrefillExperiment,
    num_new_tokens: int,
    *,
    active_decode: bool,
) -> ChunkedPrefillDecision:
    """Return a pure scheduling decision without mutating request state."""

    if num_new_tokens <= experiment.chunk_tokens:
        return ChunkedPrefillDecision(num_new_tokens)
    if experiment.contention_only and not active_decode:
        return ChunkedPrefillDecision(
            num_new_tokens,
            bypassed_no_contention=1,
        )
    return ChunkedPrefillDecision(
        experiment.chunk_tokens,
        chunks_applied=1,
        tokens_deferred=num_new_tokens - experiment.chunk_tokens,
    )
