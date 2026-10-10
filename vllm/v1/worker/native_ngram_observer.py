"""Read-only, pre-lookup evidence for the native N-gram proposer.

The observer deliberately lives in the worker process.  Scheduler counters and
request identifiers are useful for joining a receipt, but they are not a
substitute for the token corpus that the proposer actually reads.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def _tokens(value: Sequence[int], field: str) -> list[int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a token sequence")
    result = list(value)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{field} contains an invalid token id")
    return result


def _longest_suffix_match(
    current: list[int], corpus: list[int], min_n: int, max_n: int
) -> int:
    """Return the native proposer's longest repeated suffix length."""
    best = 0
    upper = min(max_n, len(current), len(corpus))
    for length in range(max(1, min_n), upper + 1):
        suffix = current[-length:]
        for index in range(len(corpus) - length + 1):
            # The native algorithm does not match the current suffix against
            # itself; it searches an earlier occurrence in the same corpus.
            if (
                corpus == current
                and index == len(corpus) - length
            ):
                continue
            if corpus[index : index + length] == suffix:
                best = length
                break
    return best


def _repeated_ngram_ratio(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for index in range(len(tokens) - n + 1):
        key = tuple(tokens[index : index + n])
        counts[key] = counts.get(key, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / float(len(tokens) - n + 1)


def _lookup_work_units(
    current_tokens: list[int],
    corpus_tokens: list[int],
    min_n: int,
    max_n: int,
) -> tuple[int, int]:
    """Count bounded candidate-window work before the n-gram lookup.

    One work unit is one token comparison in a candidate window for one
    configured n-gram size.  This is deliberately derived only from the
    current/corpus lengths and lookup bounds; draft and accepted tokens are
    not available at this observation point and must not be substituted.
    """

    del current_tokens  # The corpus windows are the lookup work surface.
    effective_max = min(max_n, len(corpus_tokens))
    windows = 0
    work = 0
    for ngram_size in range(min_n, effective_max + 1):
        candidate_windows = max(0, len(corpus_tokens) - ngram_size + 1)
        windows += candidate_windows
        work += ngram_size * candidate_windows
    return windows, work


class NativeNgramWorkerObserver:
    """Bounded worker-local receipts captured before native lookup."""

    def __init__(
        self,
        *,
        enabled: bool,
        capacity: int,
        request_prefix: str,
        worker_rank: int | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("native N-gram observer enabled must be boolean")
        if type(capacity) is not int or not 128 <= capacity <= 65536:
            raise ValueError("native N-gram observer capacity must be in [128, 65536]")
        if not str(request_prefix).strip():
            raise ValueError("native N-gram request prefix must not be empty")
        if worker_rank is not None and (
            type(worker_rank) is not int or worker_rank < 0
        ):
            raise ValueError(
                "native N-gram worker rank must be a non-negative integer"
            )
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = str(request_prefix).strip()
        self.worker_rank = worker_rank
        self.sequence = 0
        self.dropped_receipts = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls, *, worker_rank: int | None = None
    ) -> NativeNgramWorkerObserver:
        raw = os.environ.get("QBI_NATIVE_NGRAM_OBSERVER", "0").strip().lower()
        if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("QBI_NATIVE_NGRAM_OBSERVER must be a boolean flag")
        capacity_text = os.environ.get(
            "QBI_NATIVE_NGRAM_OBSERVER_CAPACITY", "8192"
        ).strip()
        if not capacity_text.isdecimal():
            raise ValueError("QBI_NATIVE_NGRAM_OBSERVER_CAPACITY must be an integer")
        return cls(
            enabled=raw in {"1", "true", "yes", "on"},
            capacity=int(capacity_text),
            request_prefix=os.environ.get(
                "QBI_NATIVE_NGRAM_REQUEST_PREFIX",
                os.environ.get(
                    "QBI_NATIVE_RECAPTURE_REQUEST_PREFIX",
                    "chatcmpl-anise-native-recapture-",
                ),
            ),
            worker_rank=worker_rank,
        )

    def record_pre_lookup_batch(
        self,
        *,
        request_ids: Sequence[str],
        request_slots: Sequence[int],
        current_token_ids: Sequence[Sequence[int]],
        lookup_token_ids: Sequence[Sequence[int]],
        draft_config: Mapping[str, Any],
        config_epoch: int | None,
        backend: str,
    ) -> dict[int, int]:
        """Freeze raw proposer inputs immediately before its lookup call.

        The returned slot-to-sequence map lets the proposer attach its result
        count after the call without replacing the pre-action evidence.
        """
        if not self.enabled:
            return {}
        if not (
            len(request_ids)
            == len(request_slots)
            == len(current_token_ids)
            == len(lookup_token_ids)
        ):
            raise ValueError("native N-gram observer batch lengths differ")
        if type(config_epoch) is not int or config_epoch < 0:
            epoch_value: int | None = None
        else:
            epoch_value = config_epoch
        captured: dict[int, int] = {}
        now = time.monotonic_ns()
        with self._lock:
            for request_id, slot, current, corpus in zip(
                request_ids, request_slots, current_token_ids, lookup_token_ids
            ):
                request_id = str(request_id)
                if not request_id.startswith(self.request_prefix):
                    continue
                if type(slot) is not int or slot < 0:
                    continue
                current_tokens = _tokens(current, "current_token_ids")
                corpus_tokens = _tokens(corpus, "lookup_token_ids")
                self.sequence += 1
                if len(self._receipts) == self.capacity:
                    self.dropped_receipts += 1
                max_n = int(
                    draft_config.get("prompt_lookup_max", len(current_tokens))
                )
                suffix = current_tokens[-max_n:] if max_n > 0 else []
                min_n = int(draft_config.get("prompt_lookup_min", 1))
                lookup_candidate_window_count, lookup_work_units = _lookup_work_units(
                    current_tokens, corpus_tokens, min_n, max_n
                )
                receipt = {
                    "schema": "qbi.native-ngram-worker-pre-lookup.v1",
                    "sequence": self.sequence,
                    "worker_rank": self.worker_rank,
                    "provider_request_id": request_id,
                    "request_slot": slot,
                    "backend": str(backend),
                    "observation_point": "worker_pre_lookup_before_drafter_call",
                    "observation_is_execution_completion": False,
                    "clock_id": "worker.monotonic_ns",
                    "observed_at_ns": now,
                    "decision_at_ns": now,
                    "config_epoch": epoch_value,
                    "epoch_status": (
                        "observed" if epoch_value is not None else "missing"
                    ),
                    "draft_config": deepcopy(dict(draft_config)),
                    "current_input_token_ids": current_tokens,
                    "current_input_suffix_token_ids": suffix,
                    "lookup_token_corpus": corpus_tokens,
                    "current_input_token_count": len(current_tokens),
                    "lookup_token_count": len(corpus_tokens),
                    "current_input_suffix_token_count": len(suffix),
                    "lookup_candidate_window_count": lookup_candidate_window_count,
                    "lookup_work_units": lookup_work_units,
                    "lookup_work_units_definition": (
                        "sum(ngram_size * candidate_windows) over configured "
                        "pre-lookup ngram sizes"
                    ),
                    "suffix_match_length": _longest_suffix_match(
                        current_tokens,
                        corpus_tokens,
                        int(draft_config.get("prompt_lookup_min", 1)),
                        max_n,
                    ),
                    "repeated_ngram_ratio": _repeated_ngram_ratio(current_tokens),
                    "lookup_result": {"status": "pending", "draft_token_count": None},
                }
                self._receipts.append(receipt)
                captured[slot] = self.sequence
        return captured

    def record_lookup_results(
        self,
        slot_to_sequence: Mapping[int, int],
        draft_counts: Mapping[int, int],
    ) -> None:
        if not self.enabled or not slot_to_sequence:
            return
        with self._lock:
            by_sequence = {int(row["sequence"]): row for row in self._receipts}
            for slot, sequence in slot_to_sequence.items():
                row = by_sequence.get(int(sequence))
                count = draft_counts.get(slot)
                if row is None or type(count) is not int or count < 0:
                    continue
                row["lookup_result"] = {
                    "status": "observed",
                    "draft_token_count": count,
                    "result_observed_at_ns": time.monotonic_ns(),
                }

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError(
                "native N-gram sequence cursor must be a non-negative integer"
            )
        with self._lock:
            return {
                "schema": "qbi.native-ngram-worker-state.v1",
                "status": "READY" if self.enabled else "NOT_READY",
                "observer_enabled": self.enabled,
                "worker_rank": self.worker_rank,
                "request_prefix": self.request_prefix,
                "capacity": self.capacity,
                "oldest_available_sequence": (
                    int(self._receipts[0]["sequence"]) if self._receipts else 0
                ),
                "latest_sequence": self.sequence,
                "dropped_receipts": self.dropped_receipts,
                "scope": "worker_pre_lookup",
                "receipts": [
                    deepcopy(row)
                    for row in self._receipts
                    if int(row["sequence"]) > after_sequence
                ],
                "field_status": {
                    "current_input_suffix": "READY",
                    "lookup_token_corpus": "READY",
                    "lookup_work_units": "READY",
                    "suffix_match_length": "READY",
                    "repeated_ngram_ratio": "READY",
                    "draft_config": "READY",
                    "clock": "READY",
                    "config_epoch": "READY" if self.enabled else "NOT_READY",
                    "post_lookup_draft_count": (
                        "observed_after_lookup" if self.enabled else "NOT_READY"
                    ),
                },
                "forbidden_substitutes": [
                    "request_id",
                    "prompt_token_id_hash",
                    "scheduled_token_count",
                    "accepted_token_count",
                ],
            }


__all__ = ["NativeNgramWorkerObserver"]
