"""Batch-scoped token-budget proposals: the write channel the batch scope lacked.

Why this module exists
----------------------
``scheduler.py`` already *reads* the token budget at batch formation
(``token_budget = self.max_num_scheduled_tokens``), so the read point is
batch-scoped in effect.  But the only writer of that value is the versioned
engine-epoch transaction, and an engine-epoch write is by construction all-worker
and long-lived.  There was therefore no way to say "for *this* formed batch, use
this budget", which is what the batch decision scope means.

This module supplies exactly that, and nothing more:

* a bounded book of proposals, each naming the batch it applies to (its member
  request ids) and the token budget that batch should use;
* a pure resolver that validates a proposed budget against the engine's own
  ``0..=startup`` ceiling and returns the receipt for it;
* a receipt bound to the batch proposal id, emitted whether the proposal was
  applied or refused, so "the batch scope did nothing here" is evidence rather
  than silence.

What it deliberately does not do
--------------------------------
* It never writes ``scheduler.max_num_scheduled_tokens``.  The scheduler hook
  uses the resolved value as the *local* budget for the step it is forming, so a
  batch-scoped decision cannot survive past its own batch and cannot be mistaken
  for an epoch change.
* It never widens the ceiling.  A proposal outside ``0..=startup_ceiling`` is
  refused with a receipt, not clamped, because clamping would silently disagree
  with what the caller asked for.
* It confers no admission.  Registering a proposal is not evidence for any of the
  versioned mechanism-admission gates, and the consumer must still be a
  qualified mechanism.

The module imports nothing from the engine so it can be reviewed and unit-tested
outside it.  Its deployed copy under ``vllm/v1/core/sched/`` must stay
byte-identical to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from copy import deepcopy
from functools import wraps
from threading import RLock
from typing import Any, Iterable, Mapping

SCHEMA_PROPOSAL = "anise.batch-scope-proposal.v1"
SCHEMA_RECEIPT = "anise.batch-scope-effective-receipt.v3"
SCHEMA_STATE = "qbi.batch-scope-proposal-state.v1"
SCHEMA_PRE_ACTION_OBSERVATION = "anise.batch-scope-pre-schedule-observation.v1"
SCHEMA_ACTUAL_BATCH = "anise.batch-scope-actual-schedule-receipt.v1"
#: Emitted inside the effective receipt when the matched batch holds at least one request whose
#: prompt is still being computed (the prefill target) and at least one request already past its
#: prompt (the decoding peer).  It is the native, same-scheduling-event evidence the Chunked
#: mechanism requires before a label may be recorded; the role split comes from the scheduler's own
#: view (`num_computed_tokens` vs `num_prompt_tokens`), never from the caller.
SCHEMA_OVERLAP_RECEIPT = "anise.target-prefill-peer-decode-overlap-receipt.v1"

ENABLED_ENV = "QBI_BATCH_SCOPE_PROPOSAL"
CAPACITY_ENV = "QBI_BATCH_SCOPE_PROPOSAL_CAPACITY"
DEFAULT_CAPACITY = 1024
MAX_CAPACITY = 65536
MIN_CAPACITY = 64
MAX_PROPOSAL_AGE_NS = 30_000_000_000

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

BUDGET_SOURCE_PROPOSAL = "batch_proposal"
BUDGET_SOURCE_SCHEDULER = "scheduler_default"

#: Member matching modes.  ``exact`` compares the declared ids with the batch's engine ids as an
#: exact set, which is all a caller can do once it knows them.  ``client_alias`` lets a caller bind a
#: batch *before* the engine has named its requests: the engine names a request
#: ``chatcmpl-<client id>-<nonce>``, so a declared client id can be matched against that shape.
#: The resolution must be one-to-one and cover the whole batch, otherwise nothing is consumed.
MEMBER_MATCH_EXACT = "exact"
MEMBER_MATCH_CLIENT_ALIAS = "client_alias"
MEMBER_MATCH_MODES = (MEMBER_MATCH_EXACT, MEMBER_MATCH_CLIENT_ALIAS)
_ENGINE_ID_PREFIX = "chatcmpl-"


class BatchScopeProposalError(ValueError):
    """Raised when a proposal cannot be accepted at all."""


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


def _int(name: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchScopeProposalError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise BatchScopeProposalError(f"{name} is below {minimum}")
    if maximum is not None and value > maximum:
        raise BatchScopeProposalError(f"{name} is above {maximum}")
    return value


def _members(value: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise BatchScopeProposalError("member_request_ids must be a sequence of ids")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise BatchScopeProposalError("member_request_ids must be non-empty text ids")
        normalized.append(item)
    if not normalized:
        raise BatchScopeProposalError("a batch proposal must name its members")
    if len(set(normalized)) != len(normalized):
        raise BatchScopeProposalError("a batch proposal must not repeat a member")
    return tuple(normalized)


def _optional_members(value: Iterable[str] | None) -> tuple[str, ...]:
    """Parse an optional role list without ever raising into the step loop.

    Unlike :func:`_members` these are observations, not a proposal: a caller that
    cannot supply them, or supplies something malformed, must not fail the batch.
    """

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return ()
    normalized = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def _seal_observation(value: Mapping[str, Any], *, expected_schema: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BatchScopeProposalError("scope observation must be a mapping")
    sealed = deepcopy(dict(value))
    supplied_digest = sealed.pop("sha256", None)
    if sealed.get("schema") != expected_schema:
        raise BatchScopeProposalError("scope observation schema is invalid")
    encoded = json.dumps(
        sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if supplied_digest is not None and supplied_digest != digest:
        raise BatchScopeProposalError("scope observation digest does not match its contents")
    sealed["sha256"] = digest
    return sealed


def _member_match_mode(value: Any) -> str:
    if value is None:
        return MEMBER_MATCH_EXACT
    if not isinstance(value, str) or value not in MEMBER_MATCH_MODES:
        raise BatchScopeProposalError(f"member_match must be one of {MEMBER_MATCH_MODES}")
    return value


def alias_matches_engine_id(alias: str, engine_request_id: str) -> bool:
    """Whether a declared client id names this engine request id.

    The engine names a request ``chatcmpl-<client id>-<nonce>``; a caller that declared its own
    client id therefore cannot know the nonce in advance.  Only this exact shape is accepted -- the
    alias itself, the alias with the response prefix, or the prefixed alias followed by ``-`` and the
    nonce -- so an alias can never match some unrelated id that merely contains it.
    """

    if engine_request_id == alias:
        return True
    prefixed = f"{_ENGINE_ID_PREFIX}{alias}"
    return engine_request_id == prefixed or engine_request_id.startswith(f"{prefixed}-")


def resolve_alias_members(aliases: tuple[str, ...], engine_ids: tuple[str, ...]) -> tuple[str, ...] | None:
    """Resolve declared aliases onto the batch's engine ids, one-to-one and complete.

    Returns the engine ids in the proposal's declared order, or ``None`` when the resolution is not a
    bijection: a partial or ambiguous match must never consume a proposal, because the proposal is
    allowed to bind exactly one batch and only one.
    """

    if len(aliases) != len(engine_ids):
        return None
    remaining = list(engine_ids)
    resolved: list[str] = []
    for alias in aliases:
        hits = [engine_id for engine_id in remaining if alias_matches_engine_id(alias, engine_id)]
        if len(hits) != 1:
            return None
        remaining.remove(hits[0])
        resolved.append(hits[0])
    return tuple(resolved) if not remaining else None


def resolve_batch_token_budget(
    *,
    proposal_budget: Any,
    scheduler_budget: Any,
    startup_ceiling: Any,
) -> tuple[int, dict[str, Any]]:
    """Validate one proposed budget and return ``(budget, receipt fields)``.

    The accepted range is ``0..=startup_ceiling``, the same ceiling the scheduler
    itself uses: the startup value is the only bound that survives every committed
    epoch, so a batch proposal may narrow the budget but never raise it above what
    the engine was started with.  A budget of ``0`` is legal and means "form no
    new tokens this step", which the scheduler already handles.

    The returned mapping records that no instance attribute was written, so a
    reader never has to infer that from prose.
    """
    ceiling = _int("startup_ceiling", startup_ceiling, minimum=1)
    scheduler_value = _int("scheduler_budget", scheduler_budget, minimum=0)
    proposed = _int("proposal_budget", proposal_budget, minimum=0)
    if proposed > ceiling:
        raise BatchScopeProposalError(
            f"proposed budget {proposed} is above the startup ceiling {ceiling}"
        )
    if proposed > scheduler_value:
        raise BatchScopeProposalError(
            f"proposed budget {proposed} is above the current scheduler budget {scheduler_value}"
        )
    return proposed, {
        "budget_source": BUDGET_SOURCE_PROPOSAL,
        "token_budget": proposed,
        "scheduler_budget": scheduler_value,
        "startup_ceiling": ceiling,
        "narrowed_by": scheduler_value - proposed,
        "instance_attribute_written": False,
    }


class BatchScopeProposal:
    """One batch-scoped budget proposal, bound to the batch it belongs to."""

    __slots__ = (
        "batch_proposal_id",
        "member_request_ids",
        "token_budget",
        "config_epoch",
        "created_at_ns",
        "member_match",
        "require_prefill_decode_overlap",
        "target_prefill_member_id",
        "peer_decode_member_id",
    )

    def __init__(
        self,
        *,
        batch_proposal_id: str,
        member_request_ids: Iterable[str],
        token_budget: Any,
        config_epoch: Any,
        created_at_ns: Any,
        member_match: Any = None,
        require_prefill_decode_overlap: Any = False,
        target_prefill_member_id: Any = None,
        peer_decode_member_id: Any = None,
    ) -> None:
        if not isinstance(batch_proposal_id, str) or not batch_proposal_id.strip():
            raise BatchScopeProposalError("batch_proposal_id must be non-empty text")
        self.batch_proposal_id = batch_proposal_id
        self.member_request_ids = _members(member_request_ids)
        self.token_budget = _int("token_budget", token_budget, minimum=0)
        self.config_epoch = _int("config_epoch", config_epoch, minimum=0)
        self.created_at_ns = _int("created_at_ns", created_at_ns, minimum=0)
        self.member_match = _member_match_mode(member_match)
        if not isinstance(require_prefill_decode_overlap, bool):
            raise BatchScopeProposalError("require_prefill_decode_overlap must be a boolean")
        self.require_prefill_decode_overlap = require_prefill_decode_overlap
        for field_name, value in (
            ("target_prefill_member_id", target_prefill_member_id),
            ("peer_decode_member_id", peer_decode_member_id),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or value not in self.member_request_ids
            ):
                raise BatchScopeProposalError(
                    f"{field_name} must name one of the proposal members"
                )
        if require_prefill_decode_overlap:
            if (
                not isinstance(target_prefill_member_id, str)
                or not isinstance(peer_decode_member_id, str)
                or target_prefill_member_id == peer_decode_member_id
            ):
                raise BatchScopeProposalError(
                    "overlap-gated proposals need distinct target-prefill and peer-decode members"
                )
        self.target_prefill_member_id = target_prefill_member_id
        self.peer_decode_member_id = peer_decode_member_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_PROPOSAL,
            "batch_proposal_id": self.batch_proposal_id,
            "member_request_ids": list(self.member_request_ids),
            "token_budget": self.token_budget,
            "config_epoch": self.config_epoch,
            "created_at_ns": self.created_at_ns,
            "member_match": self.member_match,
            "require_prefill_decode_overlap": self.require_prefill_decode_overlap,
            "target_prefill_member_id": self.target_prefill_member_id,
            "peer_decode_member_id": self.peer_decode_member_id,
        }


class BatchScopeProposalBook:
    """Bounded, request-set-addressed proposals with one receipt per attempt.

    A proposal is bound to a member set, not to a point in time: the scheduler
    forms batches from whatever is in its own view, so the only stable way to
    address "this batch" from outside is the set of requests the caller intends it
    to cover.  A proposal is consumed by the first batch whose member set it
    matches exactly, so it can affect one batch and only one.
    """

    def __init__(self, *, enabled: bool, capacity: int) -> None:
        if not isinstance(enabled, bool):
            raise BatchScopeProposalError("enabled must be a boolean flag")
        _int("capacity", capacity, minimum=MIN_CAPACITY, maximum=MAX_CAPACITY)
        self.enabled = enabled
        self.capacity = capacity
        self.sequence = 0
        self.dropped_receipts = 0
        self.refused_proposals = 0
        self.applied_proposals = 0
        self._pending: deque[BatchScopeProposal] = deque()
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = RLock()

    @classmethod
    def from_environment(cls) -> "BatchScopeProposalBook":
        raw = os.environ.get(ENABLED_ENV, "0").strip().lower()
        if raw not in _TRUE_VALUES | _FALSE_VALUES:
            raise BatchScopeProposalError(f"{ENABLED_ENV} must be a boolean flag")
        capacity_text = os.environ.get(CAPACITY_ENV, str(DEFAULT_CAPACITY)).strip()
        if not capacity_text.isdecimal():
            raise BatchScopeProposalError(f"{CAPACITY_ENV} must be an integer")
        return cls(enabled=raw in _TRUE_VALUES, capacity=int(capacity_text))

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @_synchronized
    def propose(
        self,
        *,
        batch_proposal_id: str,
        member_request_ids: Iterable[str],
        token_budget: Any,
        config_epoch: Any,
        created_at_ns: Any,
        member_match: Any = None,
        require_prefill_decode_overlap: Any = False,
        target_prefill_member_id: Any = None,
        peer_decode_member_id: Any = None,
    ) -> dict[str, Any]:
        """Accept one proposal, or raise before anything is stored."""
        proposal = BatchScopeProposal(
            batch_proposal_id=batch_proposal_id,
            member_request_ids=member_request_ids,
            token_budget=token_budget,
            config_epoch=config_epoch,
            created_at_ns=created_at_ns,
            member_match=member_match,
            require_prefill_decode_overlap=require_prefill_decode_overlap,
            target_prefill_member_id=target_prefill_member_id,
            peer_decode_member_id=peer_decode_member_id,
        )
        if len(self._pending) >= self.capacity:
            raise BatchScopeProposalError("pending batch proposal capacity is full")
        if any(existing.batch_proposal_id == proposal.batch_proposal_id for existing in self._pending):
            raise BatchScopeProposalError("batch proposal id is already pending")
        self._pending.append(proposal)
        return proposal.as_dict()

    def _take_match(
        self,
        member_request_ids: tuple[str, ...],
        *,
        prefill_request_ids: tuple[str, ...],
        decode_request_ids: tuple[str, ...],
    ) -> tuple[BatchScopeProposal, tuple[str, ...]] | None:
        """Consume the first pending proposal that binds exactly this batch.

        ``exact`` needs the declared ids to *be* the engine ids.  ``client_alias`` resolves declared
        client ids onto the engine ids one-to-one; a partial or ambiguous resolution matches nothing,
        so a proposal can never be consumed by a batch it does not fully describe.
        """

        wanted = set(member_request_ids)
        for index, proposal in enumerate(self._pending):
            if proposal.member_match == MEMBER_MATCH_EXACT:
                resolved = tuple(proposal.member_request_ids) if set(proposal.member_request_ids) == wanted else None
                if resolved is None:
                    continue
            else:
                resolved = resolve_alias_members(proposal.member_request_ids, member_request_ids)
                if resolved is None:
                    continue
            if proposal.require_prefill_decode_overlap:
                resolved_by_declared = dict(zip(proposal.member_request_ids, resolved))
                target_id = resolved_by_declared.get(proposal.target_prefill_member_id)
                peer_id = resolved_by_declared.get(proposal.peer_decode_member_id)
                if target_id not in prefill_request_ids or peer_id not in decode_request_ids:
                    continue
            del self._pending[index]
            return proposal, resolved
        return None

    @_synchronized
    def cancel(self, batch_proposal_id: str) -> dict[str, Any]:
        """Remove a proposal that the caller will no longer send to the engine."""

        if not isinstance(batch_proposal_id, str) or not batch_proposal_id.strip():
            raise BatchScopeProposalError("batch_proposal_id must be non-empty text")
        for index, proposal in enumerate(self._pending):
            if proposal.batch_proposal_id == batch_proposal_id:
                del self._pending[index]
                return {
                    "schema": "qbi.batch-scope-proposal-cancellation.v1",
                    "batch_proposal_id": batch_proposal_id,
                    "cancelled": True,
                    "status": "cancelled_before_match",
                    "pending_proposal_count": len(self._pending),
                }
        return {
            "schema": "qbi.batch-scope-proposal-cancellation.v1",
            "batch_proposal_id": batch_proposal_id,
            "cancelled": False,
            "status": "not_pending_already_consumed_or_expired",
            "pending_proposal_count": len(self._pending),
        }

    @_synchronized
    def observe(
        self,
        *,
        member_request_ids: Iterable[str],
        scheduler_budget: Any,
        startup_ceiling: Any,
        config_epoch: Any,
        observed_at_ns: Any,
        prefill_request_ids: Iterable[str] | None = None,
        decode_request_ids: Iterable[str] | None = None,
        pre_action_observation: Mapping[str, Any] | None = None,
        engine_boot_id: Any = None,
    ) -> dict[str, Any] | None:
        """Resolve a candidate schedule budget and record a provisional receipt.

        Returns the receipt when a pending proposal matched, otherwise ``None`` so
        the scheduler keeps its own budget and pays nothing for a disabled or
        unrelated proposal. A usable override is not marked effective until the
        scheduler confirms the actual scheduled request set after schedule().
        A matched proposal that fails validation produces an immediate refusal.
        """
        if not self.enabled:
            return None
        try:
            members = _members(member_request_ids)
        except BatchScopeProposalError:
            return None
        prefill = _optional_members(prefill_request_ids)
        decode = _optional_members(decode_request_ids)
        match = self._take_match(
            members,
            prefill_request_ids=prefill,
            decode_request_ids=decode,
        )
        if match is None:
            return None
        proposal, resolved_member_ids = match
        observed = _int("observed_at_ns", observed_at_ns, minimum=0)
        epoch = _int("config_epoch", config_epoch, minimum=0)
        scheduler_value = _int("scheduler_budget", scheduler_budget, minimum=0)
        ceiling = _int("startup_ceiling", startup_ceiling, minimum=1)
        if engine_boot_id is not None and (
            not isinstance(engine_boot_id, str) or not engine_boot_id.strip()
        ):
            raise BatchScopeProposalError("engine_boot_id must be non-empty text when supplied")
        if pre_action_observation is None:
            pre_action_observation = {
                "schema": SCHEMA_PRE_ACTION_OBSERVATION,
                "stage": "pre_schedule_candidate_set",
                "source": "proposal_book_aggregate_only",
                "clock_id": "monotonic_ns",
                "observed_at_ns": observed,
                "engine_boot_id": engine_boot_id,
                "config_epoch": epoch,
                "scheduler_token_budget": scheduler_value,
                "startup_token_ceiling": ceiling,
                "candidate_request_ids": list(members),
                "prefill_request_ids": list(prefill),
                "decode_request_ids": list(decode),
                "candidate_count": len(members),
                "queue_counts": None,
                "request_candidates": [],
                "detail_status": "aggregate_only",
            }
        pre_action_observation = _seal_observation(
            pre_action_observation, expected_schema=SCHEMA_PRE_ACTION_OBSERVATION
        )
        if (
            pre_action_observation.get("stage") != "pre_schedule_candidate_set"
            or pre_action_observation.get("observed_at_ns") != observed
            or pre_action_observation.get("config_epoch") != epoch
            or pre_action_observation.get("scheduler_token_budget") != scheduler_value
            or pre_action_observation.get("startup_token_ceiling") != ceiling
            or set(pre_action_observation.get("candidate_request_ids", ())) != set(members)
            or pre_action_observation.get("engine_boot_id") != engine_boot_id
        ):
            raise BatchScopeProposalError(
                "pre-action observation does not match the proposal schedule context"
            )
        # Record the actual resolution performed by this proposal book, before
        # applying the local budget. Clients must not infer aliases themselves.
        native_snapshot_sha256 = pre_action_observation["sha256"]
        binding_identity = {
            "native_candidate_snapshot_sha256": native_snapshot_sha256,
            "batch_proposal_id": proposal.batch_proposal_id,
            "caller_member_ids": list(proposal.member_request_ids),
            "resolved_member_ids": list(resolved_member_ids),
        }
        observation_id = hashlib.sha256(json.dumps(
            binding_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        bound_observation = {
            key: value for key, value in pre_action_observation.items() if key != "sha256"
        }
        bound_observation.update({
            "observation_id": observation_id,
            "native_candidate_snapshot_sha256": native_snapshot_sha256,
            "member_request_bindings": [
                {
                    "schema": "anise.batch-scope-member-request-binding.v1",
                    "caller_request_id": caller_id,
                    "native_request_id": native_id,
                    "observation_id": observation_id,
                    "batch_proposal_id": proposal.batch_proposal_id,
                    "binding_source": "scheduler_proposal_book_resolution",
                }
                for caller_id, native_id in zip(proposal.member_request_ids, resolved_member_ids)
            ],
        })
        pre_action_observation = _seal_observation(
            bound_observation, expected_schema=SCHEMA_PRE_ACTION_OBSERVATION
        )
        proposal_age_ns = observed - proposal.created_at_ns
        if proposal.config_epoch != epoch:
            budget = scheduler_value
            fields = {
                "budget_source": BUDGET_SOURCE_SCHEDULER,
                "token_budget": scheduler_value,
                "scheduler_budget": scheduler_value,
                "startup_ceiling": ceiling,
                "narrowed_by": 0,
                "instance_attribute_written": False,
            }
            budget_accepted = False
            reason = (
                "proposal config epoch differs from observed scheduler epoch: "
                f"proposal={proposal.config_epoch}, observed={epoch}"
            )
        elif proposal_age_ns < 0 or proposal_age_ns > MAX_PROPOSAL_AGE_NS:
            budget = scheduler_value
            fields = {
                "budget_source": BUDGET_SOURCE_SCHEDULER,
                "token_budget": scheduler_value,
                "scheduler_budget": scheduler_value,
                "startup_ceiling": ceiling,
                "narrowed_by": 0,
                "instance_attribute_written": False,
            }
            budget_accepted = False
            reason = (
                "batch proposal is outside its monotonic lifetime: "
                f"age_ns={proposal_age_ns}, max_age_ns={MAX_PROPOSAL_AGE_NS}"
            )
        else:
            try:
                budget, fields = resolve_batch_token_budget(
                    proposal_budget=proposal.token_budget,
                    scheduler_budget=scheduler_value,
                    startup_ceiling=ceiling,
                )
                budget_accepted = True
                reason = None
            except BatchScopeProposalError as error:
                fields = {
                    "budget_source": BUDGET_SOURCE_SCHEDULER,
                    "token_budget": scheduler_value,
                    "scheduler_budget": scheduler_value,
                    "startup_ceiling": ceiling,
                    "narrowed_by": 0,
                    "instance_attribute_written": False,
                }
                budget_accepted = False
                reason = str(error)
        self.sequence += 1
        if len(self._receipts) == self.capacity:
            self.dropped_receipts += 1
        receipt = {
            "schema": SCHEMA_RECEIPT,
            "sequence": self.sequence,
            "batch_proposal_id": proposal.batch_proposal_id,
            "member_request_ids": list(proposal.member_request_ids),
            "member_match_mode": proposal.member_match,
            "resolved_member_request_ids": list(resolved_member_ids),
            "matched_member_count": len(resolved_member_ids),
            "proposed_budget": proposal.token_budget,
            "proposal_created_at_ns": proposal.created_at_ns,
            "proposal_age_ns": proposal_age_ns,
            "max_proposal_age_ns": MAX_PROPOSAL_AGE_NS,
            "require_prefill_decode_overlap": proposal.require_prefill_decode_overlap,
            "target_prefill_member_id": proposal.target_prefill_member_id,
            "peer_decode_member_id": proposal.peer_decode_member_id,
            "proposal_config_epoch": proposal.config_epoch,
            "observation_config_epoch": epoch,
            "observed_at_ns": observed,
            "clock_id": "monotonic_ns",
            "engine_boot_id": engine_boot_id,
            "pre_action_batch_observation": pre_action_observation,
            "effective": None if budget_accepted else False,
            "consumer_applied": None if budget_accepted else False,
            "budget_override_applied": None if budget_accepted else False,
            "schedule_confirmation_required": budget_accepted,
            "schedule_confirmation_status": "awaiting" if budget_accepted else "not_required",
            "schedule_confirmed": False,
            "status": "awaiting_schedule_confirmation" if budget_accepted else "proposal_refused_before_schedule",
            "actual_scheduled_request_ids": None,
            "actual_scheduled_token_counts": None,
            "actual_total_num_scheduled_tokens": None,
            "applied_token_budget": budget if budget_accepted else None,
            "budget_limit_exhausted": None,
            "reason": "awaiting actual Scheduler.schedule confirmation" if budget_accepted else reason,
            **fields,
        }
        if not budget_accepted:
            self.refused_proposals += 1
        # Additive role evidence: which members of this one formed batch were still prefilling and
        # which were already decoding, taken from the scheduler's own per-request token counters.
        receipt["prefill_request_ids"] = list(prefill)
        receipt["decode_request_ids"] = list(decode)
        if prefill and decode:
            resolved_by_declared = dict(zip(proposal.member_request_ids, resolved_member_ids))
            target_id = (
                resolved_by_declared[proposal.target_prefill_member_id]
                if proposal.require_prefill_decode_overlap
                else prefill[0]
            )
            peer_id = (
                resolved_by_declared[proposal.peer_decode_member_id]
                if proposal.require_prefill_decode_overlap
                else decode[0]
            )
            receipt["target_prefill_peer_decode_overlap_receipt"] = {
                "schema": SCHEMA_OVERLAP_RECEIPT,
                "batch_proposal_id": proposal.batch_proposal_id,
                "observation_id": observation_id,
                "caller_target_request_id": dict(zip(resolved_member_ids, proposal.member_request_ids)).get(target_id),
                "caller_peer_request_id": dict(zip(resolved_member_ids, proposal.member_request_ids)).get(peer_id),
                "target_request_id": target_id,
                "peer_request_id": peer_id,
                "same_scheduling_event": True,
                "member_request_ids": list(proposal.member_request_ids),
                "observed_at_ns": observed,
                "clock_id": "monotonic_ns",
            }
        self._receipts.append(receipt)
        return receipt

    @_synchronized
    def confirm_scheduled_batch(
        self,
        *,
        receipt_sequence: Any,
        scheduled_token_counts: Mapping[str, Any],
        applied_token_budget: Any,
        actual_schedule_observed_at_ns: Any = None,
    ) -> dict[str, Any]:
        """Mark a proposal effective only after the scheduler returns its actual batch."""

        sequence = _int("receipt_sequence", receipt_sequence, minimum=1)
        budget = _int("applied_token_budget", applied_token_budget, minimum=0)
        if not isinstance(scheduled_token_counts, Mapping):
            raise BatchScopeProposalError("scheduled_token_counts must be a request-to-token mapping")
        counts: dict[str, int] = {}
        for request_id, value in scheduled_token_counts.items():
            if not isinstance(request_id, str) or not request_id:
                raise BatchScopeProposalError("scheduled request ids must be non-empty text")
            counts[request_id] = _int(
                f"scheduled_token_counts[{request_id}]", value, minimum=1
            )

        receipt = next(
            (item for item in self._receipts if int(item.get("sequence", -1)) == sequence),
            None,
        )
        if receipt is None:
            raise BatchScopeProposalError("schedule confirmation receipt is no longer available")
        if (receipt.get("schedule_confirmation_status") != "awaiting"
                or receipt.get("schedule_confirmation_required") is not True):
            raise BatchScopeProposalError("receipt is not awaiting schedule confirmation")

        expected_ids = tuple(receipt.get("resolved_member_request_ids", ()))
        actual_ids = tuple(counts)
        exact_member_set = (
            len(actual_ids) == len(expected_ids)
            and len(set(actual_ids)) == len(actual_ids)
            and set(actual_ids) == set(expected_ids)
        )
        total_tokens = sum(counts.values())
        pre_action = receipt.get("pre_action_batch_observation")
        if not isinstance(pre_action, Mapping):
            raise BatchScopeProposalError("receipt is missing its pre-action batch observation")
        if actual_schedule_observed_at_ns is None:
            actual_observed = int(pre_action["observed_at_ns"]) + 1
        else:
            actual_observed = _int(
                "actual_schedule_observed_at_ns", actual_schedule_observed_at_ns, minimum=0
            )
        if actual_observed < int(pre_action["observed_at_ns"]):
            raise BatchScopeProposalError("actual schedule receipt predates its pre-action observation")
        candidate_ids = set(pre_action.get("candidate_request_ids", ()))
        actual_members_subset = set(counts).issubset(candidate_ids)
        overlap = receipt.get("target_prefill_peer_decode_overlap_receipt")
        overlap_scheduled = (
            not receipt.get("require_prefill_decode_overlap")
            or (
                isinstance(overlap, Mapping)
                and overlap.get("target_request_id") in counts
                and overlap.get("peer_request_id") in counts
            )
        )
        budget_matches = (
            budget == receipt.get("token_budget")
            and receipt.get("budget_source") == BUDGET_SOURCE_PROPOSAL
        )
        consumer_applied = bool(budget_matches)
        effective = bool(
            consumer_applied
            and exact_member_set
            and actual_members_subset
            and overlap_scheduled
            and total_tokens > 0
            and total_tokens <= budget
        )
        if effective:
            reason = None
        elif not exact_member_set:
            reason = "actual scheduled request set differs from the proposal members"
        elif not actual_members_subset:
            reason = "actual scheduled request set contains a non-candidate member"
        elif not overlap_scheduled:
            reason = "target-prefill/peer-decode members were not both scheduled"
        elif not budget_matches:
            reason = "scheduler did not use the proposal token budget"
        elif total_tokens <= 0:
            reason = "no scheduled tokens were confirmed within the proposal budget"
        elif total_tokens > budget:
            reason = "actual scheduled token count exceeds the applied proposal budget"
        else:
            reason = "schedule confirmation failed an unclassified consumer check"

        actual_batch = {
            "schema": SCHEMA_ACTUAL_BATCH,
            "observation_id": pre_action.get("observation_id"),
            "stage": "post_schedule_actual_batch",
            "source": "Scheduler.schedule",
            "clock_id": "monotonic_ns",
            "observed_at_ns": actual_observed,
            "engine_boot_id": receipt.get("engine_boot_id"),
            "config_epoch": receipt.get("observation_config_epoch"),
            "batch_proposal_id": receipt.get("batch_proposal_id"),
            "pre_action_observation_sha256": pre_action.get("sha256"),
            "candidate_request_ids": list(pre_action.get("candidate_request_ids", ())),
            "actual_scheduled_request_ids": list(actual_ids),
            "actual_scheduled_token_counts": counts,
            "actual_total_num_scheduled_tokens": total_tokens,
            "applied_token_budget": budget,
            "actual_members_subset_of_candidates": actual_members_subset,
        }
        actual_batch = _seal_observation(
            actual_batch, expected_schema=SCHEMA_ACTUAL_BATCH
        )

        receipt.update({
            "effective": effective,
            "consumer_applied": consumer_applied,
            "budget_override_applied": consumer_applied,
            "schedule_confirmation_required": False,
            "schedule_confirmation_status": "confirmed",
            "schedule_confirmed": True,
            "actual_scheduled_request_ids": list(actual_ids),
            "actual_scheduled_token_counts": counts,
            "actual_total_num_scheduled_tokens": total_tokens,
            "actual_schedule_observed_at_ns": actual_observed,
            "actual_formed_batch": actual_batch,
            "applied_token_budget": budget,
            "budget_limit_exhausted": total_tokens == budget,
            "reason": reason,
        })
        if consumer_applied:
            self.applied_proposals += 1
            receipt["status"] = (
                "NATIVE_BATCH_PROPOSAL_EFFECTIVE"
                if effective else "BATCH_BUDGET_APPLIED_EFFECT_NOT_CONFIRMED"
            )
        else:
            self.refused_proposals += 1
            receipt["status"] = "SCHEDULE_CONFIRMED_PROPOSAL_NOT_EFFECTIVE"
        return deepcopy(receipt)

    @_synchronized
    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise BatchScopeProposalError("batch-scope cursor must be an integer")
        if after_sequence < 0:
            raise BatchScopeProposalError("batch-scope cursor must be non-negative")
        receipts = [
            deepcopy(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": SCHEMA_STATE,
            "observer_enabled": self.enabled,
            "capacity": self.capacity,
            "max_proposal_age_ns": MAX_PROPOSAL_AGE_NS,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "dropped_receipts": self.dropped_receipts,
            "pending_proposals": [proposal.as_dict() for proposal in self._pending],
            "pending_proposal_count": len(self._pending),
            "applied_proposals": self.applied_proposals,
            "refused_proposals": self.refused_proposals,
            "schedule_confirmation_pending_count": sum(
                receipt.get("schedule_confirmation_status") == "awaiting"
                for receipt in self._receipts
            ),
            "instance_attribute_written_by_any_batch_proposal": False,
            "receipts": receipts,
        }


def batch_scope_state_of(book: "BatchScopeProposalBook | None", after_sequence: int = 0) -> Mapping[str, Any]:
    """Read the book's state, or report an unconfigured book honestly."""
    if book is None:
        return {
            "schema": SCHEMA_STATE,
            "observer_enabled": False,
            "capacity": 0,
            "max_proposal_age_ns": MAX_PROPOSAL_AGE_NS,
            "oldest_available_sequence": 0,
            "latest_sequence": 0,
            "dropped_receipts": 0,
            "pending_proposals": [],
            "pending_proposal_count": 0,
            "applied_proposals": 0,
            "refused_proposals": 0,
            "instance_attribute_written_by_any_batch_proposal": False,
            "receipts": [],
            "unconfigured_reason": f"{ENABLED_ENV} is not set on this engine process",
        }
    return book.state(after_sequence)


__all__ = [
    "BUDGET_SOURCE_PROPOSAL",
    "BUDGET_SOURCE_SCHEDULER",
    "CAPACITY_ENV",
    "DEFAULT_CAPACITY",
    "ENABLED_ENV",
    "MAX_CAPACITY",
    "MAX_PROPOSAL_AGE_NS",
    "MEMBER_MATCH_CLIENT_ALIAS",
    "MEMBER_MATCH_EXACT",
    "MEMBER_MATCH_MODES",
    "MIN_CAPACITY",
    "SCHEMA_OVERLAP_RECEIPT",
    "SCHEMA_PROPOSAL",
    "SCHEMA_RECEIPT",
    "SCHEMA_STATE",
    "BatchScopeProposal",
    "BatchScopeProposalBook",
    "BatchScopeProposalError",
    "alias_matches_engine_id",
    "batch_scope_state_of",
    "resolve_alias_members",
    "resolve_batch_token_budget",
]
