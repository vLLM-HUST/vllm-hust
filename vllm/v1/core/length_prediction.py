# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-prediction helper for the prediction-aware admission gate.

A request may carry a predicted total output length
(``Request.predicted_length``, in tokens) supplied by an external predictor
before admission. When present, the scheduler's "the full sequence must fit"
admission check reserves room for *prompt + predicted output* instead of only
what the request has announced so far. Without a prediction (``None``) the
check is bit-for-bit upstream behaviour.

The prediction only affects scheduling while the scheduler runs its
full-sequence admission reservation (``scheduler_reserve_full_isl``, enabled
by default); with the reservation disabled the value is accepted but has no
scheduling effect. Untrusted values are checked at the boundary by
:func:`validate_predicted_length` - ``None`` or a non-negative integer, with
bools, floats, strings and negatives raising
:class:`~vllm.exceptions.VLLMValidationError` - so the scheduling path never
has to defend against a malformed prediction.

Semantics and bounds
--------------------
``predicted_length`` is an *expected total output length*, not an upper bound,
and this module is where the scheduler stops trusting the caller:

* the value is clamped to the request's own output budget
  (``Request.max_tokens``), so a caller cannot reserve more output space than
  the request is allowed to generate in the first place;
* the resulting sequence length is clamped to ``max_model_len``;
* the prediction is only honoured while the request is still prefilling. Once
  the request has produced output its real length is known, and per-step
  decode allocation stays purely reactive, exactly as upstream.

The gate decides *whether* a request may enter the running set; it does not
pin blocks beyond the step being scheduled. Clamping an oversized prediction
rather than rejecting it therefore keeps a request with a poor prediction
schedulable instead of failing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.v1.request import Request


PREDICTED_LENGTH_API_VERSION = "1.0"


def validate_predicted_length(value: int | float | bool | None) -> int | None:
    """Validate ``predicted_length``; return ``None`` if unset.

    The field is a non-negative integer token count. Bools (an ``int``
    subclass), floats, strings and negative numbers are rejected with
    :class:`~vllm.exceptions.VLLMValidationError`, so malformed input fails
    closed at the entry boundary instead of reaching the scheduler.
    """
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`predicted_length` must be a non-negative integer.",
            parameter="predicted_length",
            value=value,
        )
    if value < 0:
        raise VLLMValidationError(
            "`predicted_length` must be a non-negative integer.",
            parameter="predicted_length",
            value=value,
        )
    return value


# Annotation for the OpenAI protocol models: the same strict contract applied
# at Pydantic validation time. Untrusted JSON must be rejected (or accepted)
# here, not inside the scheduler.
PredictedLength = Annotated[int | None, BeforeValidator(validate_predicted_length)]


def predicted_full_sequence_tokens(
    request: Request,
    predicted_length: int | None,
    max_model_len: int,
) -> int:
    """Return the sequence length the full-sequence admission check plans for.

    Args:
        request: The request being admitted.
        predicted_length: Expected total output length in tokens, or ``None``
            when no prediction is available.
        max_model_len: The model's maximum sequence length.

    Returns:
        The number of tokens the admission check should plan for. Never
        exceeds ``max_model_len``, and never exceeds the prompt length plus
        the request's own output budget.

    """
    if predicted_length is not None and (
        isinstance(predicted_length, bool) or not isinstance(predicted_length, int)
    ):
        raise ValueError(
            "predicted_length must be an integer number of tokens or None, "
            f"got {predicted_length!r}"
        )
    current = min(request.num_tokens, max_model_len)
    if predicted_length is None:
        return current
    if request.num_computed_tokens >= request.num_prompt_tokens:
        # Decoding already started, so the real length is known: adding a
        # prediction here would only inflate the reserve.
        return current
    # Never trust a prediction beyond what the request may generate.
    predicted = max(0, min(predicted_length, request.max_tokens))
    generated = max(0, current - request.num_prompt_tokens)
    extra = max(0, predicted - generated)
    return min(current + extra, max_model_len)
