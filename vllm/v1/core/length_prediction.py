# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-prediction helper for the prediction-aware admission gate.

A request may carry a predicted total output length
(``Request.predicted_length``, in tokens) supplied by an external predictor
before admission. When present, the scheduler's "the full sequence must fit"
admission check reserves room for *prompt + predicted output* instead of only
what the request has announced so far. Without a prediction (``None``) the
check is bit-for-bit upstream behaviour.

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
pin blocks beyond the step being scheduled. Clamping rather than rejecting
therefore keeps a mis-predicted request schedulable instead of failing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


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
