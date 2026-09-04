# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-prediction helpers shared by the scheduler and KV cache manager.

A request may carry a predicted total output length (``Request.predicted_length``,
tokens). When present, the scheduler's admission-time "full sequence must fit"
check reserves enough blocks for *prompt + predicted output* instead of only the
prompt.  No prediction (``None``) keeps the upstream behaviour bit-for-bit.

The prediction value is expected to already include any safety margin the
caller wants (e.g. ``ceil(predicted * 1.2)``); this module stays agnostic.
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
    """Return the sequence length the admission check should reserve for.

    * ``predicted_length is None`` -> current token count (upstream behaviour).
    * prediction present and the request is still prefilling -> current tokens
      plus the predicted *remaining* output, capped at ``max_model_len``.
    * prediction present but the request already finished prefill -> current
      token count only: per-step decode allocation behaviour is unchanged and
      remains purely reactive, exactly as upstream.
    """
    current = request.num_tokens
    if predicted_length is None:
        return current
    if request.num_computed_tokens >= request.num_prompt_tokens:
        return current
    generated = max(0, current - request.num_prompt_tokens)
    extra = max(0, predicted_length - generated)
    return min(current + extra, max_model_len)
