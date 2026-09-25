# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-sequence admission sizing for explicitly known output budgets.

Follows the prediction-aware reserve mechanism from Xie Hanlong's
f17a541f2b55aecb6800c9a55849f3a0d60c96e0, using declared ignore_eos budgets
instead of learned predictions. This is a capacity check, not eager allocation
or an exclusive reservation held across subsequent scheduler iterations.
"""


def admission_sequence_tokens(
    request, max_model_len: int, *, reserve_output_budget=False
):
    current = min(request.num_tokens, max_model_len)
    if not reserve_output_budget:
        return current
    params = request.sampling_params
    if params is None or params.ignore_eos is not True:
        return current
    if request.num_computed_tokens >= request.num_prompt_tokens:
        return current
    remaining = max(0, request.max_tokens - request.num_output_tokens)
    return min(request.num_tokens + remaining, max_model_len)
