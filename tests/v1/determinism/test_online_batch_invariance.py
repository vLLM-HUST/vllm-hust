# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP-based batch invariance test: send requests to a running
vLLM server and compare BS=1 vs BS=N results (tokens and per-step logprobs).

Environment variables:
  - VLLM_TEST_MODEL: served model name (e.g., Qwen/Qwen3-1.7B / DeepSeek-R1)
  - VLLM_TP_SIZE: tensor parallelism size (e.g., 4)
  - VLLM_TEST_SERVER_ARGS: extra server args (e.g., "--data-parallel-size 4")

"""

import json
import os
import random
import sys
from typing import Any

import httpx
import openai
import pytest
from utils import BACKENDS, TEST_MODEL, _random_prompt, skip_if_not_cuda_alike

from tests.utils import RemoteOpenAIServer


def test_qwen_gdn_speculative_prefix_replay_preserves_greedy_tokens() -> None:
    """Real HTTP regression: prefill, cached replay and Suffix verification agree.

    This requires the same invariant numerical policy in every phase, rather
    than assuming greedy decoding alone makes a hybrid model deterministic.
    VLLM_TEST_MODEL and VLLM_TP_SIZE allow the original 35B/TP2 acceptance run.
    """
    model = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
    sequence = " ".join(f"item{i:03d}" for i in range(1, 41))
    prompts = [
        "This is a verbatim copy task. Preserve every word.\n"
        + ("Reference: " + sequence + "\n") * 12
        + f"Example {number}. Source: "
        + sequence
        + " END\nCopy:"
        for number in (1, 2)
    ]
    common = [
        "--host",
        "127.0.0.1",
        "--served-model-name",
        "gdn-parity",
        "--tensor-parallel-size",
        os.environ.get("VLLM_TP_SIZE", "1"),
        "--max-model-len",
        os.environ.get("VLLM_TEST_MAX_MODEL_LEN", "4096"),
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.85",
        "--distributed-executor-backend",
        "mp",
        "--block-size",
        "1088",
        "--max-num-batched-tokens",
        "1088",
        "--no-async-scheduling",
    ]
    env = {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_GDN_DECODE_KERNEL": "triton",
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    baseline = None
    prompt_ids = None

    def counter(metrics: str, name: str) -> float:
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        )

    for apc, suffix in ((False, False), (True, False), (False, True), (True, True)):
        args = common + [
            "--enable-prefix-caching" if apc else "--no-enable-prefix-caching"
        ]
        if suffix:
            args += [
                "--speculative-config",
                json.dumps({"method": "suffix", "num_speculative_tokens": 5}),
            ]
        with (
            RemoteOpenAIServer(model, args, env_dict=env) as server,
            httpx.Client(timeout=120, trust_env=False) as client,
        ):
            if prompt_ids is None:
                prompt_ids = []
                for prompt in prompts:
                    response = client.post(
                        server.url_for("tokenize"),
                        json={
                            "model": "gdn-parity",
                            "prompt": prompt,
                            "add_special_tokens": False,
                        },
                    )
                    response.raise_for_status()
                    prompt_ids.append(response.json()["tokens"])
            before = client.get(server.url_for("metrics"))
            before.raise_for_status()
            for _ in range(2):
                outputs = []
                for ids in prompt_ids:
                    token_ids: list[int] = []
                    text, finish, done = "", [], False
                    with client.stream(
                        "POST",
                        server.url_for("v1/completions"),
                        json={
                            "model": "gdn-parity",
                            "prompt": ids,
                            "temperature": 0,
                            "seed": 42,
                            "max_tokens": 512,
                            "add_special_tokens": False,
                            "stop": ["END", "\n\n"],
                            "stream": True,
                            "return_token_ids": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                done = True
                                continue
                            event = json.loads(data)
                            assert "error" not in event, event
                            for choice in event.get("choices", []):
                                text += choice.get("text", "")
                                token_ids.extend(choice.get("token_ids") or [])
                                if choice.get("finish_reason"):
                                    finish.append(choice["finish_reason"])
                    assert done and finish == ["stop"] and text and token_ids
                    outputs.append((text, token_ids))
                if baseline is None:
                    baseline = outputs
                assert outputs == baseline, (apc, suffix)
            after = client.get(server.url_for("metrics"))
            after.raise_for_status()
            names = []
            if apc:
                names.append("vllm:prefix_cache_hits_total")
            if suffix:
                names += [
                    "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_accepted_tokens_total",
                ]
            for name in names:
                assert counter(after.text, name) > counter(before.text, name), name


def _request_completion(
    client: openai.OpenAI,
    model: str,
    prompt: Any,
    sp: dict[str, Any],
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)

    for attempt in range(max_retries + 1):
        try:
            completion = client.completions.create(**payload)
            # Convert to plain dict so downstream logic can keep using
            # dict-style access just like with raw HTTP JSON.
            return completion.model_dump()
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"Error: {e}\n")
            return None
    return None


def _extract_tokens_and_logprobs(
    choice: dict[str, Any],
) -> tuple[list[Any], list[float] | None]:
    tokens: list[Any] = []
    token_logprobs: list[float] | None = None
    lp = choice.get("logprobs")
    if lp and isinstance(lp, dict):
        tokens = lp.get("token_ids") or lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs", None)
    return tokens, token_logprobs


def _compare_bs1_vs_bsn_single_process(
    prompts: list[str],
    sp_kwargs: dict[str, Any],
    client: openai.OpenAI,
    model_name: str,
) -> None:
    # BS=1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    for p in prompts:
        resp = _request_completion(client, model_name, p, sp_kwargs)
        if resp is None or not resp.get("choices"):
            raise AssertionError("BS=1 empty/failed response")
        choice = resp["choices"][0]
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(
                "logprobs not returned; ensure server supports 'logprobs'"
            )
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))

    # BS=N
    bsN_tokens_per_prompt: list[list[Any]] = [None] * len(prompts)  # type: ignore[list-item]
    bsN_logprobs_per_prompt: list[list[float] | None] = [None] * len(prompts)
    resp = _request_completion(client, model_name, prompts, sp_kwargs)
    if resp is None or not resp.get("choices"):
        raise AssertionError("BS=N empty/failed batched response")
    choices = resp.get("choices", [])
    if len(choices) != len(prompts):
        raise AssertionError(
            f"BS=N choices length {len(choices)} != num prompts {len(prompts)}"
        )
    for idx, choice in enumerate(choices):
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(f"BS=N missing logprobs for prompt {idx}")
        bsN_tokens_per_prompt[idx] = list(toks)
        bsN_logprobs_per_prompt[idx] = list(lps)

    # compare
    for i, (tokens_bs1, tokens_bsN, logprobs_bs1, logprobs_bsN) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
        )
    ):
        if tokens_bs1 != tokens_bsN:
            raise AssertionError(
                f"Prompt {i} (sampling): Different tokens sampled. "
                f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
            )
        if logprobs_bs1 is None or logprobs_bsN is None:
            raise AssertionError(f"Prompt {i}: Missing logprobs in one of the runs")
        if len(logprobs_bs1) != len(logprobs_bsN):
            raise AssertionError(
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS=N)."
            )
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a != b:
                diff = abs(a - b)
                raise AssertionError(
                    f"Prompt {i} Step {t}: Bitwise mismatch "
                    f"(abs diff={diff:.6e}). "
                    f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
                )


@skip_if_not_cuda_alike
@pytest.mark.parametrize("backend", BACKENDS)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    backend: str,
) -> None:
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))
    prompts_all = [_random_prompt(10, 50) for _ in range(32)]

    sp_kwargs: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 8,
        "seed": 42,
        "logprobs": 5,
    }

    tp_size = os.getenv("VLLM_TP_SIZE", "1")
    server_args: list[str] = [
        "--max-model-len=8192",
        "--max-num-seqs=32",
        f"--attention-backend={backend}",
    ]
    if tp_size:
        server_args += ["-tp", tp_size]
    server_args += os.getenv("VLLM_TEST_SERVER_ARGS", "").split()

    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()
        _compare_bs1_vs_bsn_single_process(
            prompts=prompts_all,
            sp_kwargs=sp_kwargs,
            client=client,
            model_name=TEST_MODEL,
        )
