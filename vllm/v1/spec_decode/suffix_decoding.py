# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig, observer=None):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_cached_requests = config.suffix_decoding_max_cached_requests
        self.observer = observer

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=self.max_cached_requests,
        )

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
        config_epoch: int | None = None,
    ) -> list[list[int]]:
        assert num_speculative_tokens == self.num_speculative_tokens
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            propose_started_ns = time.perf_counter_ns()
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            if self.observer is not None and self.observer.enabled:
                self.observer.record_pre_verification_decision(
                    request_id=req_id,
                    request_slot=i,
                    context_token_count=int(num_tokens),
                    prompt_token_count=int(input_batch.num_prompt_tokens[index]),
                    sampled_token_count=len(sampled_ids),
                    pattern_token_count=len(pattern),
                    match_length=int(draft.match_len),
                    draft_token_count=len(draft.token_ids),
                    draft_score=float(draft.score),
                    draft_probabilities=draft.probs,
                    proposer_elapsed_ns=time.perf_counter_ns() - propose_started_ns,
                    active_request_count=len(input_batch.req_id_to_index),
                    config_epoch=config_epoch,
                )

            draft_token_ids.append(draft.token_ids)

        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def reset_cache(self) -> dict[str, int]:
        """Clear request history for a controlled, scope-matched TRAIN pair."""
        if self.observer is None or not self.observer.enabled:
            raise RuntimeError("Suffix cache reset is available only with the TRAIN observer enabled")
        # A completed request can remain in the local proposer tree until the
        # next proposal step cleans it up. EngineCore calls this only after its
        # scheduler proves the service has no unfinished requests.
        retained_active_requests = len(self.suffix_cache.active_requests)
        cached_requests = len(self.suffix_cache.cached_requests)
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=self.max_tree_depth,
            max_cached_requests=self.max_cached_requests,
        )
        return {
            "cleared_cached_requests": cached_requests,
            "cleared_retained_active_requests": retained_active_requests,
            "active_requests": 0,
        }

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
