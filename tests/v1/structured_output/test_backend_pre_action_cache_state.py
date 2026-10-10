# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
)
from vllm.v1.structured_output.backend_outlines import OutlinesBackend


def test_outlines_pre_action_cache_state_is_read_only_and_does_not_claim_hit():
    backend = object.__new__(OutlinesBackend)
    cache = {}
    backend.cache = cache

    state = backend.compiled_grammar_cache_state()

    assert state["cache_ready"] is True
    assert state["cache_ready_source_field"] == (
        "vllm.v1.structured_output.backend_outlines.OutlinesBackend.cache"
    )
    assert state["cache_size_bytes"] is None
    assert state["cache_limit_bytes"] is None
    assert cache == {}
    assert "cache_hit" not in state


def test_outlines_missing_cache_object_stays_unknown():
    backend = object.__new__(OutlinesBackend)
    backend.cache = None

    state = backend.compiled_grammar_cache_state()

    assert state["cache_ready"] is None
    assert state["cache_size_bytes"] is None
    assert state["cache_limit_bytes"] is None


def test_guidance_and_lmfe_report_no_backend_owned_grammar_cache():
    backend_sources = (
        (
            GuidanceBackend,
            "vllm.v1.structured_output.backend_guidance."
            "GuidanceBackend.compile_grammar",
        ),
        (
            LMFormatEnforcerBackend,
            "vllm.v1.structured_output.backend_lm_format_enforcer."
            "LMFormatEnforcerBackend.compile_grammar",
        ),
    )
    for backend_type, source_field in backend_sources:
        backend = object.__new__(backend_type)

        state = backend.compiled_grammar_cache_state()

        assert state["cache_ready"] is False
        assert state["cache_size_bytes"] is None
        assert state["cache_limit_bytes"] is None
        assert state["cache_ready_source_field"] == source_field
