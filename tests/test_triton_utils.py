# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest import mock

from vllm.triton_utils import importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def test_active_out_of_tree_triton_driver_is_supported(monkeypatch):
    active_driver = SimpleNamespace(is_active=lambda: True)
    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(driver=SimpleNamespace(active=active_driver))
    )
    monkeypatch.setattr(importing.current_platform, "is_out_of_tree", lambda: True)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    assert importing._has_active_out_of_tree_triton_driver()


def test_in_tree_platform_does_not_use_out_of_tree_driver(monkeypatch):
    monkeypatch.setattr(importing.current_platform, "is_out_of_tree", lambda: False)

    assert not importing._has_active_out_of_tree_triton_driver()


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"
