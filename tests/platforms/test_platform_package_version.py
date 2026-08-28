# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.metadata import PackageNotFoundError

import pytest

from vllm.platforms import vllm_version_matches_substr


def test_vllm_hust_distribution_name_is_supported(monkeypatch: pytest.MonkeyPatch):
    def fake_version(package_name: str) -> str:
        if package_name == "vllm-hust":
            return "0.23.1+empty"
        raise PackageNotFoundError(package_name)

    monkeypatch.setattr("importlib.metadata.version", fake_version)

    assert vllm_version_matches_substr("empty")
    assert not vllm_version_matches_substr("cpu")


def test_missing_vllm_distributions_raise(monkeypatch: pytest.MonkeyPatch):
    def missing_version(package_name: str) -> str:
        raise PackageNotFoundError(package_name)

    monkeypatch.setattr("importlib.metadata.version", missing_version)

    with pytest.raises(PackageNotFoundError, match="vllm, vllm-hust"):
        vllm_version_matches_substr("cpu")
