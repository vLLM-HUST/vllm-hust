# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.version import Version

from vllm import version

ROOT = Path(__file__).resolve().parents[1]
HUST_VERSION_METADATA = ROOT / "upstream_version.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def test_hust_version_metadata_is_well_formed():
    metadata = json.loads(HUST_VERSION_METADATA.read_text(encoding="utf-8"))

    assert set(metadata) == {
        "release_version",
        "upstream_version",
        "upstream_commit",
    }
    release = Version(metadata["release_version"])
    upstream = Version(metadata["upstream_version"])
    assert release.release[:2] == upstream.release[:2]
    assert len(metadata["upstream_commit"]) == 40
    int(metadata["upstream_commit"], 16)


def test_hust_upstream_anchor_tracks_the_latest_main_sync():
    if not (ROOT / ".git").exists() and not (ROOT / ".git").is_file():
        return

    merge = next(
        commit
        for commit in _git(
            "log",
            "--first-parent",
            "--merges",
            "--format=%H%x09%s",
        ).splitlines()
        if "Merge upstream vllm-project/vllm main" in commit
    )
    merge_commit = merge.split("\t", 1)[0]
    upstream_parent = _git("rev-parse", f"{merge_commit}^2")
    metadata = json.loads(HUST_VERSION_METADATA.read_text(encoding="utf-8"))

    assert metadata["upstream_commit"] == upstream_parent


def test_version_is_defined():
    assert version.__version__ is not None


def test_version_tuple():
    # (major, minor, patch) plus optional pre-release tag, dev-distance, and
    # git-hash components - setuptools_scm can emit any combination of them,
    # e.g. 6 when a dev build sits on top of an rc-tagged commit.
    assert len(version.__version_tuple__) >= 3


@pytest.mark.parametrize(
    "version_tuple, version_str, expected",
    [
        ((0, 0, "dev"), "0.0", True),
        ((0, 0, "dev"), "foobar", True),
        ((0, 7, 4), "0.6", True),
        ((0, 7, 4), "0.5", False),
        ((0, 7, 4), "0.7", False),
        ((1, 2, 3), "1.1", True),
        ((1, 2, 3), "1.0", False),
        ((1, 2, 3), "1.2", False),
        # This won't work as expected
        ((1, 0, 0), "1.-1", True),
        ((1, 0, 0), "0.9", False),
        ((1, 0, 0), "0.17", False),
    ],
)
def test_prev_minor_version_was(version_tuple, version_str, expected):
    with patch("vllm.version.__version_tuple__", version_tuple):
        assert version._prev_minor_version_was(version_str) == expected
