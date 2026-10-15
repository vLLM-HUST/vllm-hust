# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable proof imports stay local to exact files and import families."""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.pre_commit.check_forbidden_imports import check_file


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("native_semantic_collector.py", "import base64\nimport re\n", 0),
        ("native_tensor_artifact_transport.py", "import base64\nimport re\n", 0),
        ("native_tensor_witness.py", "import base64\n", 0),
        ("native_tensor_runtime_contract.py", "import re\n", 0),
        ("neighbor.py", "import base64\n", 1),
        ("neighbor.py", "import re\n", 1),
        ("native_tensor_witness.py", "import re\n", 1),
        ("native_tensor_runtime_contract.py", "import base64\n", 1),
        ("native_semantic_collector.py", "import pickle\n", 1),
    ],
)
def test_portable_exception_preserves_other_files_and_rules(
    tmp_path, monkeypatch, name, source, expected
):
    monkeypatch.chdir(tmp_path)
    path = Path("vllm/plugins") / name
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert check_file(str(path)) == expected


def test_canonical_protocol_modules_import_without_site_packages():
    root = Path(__file__).resolve().parents[2]
    # Namespace shells load the actual protocol bytes without starting vLLM.
    # The isolated interpreter cannot import C-extension fast-codec packages.
    script = """
import importlib, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
for name, path in [('vllm', root / 'vllm'),
                   ('vllm.plugins', root / 'vllm/plugins')]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
for name in ['native_semantic_collector', 'native_tensor_artifact_transport',
             'native_tensor_runtime_contract', 'native_tensor_witness',
             'native_attention_operator_observer', 'native_phase_drain']:
    module = importlib.import_module('vllm.plugins.' + name)
    assert pathlib.Path(module.__file__).parent == root / 'vllm/plugins'
assert not {'pybase64', 'regex', 'torch', 'torch_npu'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", script, str(root)], check=True)
