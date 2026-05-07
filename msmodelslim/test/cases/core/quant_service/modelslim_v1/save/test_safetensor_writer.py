#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

"""Unit tests for BufferedSafetensorsWriter._dedupe_shared_storage (309aeec)."""

import tempfile
from unittest.mock import MagicMock

import pytest
import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import BufferedSafetensorsWriter


@pytest.fixture
def writer(temp_dir):
    """BufferedSafetensorsWriter with mock logger and temp directory."""
    logger = MagicMock()
    # BufferedSafetensorsWriter sets save_directory via setter which calls get_write_directory
    w = BufferedSafetensorsWriter(
        logger=logger,
        max_gb_size=4,
        save_directory=temp_dir,
        save_prefix="quant_model_weights",
    )
    return w


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    return d


class TestDedupeSharedStorage:
    """Tests for _dedupe_shared_storage."""

    def test_no_shared_storage_unchanged(self, writer):
        """Multiple keys with different tensors -> output same keys and tensors."""
        a = torch.randn(2, 3)
        b = torch.randn(4, 5)
        keys_dict = {"model.layer0.weight": a, "model.layer1.weight": b}
        out = writer._dedupe_shared_storage(keys_dict)
        assert set(out.keys()) == set(keys_dict.keys())
        assert out["model.layer0.weight"] is a
        assert out["model.layer1.weight"] is b

    def test_shared_storage_embed_tokens_kept_lm_head_cloned(self, writer):
        """When embed_tokens.weight and lm_head.weight share storage, keep embed_tokens, clone lm_head."""
        t = torch.randn(4, 8)
        keys_dict = {
            "model.embed_tokens.weight": t,
            "model.lm_head.weight": t,
        }
        out = writer._dedupe_shared_storage(keys_dict)
        assert "model.embed_tokens.weight" in out
        assert "model.lm_head.weight" in out
        # embed_tokens is first in sort order (0), so it gets the original
        assert out["model.embed_tokens.weight"] is t
        # lm_head shares storage, so it gets a clone
        assert out["model.lm_head.weight"] is not t
        assert out["model.lm_head.weight"].data_ptr() != t.data_ptr()
        torch.testing.assert_close(out["model.lm_head.weight"], t)
