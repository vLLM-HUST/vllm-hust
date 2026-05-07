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


import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from msmodelslim.model.qwen3_vl.model_adapter import Qwen3VLModelAdapter
from msmodelslim.utils.exception import UnsupportedError


class TestQwen3VLModelAdapterGetRotateMapTieWordEmbeddings(unittest.TestCase):
    """get_rotate_map raises UnsupportedError when tie_word_embeddings=True."""

    def test_tie_word_embeddings_on_config_raises(self):
        with patch("msmodelslim.model.qwen3_vl.model_adapter.VLMBaseModelAdapter.__init__", return_value=None):
            adapter = Qwen3VLModelAdapter.__new__(Qwen3VLModelAdapter)
            adapter.config = MagicMock()
            adapter.config.torch_dtype = None
            adapter.config.tie_word_embeddings = True
            adapter.config.text_config = None
        with self.assertRaises(UnsupportedError) as ctx:
            adapter.get_rotate_map(block_size=64)
        self.assertIn("tie_word_embeddings", str(ctx.exception))
        self.assertIn("QuaRot", str(ctx.exception))
