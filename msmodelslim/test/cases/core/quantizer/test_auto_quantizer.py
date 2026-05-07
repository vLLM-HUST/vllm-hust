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

import pytest
from pydantic import ValidationError

from msmodelslim.core.quantizer.base import AutoActQuantizer, AutoWeightQuantizer, QConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestAutoActQuantizerFactory:
    """测试AutoActQuantizer工厂构造方法"""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        # 测试None配置
        with pytest.raises(ValidationError):
            AutoActQuantizer.from_config(None)

        # 测试非QConfig对象
        with pytest.raises(SchemaValidateError):
            AutoActQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.ir.qal.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoActQuantizer)
        class MyActQuantizer(AutoActQuantizer):

            def __init__(self, config: QConfig):
                super().__init__()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoActQuantizer.from_config(config)
        assert isinstance(quantizer, MyActQuantizer)


class TestAutoWeightQuantizerFactory:
    """测试AutoWeightQuantizer工厂构造方法"""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        # 测试None配置
        with pytest.raises(ValidationError):
            AutoWeightQuantizer.from_config(None)

        # 测试非QConfig对象
        with pytest.raises(SchemaValidateError):
            AutoWeightQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.ir.qal.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoWeightQuantizer)
        class MyWeightQuantizer(AutoWeightQuantizer):

            def __init__(self, config: QConfig):
                super().__init__()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoWeightQuantizer.from_config(config)
        assert isinstance(quantizer, MyWeightQuantizer)
