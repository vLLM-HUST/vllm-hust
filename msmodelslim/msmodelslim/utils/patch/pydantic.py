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

import sys

from pydantic import BaseModel as PydanticBaseModel, ValidationError

from msmodelslim.utils.exception import SchemaValidateError


def patch_pydantic():
    """
    通过patch的方式替换pydantic的BaseModel，将ValidationError转换为SchemaValidateError
    """

    # 保存原始的BaseModel
    original_base_model = PydanticBaseModel

    class PatchedBaseModel(original_base_model):
        """
        自定义BaseModel，将pydantic的ValidationError转换为项目的SchemaValidateError
        """

        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

        @classmethod
        def model_validate(cls, *args, **kwargs):
            try:
                return super().model_validate(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

        @classmethod
        def model_validate_json(cls, *args, **kwargs):
            try:
                return super().model_validate_json(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

    # 全局替换pydantic的BaseModel
    import pydantic
    pydantic.BaseModel = PatchedBaseModel

    # 替换sys.modules中的BaseModel，确保所有import都能获取到patched版本
    if 'pydantic' in sys.modules:
        sys.modules['pydantic'].BaseModel = PatchedBaseModel

    # 为了确保所有可能的导入路径都被覆盖
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('pydantic'):
            module = sys.modules[module_name]
            if hasattr(module, 'BaseModel'):
                module.BaseModel = PatchedBaseModel
