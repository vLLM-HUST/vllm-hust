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
"""
Pydantic 特定的验证函数包装器。

这些函数专门为 Pydantic AfterValidator 设计，能够自动从 ValidationInfo 获取字段名。
基于 value.py 中的通用验证函数进行包装。
"""

from typing import Any, List

from pydantic import ValidationInfo

from msmodelslim.utils.security import validate_safe_host, validate_safe_endpoint
from msmodelslim.utils.validation.value import (
    is_port as _is_port,
    greater_than_zero as _greater_than_zero,
    int_greater_than_zero as _int_greater_than_zero,
    at_least_one_element as _at_least_one_element,
    validate_normalized_value as _validate_normalized_value,
    is_boolean as _is_boolean,
    is_string_list as _is_string_list,
    validate_str_length as _validate_str_length,
    non_empty_string as _non_empty_string,
)


def at_least_one_element(v: Any, info: ValidationInfo) -> Any:
    """
    给 Pydantic AfterValidator 使用的至少一个元素校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    """
    field_name = getattr(info, 'field_name', None) or "value"
    return _at_least_one_element(v, param_name=field_name)


def is_safe_host(v: str, info: ValidationInfo) -> str:
    """
    给 Pydantic AfterValidator 使用的 host 安全校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。

    Args:
        v: 要验证的主机地址
        info: Pydantic ValidationInfo 对象，用于获取字段名

    Returns:
        验证后的主机地址

    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated

        class Config(BaseModel):
            host: Annotated[str, AfterValidator(is_safe_host)] = "localhost"
        ```
    """
    # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
    field_name = getattr(info, 'field_name', None) or "host"
    return validate_safe_host(v, field_name=field_name)


def is_safe_endpoint(v: str, info: ValidationInfo) -> str:
    """
    给 Pydantic AfterValidator 使用的 endpoint 安全校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。

    Args:
        v: 要验证的端点路径
        info: Pydantic ValidationInfo 对象，用于获取字段名

    Returns:
        验证后的端点路径

    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated

        class Config(BaseModel):
            endpoint: Annotated[str, AfterValidator(is_safe_endpoint)] = "/api"
        ```
    """
    # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
    field_name = getattr(info, 'field_name', None) or "endpoint"
    return validate_safe_endpoint(v, field_name=field_name)


def is_port(v: Any, info: ValidationInfo) -> int:
    """
    给 Pydantic AfterValidator 使用的端口号校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。

    Args:
        v: 要验证的端口号
        info: Pydantic ValidationInfo 对象，用于获取字段名

    Returns:
        验证后的端口号

    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated

        class Config(BaseModel):
            port: Annotated[int, AfterValidator(is_port)] = 8080
        ```
    """
    # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
    field_name = getattr(info, 'field_name', None) or "port"
    return _is_port(v, param_name=field_name)


def greater_than_zero(v: Any, info: ValidationInfo) -> Any:
    """
    给 Pydantic AfterValidator 使用的大于零校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    
    Args:
        v: 要验证的数值
        info: Pydantic ValidationInfo 对象，用于获取字段名
        
    Returns:
        验证后的数值
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            timeout: Annotated[int, AfterValidator(greater_than_zero)] = 60
        ```
    """
    # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
    field_name = getattr(info, 'field_name', None) or "value"
    return _greater_than_zero(v, param_name=field_name)


def int_greater_than_zero(v: Any, info: ValidationInfo) -> Any:
    """
    给 Pydantic AfterValidator 使用的大于零校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。

    Args:
        v: 要验证的数值
        info: Pydantic ValidationInfo 对象，用于获取字段名

    Returns:
        验证后的数值
    """
    # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
    field_name = getattr(info, 'field_name', None) or "value"
    return _int_greater_than_zero(v, param_name=field_name)


def validate_normalized_value(v: Any, info: ValidationInfo) -> float:
    """
    给 Pydantic AfterValidator 使用的归一化值校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    验证值必须是 float 或 None 类型，且在 (0, 1) 范围内。
    
    Args:
        v: 要验证的归一化值
        info: Pydantic ValidationInfo 对象，用于获取字段名
        
    Returns:
        验证后的归一化值
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            threshold: Annotated[float, AfterValidator(validate_normalized_value)] = 0.5
        ```
    """
    field_name = getattr(info, 'field_name', None) or "value"
    return _validate_normalized_value(v, param_name=field_name)


def is_boolean(v: Any, info: ValidationInfo) -> bool:
    """
    给 Pydantic AfterValidator 使用的布尔类型校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    
    Args:
        v: 要验证的值
        info: Pydantic ValidationInfo 对象，用于获取字段名
        
    Returns:
        验证后的布尔值
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            enabled: Annotated[bool, AfterValidator(is_boolean)] = True
        ```
    """
    field_name = getattr(info, 'field_name', None) or "value"
    return _is_boolean(v, param_name=field_name)


def is_string_list(v: Any, info: ValidationInfo) -> List[str]:
    """
    给 Pydantic AfterValidator 使用的字符串列表校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    
    Args:
        v: 要验证的列表
        info: Pydantic ValidationInfo 对象，用于获取字段名
        
    Returns:
        验证后的字符串列表
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            tags: Annotated[list, AfterValidator(is_string_list)] = ["tag1", "tag2"]
        ```
    """
    field_name = getattr(info, 'field_name', None) or "value"
    return _is_string_list(v, param_name=field_name)


def validate_str_length(max_len: int = 300):
    """
    创建一个字符串长度验证器，用于 Pydantic AfterValidator。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    
    Args:
        max_len: 允许的最大长度，默认为 100
        
    Returns:
        一个可用于 Pydantic AfterValidator 的验证函数
        
    Example:
        ```python
        from pydantic import BaseModel, AfterValidator
        from typing import Annotated
        from msmodelslim.utils.validation.pydantic import validate_str_length
        
        class Config(BaseModel):
            # 使用默认最大长度 100
            description: Annotated[str, AfterValidator(validate_str_length())] = "default"
            
            # 使用自定义最大长度
            short_text: Annotated[str, AfterValidator(validate_str_length(max_len=100))] = ""
        ```
    """

    def validator(v: str, info: ValidationInfo) -> str:
        field_name = getattr(info, 'field_name', None) or "string"
        _validate_str_length(v, str_name=field_name, max_len=max_len)
        return v

    return validator


def non_empty_string(v: str, info: ValidationInfo) -> str:
    """
    给 Pydantic AfterValidator 使用的非空字符串校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    验证字符串不能为 None 或空（去除空白后）。
    
    Args:
        v: 要验证的字符串
        info: Pydantic ValidationInfo 对象，用于获取字段名
        
    Returns:
        验证后的非空字符串
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            name: Annotated[str, AfterValidator(non_empty_string)] = "default"
        ```
    """
    field_name = getattr(info, 'field_name', None) or "value"
    return _non_empty_string(v, field_name=field_name)
