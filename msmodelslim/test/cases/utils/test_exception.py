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
msmodelslim.utils.exception 模块的单元测试
"""

import unittest

from msmodelslim.utils.exception import ModelslimError


class TestModelslimError(unittest.TestCase):
    """测试 ModelslimError 基类"""

    def test_str_when_error_with_message_then_return_formatted_string(self):
        """测试__str__方法：当错误包含消息时，应返回格式化的字符串"""
        error = ModelslimError("Something unexpected happened")

        expected_str = "Code: 0, Message: Something unexpected happened"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_message_and_action_then_return_formatted_string_with_tip(self):
        """测试__str__方法：当错误包含消息和解决推荐时，应返回包含TIP的格式化字符串"""
        error = ModelslimError("Python version not compatible",
                               action="Please upgrade to Python 3.8 or higher")

        expected_str = "Code: 0, Message: Python version not compatible\nTIP: Please upgrade to Python 3.8 or higher"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_without_message_then_use_default_message(self):
        """测试__str__方法：当错误不包含消息时，应使用默认消息"""
        error = ModelslimError()

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_empty_message_then_use_default_message(self):
        """测试__str__方法：当错误消息为空字符串时，应使用默认消息"""
        error = ModelslimError("")

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_none_message_then_use_default_message(self):
        """测试__str__方法：当错误消息为None时，应使用默认消息"""
        error = ModelslimError(None)

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_repr_when_error_with_message_then_return_formatted_repr(self):
        """测试__repr__方法：当错误包含消息时，应返回格式化的repr字符串"""
        error = ModelslimError("Something unexpected happened")

        expected_repr = "[ModelslimError] Code: 0, Message: Something unexpected happened"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_message_and_action_then_return_formatted_repr_with_tip(self):
        """测试__repr__方法：当错误包含消息和解决推荐时，应返回包含TIP的格式化repr字符串"""
        error = ModelslimError("Python version not compatible",
                               action="Please upgrade to Python 3.8 or higher")

        expected_repr = "[ModelslimError] Code: 0, Message: Python version not compatible, TIP: Please upgrade to Python 3.8 or higher"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_without_message_then_use_default_message(self):
        """测试__repr__方法：当错误不包含消息时，应使用默认消息"""
        error = ModelslimError()

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_empty_message_then_use_default_message(self):
        """测试__repr__方法：当错误消息为空字符串时，应使用默认消息"""
        error = ModelslimError("")

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_none_message_then_use_default_message(self):
        """测试__repr__方法：当错误消息为None时，应使用默认消息"""
        error = ModelslimError(None)

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_create_exception_when_valid_parameters_then_create_custom_error_class(self):
        """测试create_exception类方法：使用有效参数时应成功创建自定义错误类"""
        CustomError = ModelslimError.create_exception("CustomError", 999, "Custom error message")

        # 测试创建的错误类
        error = CustomError("Test message")
        self.assertEqual(error.code, 999)
        self.assertEqual(error.default_message, "Custom error message")
        self.assertEqual(str(error), "Code: 999, Message: Test message")
        self.assertEqual(repr(error), "[CustomError] Code: 999, Message: Test message")


if __name__ == "__main__":
    unittest.main()
