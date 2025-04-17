# -*- coding: utf-8 -*-
"""
deepsc.models 包初始化
=====================

统一在这里 **导入所有带装饰器注册的子模块**，
避免外部调用 registry 时出现 “XX 未注册” 的问题。
"""

from importlib import import_module

# --------------------------------------------------
# 1) 导入 registry，使装饰器可用
from .registry import get_channel, register_channel  # noqa: F401

# 2) **强制**导入包含注册装饰器的实现文件
#    如需新增模块，直接在列表里加即可。
_submodules = [
    '.channel',     # 目前只有信道放这里
    # '.losses',    # 若将来有自定义 Loss，也写到这里
]

for _m in _submodules:
    import_module(__name__ + _m)
