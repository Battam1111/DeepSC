# -*- coding: utf-8 -*-
"""
注册表工具：用于动态添加新的信道模型、损失函数、评测指标等。
调用方式：
    from deepsc.models import registry as reg
    @reg.register_channel('AWGN')
    class AWGNChannel(BaseChannel): ...
"""
from typing import Dict, Type

_CHANNELS: Dict[str, Type] = {}

def register_channel(name: str):
    """装饰器：注册信道类到全局表"""
    def decorator(cls):
        _CHANNELS[name.upper()] = cls
        return cls
    return decorator

def get_channel(name: str):
    if name.upper() not in _CHANNELS:
        raise ValueError(f'信道 {name} 未注册')
    return _CHANNELS[name.upper()]
