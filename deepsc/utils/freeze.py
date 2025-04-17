# -*- coding: utf-8 -*-
"""
参数冻结 / 解冻辅助函数
"""
import torch.nn as nn
from typing import Iterable

def freeze(module: nn.Module):
    """
    将模块参数冻结（requires_grad=False）
    """
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module: nn.Module):
    """
    将模块参数解冻
    """
    for p in module.parameters():
        p.requires_grad_(True)

def count_trainable_params(model: nn.Module) -> int:
    """
    统计可训练参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
