# -*- coding: utf-8 -*-
"""
功率归一化
========
提供符合论文要求的功率归一化函数，确保传输符号的平均功率为指定值。
"""
import torch
import math

def power_normalize(x: torch.Tensor, target_power: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    将特征整体发射功率归一化到 target_power
    
    根据论文公式：x = √(target_power) * x / ||x||
    其中 ||x|| 是均方根功率。
    
    参数:
        x: 输入张量，形状任意
        target_power: 目标平均功率，默认为1.0
        dim: 沿哪个维度计算功率，默认为最后一个维度
        
    返回:
        功率归一化后的张量，形状与输入相同
    """
    # 计算均方根功率: √(1/N * Σ|x_i|²)
    rms = torch.sqrt(torch.mean(x.pow(2), dim=dim, keepdim=True).clamp_min(1e-9))
    
    # 缩放因子: √(target_power) / rms
    scale = torch.sqrt(torch.tensor(target_power, device=x.device)) / rms
    
    # 缩放输入
    return x * scale