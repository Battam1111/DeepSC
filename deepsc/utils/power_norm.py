# -*- coding: utf-8 -*-
"""
符号功率归一化（论文中的 PowerNormalize）
"""

import torch

def power_normalize(x: torch.Tensor,
                    target_power: float = 1.0) -> torch.Tensor:
    """
    将特征整体发射功率归一化到 target_power
    论文式：P_out = P_in / RMS(x)^2 * target
    """
    # RMS power
    rms = x.pow(2).mean().sqrt().clamp_min(1e-9)
    scale = (target_power ** 0.5) / rms
    return x * scale
