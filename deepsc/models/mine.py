# -*- coding: utf-8 -*-
"""
互信息估计器 MINE（可插拔）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsc.models.registry import register_channel  # 仅示例：此处不注册

class MINE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        :param z1: Tx 符号 [*, latent_dim]
        :param z2: Rx 符号 [*, latent_dim]
        :return: MI lower‑bound
        """
        # 构建 joint & marginal
        joint = torch.cat([z1, z2], dim=-1)
        idx = torch.randperm(z2.size(0), device=z2.device)
        marginal = torch.cat([z1, z2[idx]], dim=-1)

        t = self.net(joint)
        et = torch.exp(self.net(marginal))
        mi_lb = t.mean() - torch.log(et.mean())
        return mi_lb
