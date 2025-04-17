# -*- coding: utf-8 -*-
"""
Channel Encoder / Decoder 以及信道传播模型抽象
"""
import torch
import torch.nn as nn
import math
from .registry import register_channel
from typing import Optional

class ChannelEncoder(nn.Module):
    """通道前向映射：d_model → n_symbols×2 (I/Q)"""
    def __init__(self, d_model: int = 128, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)   # ★ 新增
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return self.net(x)

class ChannelDecoder(nn.Module):
    """通道逆映射：latent_dim → d_model"""
    def __init__(self, latent_dim: int = 16, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# --- 信道传播模型 ----------------------------------------------------------- #
class BaseChannel(nn.Module):
    """所有信道模型需继承该基类，实现 forward(tx, n_var)"""
    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        raise NotImplementedError

@register_channel('AWGN')
class AWGNChannel(BaseChannel):
    """加性高斯白噪声信道。保留梯度以便端到端反传。"""
    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        noise = torch.randn_like(tx) * n_var
        return tx + noise


@register_channel('RAYLEIGH')
class RayleighChannel(BaseChannel):
    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        # 复系数 h ~ CN(0,1), 与 I/Q 向量逐元素相乘
        h_real = torch.randn_like(tx[..., 0:1]) / math.sqrt(2)
        h_imag = torch.randn_like(tx[..., 0:1]) / math.sqrt(2)
        h = torch.cat([h_real, h_imag], dim=-1)          # same shape as tx
        rx = tx * h + torch.randn_like(tx) * n_var
        # 理想信道估计：除以 h
        return rx / h.clamp_min(1e-3)

@register_channel('RICIAN')
class RicianChannel(BaseChannel):
    """
    里克信道：K 因子可配置；遵循论文设置 K=1
    """
    def __init__(self, K: float = 1.0):
        super().__init__()
        self.K = K

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        b, l, _ = tx.shape
        mean = math.sqrt(self.K / (self.K + 1))
        std = math.sqrt(1 / (self.K + 1))
        h_real = torch.normal(mean, std, size=(b, 1, 1), device=tx.device)
        h_imag = torch.normal(mean, std, size=(b, 1, 1), device=tx.device)
        h = torch.cat([h_real, -h_imag, h_imag, h_real], dim=-1).view(b, 1, 2, 2)
        tx_mat = tx.view(b, l, -1, 2)
        rx = torch.matmul(tx_mat, h).view_as(tx)
        rx = rx + torch.randn_like(rx) * n_var
        # 理想信道估计
        h_inv = torch.inverse(h.squeeze(1))
        rx_mat = rx.view(b, l, -1, 2)
        rx_hat = torch.matmul(rx_mat, h_inv).view_as(tx)
        return rx_hat

@register_channel('ERASURE')
class ErasureChannel(BaseChannel):
    """
    擦除信道：以概率 p 将符号置 0；p 根据 n_var 自适应
    """
    def __init__(self, p: Optional[float] = None) -> None:
        super().__init__()
        self.p = p

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        p = self.p if self.p is not None else min(0.9, n_var * 10)
        mask = (torch.rand_like(tx[..., :1]) > p).float()
        noise = torch.randn_like(tx) * n_var
        return tx * mask + noise
