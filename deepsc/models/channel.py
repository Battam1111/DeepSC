# deepsc/models/channel.py - 完善的信道模型
# -*- coding: utf-8 -*-
"""
Channel Encoder / Decoder 以及信道传播模型
=========================================

包含各种信道模型实现:
- AWGN: 加性高斯白噪声信道
- RAYLEIGH: 瑞利衰落信道
- RICIAN: 莱斯衰落信道
- ERASURE: 擦除信道

所有信道模型都保留梯度，支持端到端反向传播训练。
"""
import torch
import torch.nn as nn
import math
from .registry import register_channel
from typing import Optional, Tuple

class ChannelEncoder(nn.Module):
    """
    通道编码器：将语义特征映射到适合传输的符号
    
    参数:
        d_model: 输入特征维度
        latent_dim: 输出符号维度
        use_layernorm: 是否使用层归一化
    """
    def __init__(self, d_model: int = 512, latent_dim: int = 16, use_layernorm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: [batch, seq_len, d_model] 输入特征
            
        返回:
            [batch, seq_len, latent_dim] 编码后的符号
        """
        return self.net(x)

class ChannelDecoder(nn.Module):
    """
    通道解码器：将接收到的符号映射回语义特征空间
    
    参数:
        latent_dim: 输入符号维度
        d_model: 输出特征维度
        use_residual: 是否使用残差连接
    """
    def __init__(self, latent_dim: int = 16, d_model: int = 512, use_residual: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.use_residual = use_residual
        if use_residual:
            self.shortcut = nn.Linear(latent_dim, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            z: [batch, seq_len, latent_dim] 接收到的符号
            
        返回:
            [batch, seq_len, d_model] 解码后的特征
        """
        out = self.net(z)
        if self.use_residual:
            out = out + self.shortcut(z)
        return out

# --- 信道传播模型 ----------------------------------------------------------- #
class BaseChannel(nn.Module):
    """
    所有信道模型的基类
    
    所有派生类必须实现forward方法，接收发送符号和噪声方差，返回接收符号
    """
    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        信道传播
        
        参数:
            tx: [batch, seq_len, latent_dim] 发送符号
            n_var: 噪声方差
            
        返回:
            [batch, seq_len, latent_dim] 接收符号
        """
        raise NotImplementedError

@register_channel('AWGN')
class AWGNChannel(BaseChannel):
    """
    加性高斯白噪声信道
    
    模型: y = x + n，其中n ~ N(0, n_var)
    """
    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        AWGN信道传播
        
        参数:
            tx: [batch, seq_len, latent_dim] 发送符号
            n_var: 噪声方差
            
        返回:
            [batch, seq_len, latent_dim] 接收符号
        """
        # 生成与输入相同形状的高斯白噪声
        noise = torch.randn_like(tx) * math.sqrt(n_var)
        return tx + noise


@register_channel('RAYLEIGH')
class RayleighChannel(BaseChannel):
    """
    瑞利衰落信道 (已修复 clamp_abs 错误)

    模型: y = h*x + n，其中h ~ CN(0,1)，n ~ CN(0,n_var)

    参数:
        perfect_csi: 是否假设完美信道状态信息(CSI)
    """
    def __init__(self, perfect_csi: bool = True):
        super().__init__()
        self.perfect_csi = perfect_csi

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        瑞利信道传播

        参数:
            tx: [batch, seq_len, latent_dim] 发送符号 (假定为实数)
            n_var: 噪声方差

        返回:
            [batch, seq_len, latent_dim] 接收符号 (实数部分)
        """
        # 检查输入是否已经是复数，如果不是，则将其视为复数的实部
        if not torch.is_complex(tx):
            tx_complex = torch.complex(tx, torch.zeros_like(tx))
        else:
            tx_complex = tx # 如果输入已经是复数则直接使用

        # 生成复高斯随机变量作为信道系数 h ~ CN(0,1)
        # h 的形状应与 tx 兼容，通常是在 batch 和 seq_len 维度上独立衰落
        h_real = torch.randn_like(tx_complex.real) / math.sqrt(2)
        h_imag = torch.randn_like(tx_complex.imag) / math.sqrt(2)
        h = torch.complex(h_real, h_imag) # h: [batch, seq_len, latent_dim]

        # 添加复高斯噪声 n ~ CN(0, n_var)
        # 注意：n_var 是复噪声的总方差 E[|n|^2] = E[n_r^2] + E[n_i^2] = 2 * sigma^2
        # 因此，实部和虚部的方差各为 n_var / 2
        noise_std_per_dim = math.sqrt(n_var / 2)
        noise_real = torch.randn_like(tx_complex.real) * noise_std_per_dim
        noise_imag = torch.randn_like(tx_complex.imag) * noise_std_per_dim
        noise = torch.complex(noise_real, noise_imag)

        # 信道传播: y = h*x + n
        rx_complex = h * tx_complex + noise

        # 信道均衡（假设完美CSI） y_eq = y / h = y * conj(h) / |h|^2
        if self.perfect_csi:
            # 计算 h 的幅度平方 |h|^2
            h_abs_sq = h.abs().pow(2)
            # 使用 clamp_min 保证分母不为零 (Tensor 有 clamp_min 方法)
            h_abs_sq_clamped = h_abs_sq.clamp_min(1e-9) # 使用更小的 epsilon 防止数值问题
            # 执行均衡: y * conj(h) / |h|^2_clamped
            rx_equalized = rx_complex * h.conj() / h_abs_sq_clamped
        else:
            # 如果没有完美 CSI，则直接返回接收信号（未均衡）
            rx_equalized = rx_complex

        # 返回接收信号的实部 (或均衡后的实部)
        # 如果下游模块期望实数输入，则取实部
        # 如果下游可以处理复数，可以考虑返回 rx_equalized
        return rx_equalized.real

@register_channel('RICIAN')
class RicianChannel(BaseChannel):
    """
    莱斯衰落信道 (已修复 clamp_abs 错误)

    模型: y = h*x + n，其中h服从莱斯分布，n ~ CN(0,n_var)

    参数:
        K: 莱斯K因子，表示视距分量与散射分量的功率比
        perfect_csi: 是否假设完美信道状态信息
    """
    def __init__(self, K: float = 1.0, perfect_csi: bool = True):
        super().__init__()
        # 验证 K 因子是否有效
        if K < 0:
            raise ValueError("莱斯 K 因子必须是非负数。")
        self.K = K
        self.perfect_csi = perfect_csi

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        莱斯信道传播

        参数:
            tx: [batch, seq_len, latent_dim] 发送符号 (假定为实数)
            n_var: 噪声方差

        返回:
            [batch, seq_len, latent_dim] 接收符号 (实数部分)
        """
        # 检查输入是否已经是复数
        if not torch.is_complex(tx):
            tx_complex = torch.complex(tx, torch.zeros_like(tx))
        else:
            tx_complex = tx

        # 计算莱斯分布参数
        # 总功率归一化为 1, 即 E[|h|^2] = los_power + nlos_power = 1
        los_power = self.K / (self.K + 1)  # 视距(LoS)分量功率
        nlos_power = 1 / (self.K + 1) # 非视距(NLoS)散射分量功率

        # 生成非视距分量 h_nlos ~ CN(0, nlos_power)
        h_nlos_real = torch.randn_like(tx_complex.real) * math.sqrt(nlos_power / 2)
        h_nlos_imag = torch.randn_like(tx_complex.imag) * math.sqrt(nlos_power / 2)

        # 生成视距分量 h_los (确定性部分，幅度为 sqrt(los_power)，相位通常设为0)
        # 这里假设 LoS 分量的均值为 sqrt(los_power) + 0j
        h_los_real = torch.full_like(tx_complex.real, math.sqrt(los_power))
        h_los_imag = torch.zeros_like(tx_complex.imag)

        # 合并视距和非视距分量 h = h_los + h_nlos
        h_real = h_los_real + h_nlos_real
        h_imag = h_los_imag + h_nlos_imag
        h = torch.complex(h_real, h_imag)

        # 添加复高斯噪声 n ~ CN(0, n_var)
        noise_std_per_dim = math.sqrt(n_var / 2)
        noise_real = torch.randn_like(tx_complex.real) * noise_std_per_dim
        noise_imag = torch.randn_like(tx_complex.imag) * noise_std_per_dim
        noise = torch.complex(noise_real, noise_imag)

        # 信道传播: y = h*x + n
        rx_complex = h * tx_complex + noise

        # 信道均衡（假设完美CSI） y_eq = y * conj(h) / |h|^2
        if self.perfect_csi:
            h_abs_sq = h.abs().pow(2)
            h_abs_sq_clamped = h_abs_sq.clamp_min(1e-9) # 使用 clamp_min
            rx_equalized = rx_complex * h.conj() / h_abs_sq_clamped
        else:
            rx_equalized = rx_complex

        # 返回实部
        return rx_equalized.real

@register_channel('ERASURE')
class ErasureChannel(BaseChannel):
    """
    擦除信道
    
    以概率p将某些符号完全擦除（置零）
    
    参数:
        p: 擦除概率，如果为None则根据n_var自适应设置
        snr_scaling: n_var转换为擦除概率的缩放因子
    """
    def __init__(self, p: Optional[float] = None, snr_scaling: float = 10.0) -> None:
        super().__init__()
        self.p = p
        self.snr_scaling = snr_scaling

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        擦除信道传播
        
        参数:
            tx: [batch, seq_len, latent_dim] 发送符号
            n_var: 噪声方差（若self.p=None则用于计算擦除概率）
            
        返回:
            [batch, seq_len, latent_dim] 接收符号
        """
        # 确定擦除概率
        if self.p is not None:
            p = self.p
        else:
            # 根据SNR动态调整擦除概率，n_var越大，擦除概率越高
            p = min(0.9, n_var * self.snr_scaling)
        
        # 生成擦除掩码：以概率(1-p)保留，概率p擦除
        mask = (torch.rand_like(tx[:, :, :1]) > p).float()
        
        # 添加少量高斯噪声
        noise = torch.randn_like(tx) * math.sqrt(n_var)
        
        # 应用擦除和添加噪声
        return tx * mask + noise