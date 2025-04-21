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
    瑞利衰落信道
    
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
            tx: [batch, seq_len, latent_dim] 发送符号
            n_var: 噪声方差
            
        返回:
            [batch, seq_len, latent_dim] 接收符号
        """
        # 生成复高斯随机变量作为信道系数
        h_real = torch.randn_like(tx) / math.sqrt(2)
        h_imag = torch.randn_like(tx) / math.sqrt(2)
        h = torch.complex(h_real, h_imag)
        
        # 添加高斯噪声
        noise_real = torch.randn_like(tx) * math.sqrt(n_var/2)
        noise_imag = torch.randn_like(tx) * math.sqrt(n_var/2)
        noise = torch.complex(noise_real, noise_imag)
        
        # 信道传播
        tx_complex = torch.complex(tx, torch.zeros_like(tx))
        rx_complex = h * tx_complex + noise
        
        # 信道均衡（假设完美CSI）
        if self.perfect_csi:
            rx_complex = rx_complex / h.clamp_abs(min=1e-8)
        
        # 返回实部
        return rx_complex.real

@register_channel('RICIAN')
class RicianChannel(BaseChannel):
    """
    莱斯衰落信道
    
    模型: y = h*x + n，其中h服从莱斯分布，n ~ CN(0,n_var)
    
    参数:
        K: 莱斯K因子，表示视距分量与散射分量的功率比
        perfect_csi: 是否假设完美信道状态信息
    """
    def __init__(self, K: float = 1.0, perfect_csi: bool = True):
        super().__init__()
        self.K = K
        self.perfect_csi = perfect_csi

    def forward(self, tx: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        莱斯信道传播
        
        参数:
            tx: [batch, seq_len, latent_dim] 发送符号
            n_var: 噪声方差
            
        返回:
            [batch, seq_len, latent_dim] 接收符号
        """
        # 计算莱斯分布参数
        nlos_power = 1 / (self.K + 1)  # 非视距分量功率
        los_power = self.K / (self.K + 1)  # 视距分量功率
        
        # 生成信道系数
        h_nlos_real = torch.randn_like(tx) * math.sqrt(nlos_power/2)
        h_nlos_imag = torch.randn_like(tx) * math.sqrt(nlos_power/2)
        
        # 视距分量（确定性）
        h_los_real = torch.ones_like(tx) * math.sqrt(los_power)
        h_los_imag = torch.zeros_like(tx)
        
        # 合并视距和非视距分量
        h_real = h_los_real + h_nlos_real
        h_imag = h_los_imag + h_nlos_imag
        h = torch.complex(h_real, h_imag)
        
        # 添加噪声
        noise_real = torch.randn_like(tx) * math.sqrt(n_var/2)
        noise_imag = torch.randn_like(tx) * math.sqrt(n_var/2)
        noise = torch.complex(noise_real, noise_imag)
        
        # 信道传播
        tx_complex = torch.complex(tx, torch.zeros_like(tx))
        rx_complex = h * tx_complex + noise
        
        # 信道均衡
        if self.perfect_csi:
            rx_complex = rx_complex / h.clamp_abs(min=1e-8)
        
        # 返回实部
        return rx_complex.real

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