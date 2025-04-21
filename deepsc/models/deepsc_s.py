# deepsc/models/deepsc_s.py - 语音语义通信系统
import torch
import torch.nn as nn
import math
from .transformer import MultiHeadAttention, FeedForward
from ..utils.power_norm import power_normalize

class AudioFeatureExtractor(nn.Module):
    """
    音频特征提取器
    
    将原始音频波形转换为适合语义编码的特征表示
    
    参数:
        in_channels: 输入音频通道数
        hidden_dim: 隐藏层维度
        n_layers: 卷积层数量
    """
    def __init__(self, in_channels=1, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = []
        
        # 第一层：时域到频域的转换
        layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ))
        
        # 中间层：特征提取
        for _ in range(n_layers - 2):
            layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 最后一层：降采样到目标维度
        layers.append(nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim)
        ))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        参数:
            x: [batch, channels, time_steps] 音频波形
        返回:
            [batch, time_steps', hidden_dim] 特征序列
        """
        # [batch, channels, time] -> [batch, hidden, time']
        x = self.network(x)
        # [batch, hidden, time'] -> [batch, time', hidden]
        x = x.transpose(1, 2)
        return x

class AudioSynthesizer(nn.Module):
    """
    音频合成器
    
    将语义特征转换回音频波形
    
    参数:
        hidden_dim: 隐藏层维度
        out_channels: 输出音频通道数
        n_layers: 转置卷积层数量
    """
    def __init__(self, hidden_dim=128, out_channels=1, n_layers=3):
        super().__init__()
        layers = []
        
        # 第一层：适配维度
        layers.append(nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ))
        
        # 中间层：特征转换
        for _ in range(n_layers - 2):
            layers.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 最后一层：生成波形
        layers.append(nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=16, stride=8, padding=4),
            nn.Tanh()  # 将输出限制在 [-1, 1] 范围内，符合归一化音频
        ))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        参数:
            x: [batch, time_steps, hidden_dim] 特征序列
        返回:
            [batch, channels, time_steps'] 重建的音频波形
        """
        # [batch, time, hidden] -> [batch, hidden, time]
        x = x.transpose(1, 2)
        # [batch, hidden, time] -> [batch, channels, time']
        x = self.network(x)
        return x

class AudioEncoderLayer(nn.Module):
    """音频编码器层，基于Transformer结构"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class AudioEncoder(nn.Module):
    """音频编码器，提取音频语义信息"""
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.feature_extractor = AudioFeatureExtractor(in_channels=1, hidden_dim=input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            AudioEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        参数:
            x: [batch, channels, time] 音频输入
        返回:
            [batch, seq_len, d_model] 语义编码
        """
        # 特征提取
        x = self.feature_extractor(x)
        # 投影到模型维度
        x = self.input_proj(x)
        # 添加位置编码
        x = self.pos_encoder(x)
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        return x

class AudioDecoder(nn.Module):
    """音频解码器，生成音频波形"""
    def __init__(self, d_model, output_dim, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            AudioEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, output_dim)
        self.synthesizer = AudioSynthesizer(hidden_dim=output_dim, out_channels=1)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: [batch, seq_len, d_model] 语义编码
        返回:
            [batch, channels, time] 重建的音频波形
        """
        # 输入投影
        x = self.input_proj(x)
        # 位置编码
        x = self.pos_encoder(x)
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        # 输出投影
        x = self.output_proj(x)
        # 音频合成
        x = self.synthesizer(x)
        return x

class PositionalEncoding(nn.Module):
    """位置编码，与文本版本相同"""
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DeepSC_S(nn.Module):
    """
    语音语义通信系统 (DeepSC-S)
    
    端到端的语音传输系统，通过语义编码实现鲁棒性传输
    
    参数:
        cfg: 配置字典，包含模型参数
    """
    def __init__(self, cfg):
        super().__init__()
        # 模型配置参数
        d_model = cfg['d_model']
        n_layers = cfg['n_layers']
        n_heads = cfg['n_heads']
        d_ff = cfg['d_ff']
        dropout = cfg['dropout']
        latent_dim = cfg['latent_dim']
        
        # 音频编码器
        self.encoder = AudioEncoder(
            input_dim=128,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 信道编码器
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        
        # 信道解码器
        self.channel_decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 音频解码器
        self.decoder = AudioDecoder(
            d_model=d_model,
            output_dim=128,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
    def forward(self, x, n_var, channel, *, return_tx_rx=False):
        """
        端到端前向传播
        
        参数:
            x: [batch, channels, time] 输入音频波形
            n_var: 噪声方差
            channel: 信道模型
            return_tx_rx: 是否返回发送/接收的信道符号
            
        返回:
            重建的音频波形，或元组 (重建波形, 发送符号, 接收符号)
        """
        # 编码
        enc = self.encoder(x)
        
        # 信道编码
        tx = power_normalize(self.channel_encoder(enc))
        
        # 信道传输
        rx = channel(tx, n_var)
        
        # 信道解码
        dec_input = self.channel_decoder(rx)
        
        # 解码生成音频
        out = self.decoder(dec_input)
        
        if return_tx_rx:
            return out, tx, rx
        else:
            return out