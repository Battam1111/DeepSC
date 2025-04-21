# -*- coding: utf-8 -*-
"""
严格按照论文实现的DeepSC模型
============================

本模块提供了完全遵循论文的DeepSC实现，包括:
1. 3层Transformer编码器，8个注意力头
2. 特定结构的信道编码器和解码器
3. 16维信道符号
4. 准确的功率归一化实现

该实现确保与论文描述的架构完全一致。
"""

import torch
import torch.nn as nn
import math

from ..utils.power_norm import power_normalize

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为输入序列添加位置信息，使模型能够感知序列中的位置
    
    参数:
        d_model: 模型维度
        dropout: Dropout概率
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码
        
        参数:
            x: [batch, seq_len, d_model] 输入张量
            
        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (已修复 AMP 兼容性)

    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: Dropout概率
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # 线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        多头注意力机制前向传播

        参数:
            q: [batch, seq_len_q, d_model] 查询
            k: [batch, seq_len_k, d_model] 键
            v: [batch, seq_len_v, d_model] 值
            mask: [batch, 1, seq_len_q, seq_len_k] or [batch, 1, 1, seq_len_k] 布尔掩码
                  约定：模型生成的 mask 中 True 表示 *有效* 位置，False 表示 *无效* (padding/future) 位置。
                  masked_fill 需要填充 True 的位置，所以我们需要填充 mask 为 False 的地方。

        返回:
            [batch, seq_len_q, d_model] 注意力输出
        """
        batch_size = q.size(0)

        # 线性变换并切分头: [B, L, D] -> [B, H, L, Dk]
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力: scores [B, H, Lq, Lk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 获取 scores 张量当前的数据类型（可能是 float32 或 float16）
            # 并获取该类型能表示的最小值
            fill_value = torch.finfo(scores.dtype).min
            # 根据约定，mask 中 True 是有效位，False 是无效位
            # masked_fill 要求填充 True 的位置，所以我们填充 mask == False 的位置
            # PyTorch 会自动广播 mask (例如 [B, 1, 1, Lk] -> [B, H, Lq, Lk])
            scores = scores.masked_fill(mask == False, fill_value) # <--- 修改此行

        attn_weights = torch.softmax(scores, dim=-1)
        # 添加 nan_to_num 以防止因整行/列被屏蔽导致 softmax 输出 NaN
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # 计算加权和: context [B, H, Lq, Dk]
        output = torch.matmul(attn_weights, v)

        # 合并头: [B, H, Lq, Dk] -> [B, Lq, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.output_linear(output)

class FeedForward(nn.Module):
    """
    前馈神经网络
    
    由两个线性变换和一个ReLU激活组成
    
    参数:
        d_model: 输入/输出维度
        d_ff: 隐藏层维度
        dropout: Dropout概率
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch, seq_len, d_model] 输入张量
            
        返回:
            相同形状的输出张量
        """
        return self.net(x)

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    由自注意力机制和前馈网络组成
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        """
        编码器层前向传播
        
        参数:
            x: [batch, seq_len, d_model] 输入张量
            mask: [batch, 1, 1, seq_len] 掩码
            
        返回:
            [batch, seq_len, d_model] 输出张量
        """
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    由自注意力机制、编码器-解码器注意力机制和前馈网络组成
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        解码器层前向传播
        
        参数:
            x: [batch, seq_len, d_model] 输入张量
            enc_output: [batch, src_seq_len, d_model] 编码器输出
            src_mask: [batch, 1, 1, src_seq_len] 源序列掩码
            tgt_mask: [batch, 1, tgt_seq_len, tgt_seq_len] 目标序列掩码
            
        返回:
            [batch, seq_len, d_model] 输出张量
        """
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力子层
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x

class Encoder(nn.Module):
    """
    Transformer编码器
    
    由词嵌入、位置编码和多个编码器层组成
    
    参数:
        vocab_size: 词表大小
        d_model: 模型维度
        n_layers: 编码器层数
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率
        max_len: 最大序列长度
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, src, src_mask):
        """
        编码器前向传播
        
        参数:
            src: [batch, seq_len] 输入序列
            src_mask: [batch, 1, 1, seq_len] 源序列掩码
            
        返回:
            [batch, seq_len, d_model] 编码器输出
        """
        # 词嵌入和位置编码
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # 依次通过每个编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class Decoder(nn.Module):
    """
    Transformer解码器
    
    由词嵌入、位置编码和多个解码器层组成
    
    参数:
        vocab_size: 词表大小
        d_model: 模型维度
        n_layers: 解码器层数
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率
        max_len: 最大序列长度
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        """
        解码器前向传播
        
        参数:
            tgt: [batch, seq_len] 目标序列
            enc_output: [batch, src_seq_len, d_model] 编码器输出
            src_mask: [batch, 1, 1, src_seq_len] 源序列掩码
            tgt_mask: [batch, 1, tgt_seq_len, tgt_seq_len] 目标序列掩码
            
        返回:
            [batch, seq_len, d_model] 解码器输出
        """
        # 词嵌入和位置编码
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # 依次通过每个解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x

class ChannelEncoder(nn.Module):
    """
    信道编码器
    
    将语义特征压缩为适合传输的低维符号
    
    参数:
        d_model: 输入特征维度
        latent_dim: 输出符号维度
    """
    def __init__(self, d_model=512, latent_dim=16):
        super().__init__()
        
        # 严格按照论文实现两层结构
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        """
        信道编码器前向传播
        
        参数:
            x: [batch, seq_len, d_model] 输入特征
            
        返回:
            [batch, seq_len, latent_dim] 编码后的符号
        """
        return self.net(x)

class ChannelDecoder(nn.Module):
    """
    信道解码器
    
    将接收到的低维符号恢复为语义特征
    
    参数:
        latent_dim: 输入符号维度
        d_model: 输出特征维度
    """
    def __init__(self, latent_dim=16, d_model=512):
        super().__init__()
        
        # 严格按照论文实现结构
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        """
        信道解码器前向传播
        
        参数:
            x: [batch, seq_len, latent_dim] 接收到的符号
            
        返回:
            [batch, seq_len, d_model] 解码后的特征
        """
        return self.net(x)

class DeepSCStrict(nn.Module):
    """
    DeepSC：严格按照论文实现的深度学习赋能语义通信系统
    
    参数:
        config: 配置字典，包含模型参数
    """
    def __init__(self, config):
        super().__init__()
        
        # 提取配置参数
        vocab_size = config['vocab_size']
        d_model = config['d_model']
        n_layers = config['n_layers']
        n_heads = config['n_heads']
        d_ff = config['d_ff']
        latent_dim = config['latent_dim']
        dropout = config['dropout']
        pad_idx = config['pad_idx']
        max_len = config.get('max_len', 100)
        
        self.pad_idx = pad_idx
        
        # 语义编码器
        self.encoder = Encoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        
        # 信道编码器
        self.channel_encoder = ChannelEncoder(d_model, latent_dim)
        
        # 信道解码器
        self.channel_decoder = ChannelDecoder(latent_dim, d_model)
        
        # 语义解码器
        self.decoder = Decoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _create_masks(self, src, tgt):
        """
        创建源序列和目标序列的掩码
        
        参数:
            src: [batch, src_len] 源序列
            tgt: [batch, tgt_len] 目标序列
            
        返回:
            (src_mask, tgt_mask) 掩码元组
        """
        # 源序列填充掩码：[batch, 1, 1, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 目标序列填充掩码：[batch, 1, tgt_len]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 目标序列后续掩码：[1, tgt_len, tgt_len]
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.triu(
            torch.ones((1, tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
            diagonal=1
        )
        
        # 组合填充掩码和后续掩码：[batch, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & ~tgt_sub_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, n_var, channel, return_tx_rx=False):
        """
        DeepSC前向传播
        
        参数:
            src: [batch, seq_len] 输入序列
            n_var: 噪声方差
            channel: 信道模型对象
            return_tx_rx: 是否返回发送/接收符号
            
        返回:
            logits或(logits, tx, rx)元组
        """
        # 准备目标序列（输入序列向右移一位）
        tgt = src[:, :-1]
        
        # 创建掩码
        src_mask, tgt_mask = self._create_masks(src, tgt)
        
        # 语义编码
        enc_output = self.encoder(src, src_mask)
        
        # 信道编码
        tx = self.channel_encoder(enc_output)
        
        # 功率归一化
        tx = power_normalize(tx)
        
        # 信道传输
        rx = channel(tx, n_var)
        
        # 信道解码
        dec_input = self.channel_decoder(rx)
        
        # 语义解码
        dec_output = self.decoder(tgt, dec_input, src_mask, tgt_mask)
        
        # 输出投影
        logits = self.output_proj(dec_output)
        
        if return_tx_rx:
            return logits, tx, rx
        else:
            return logits