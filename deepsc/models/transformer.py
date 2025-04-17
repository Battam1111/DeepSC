# -*- coding: utf-8 -*-
"""
DeepSC 主体模型
==================================================
由四部分组成：
1. Encoder(Transformer)    —— 语义编码
2. ChannelEncoder          —— 压缩→16 维符号
3. ChannelDecoder          —— 16 维符号→语义
4. Decoder(Transformer)    —— 语义解码到词概率
"""
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsc.utils.mask import subsequent_mask
from deepsc.utils.power_norm import power_normalize


# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 预计算位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------- Multi‑Head Attention ----------
class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块
    参数:
      d_model:  模型维度（词向量维度）
      n_heads:  注意力头数
      dropout:  Dropout 概率
    输入:
      q, k, v:  张量形状 [batch, seq_len, d_model]
      mask:     掩码张量，形状 [batch, 1, 1, seq_len] 或 [batch, heads, seq_len, seq_len]，
                dtype=torch.bool，True 表示要屏蔽的位置
    输出:
      context:  张量形状 [batch, seq_len, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = .1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 线性投影：Q, K, V, 以及输出映射
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        切分头：将 [batch, seq_len, d_model]
        → [batch, n_heads, seq_len, d_k]
        """
        b, l, _ = x.size()
        # 先 reshape 成 [batch, seq_len, n_heads, d_k]，再交换维度
        return x.view(b, l, self.n_heads, self.d_k) \
                .transpose(1, 2)  # → [batch, n_heads, seq_len, d_k]

    def _scaled_dot(self,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    mask: torch.Tensor = None
                   ) -> torch.Tensor:
        """
        核心打分与加权：scaled dot‑product attention
        q,k,v: [batch, heads, seq_len, d_k]
        mask:  同步 broadcast 到 [batch, heads, seq_len, seq_len]
        """
        # 1) Q·K^T 并缩放
        #    score: [batch, heads, seq_len, seq_len]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2) 如果给定 mask，则将 mask 位置填充为非常小的值 (近似 -∞)，这样 softmax 后贴近 0
        if mask is not None:
            # 确保 mask 为 bool 类型
            mask_bool = mask.bool()
            # 动态获取该 dtype 能表示的最小值
            # half: 约 -6.55e4；float: 约 -3.4e38
            fill_value = torch.finfo(score.dtype).min
            # 直接填充
            score = score.masked_fill(mask_bool, fill_value)

        # 3) softmax + dropout
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # 4) 加权求和得到上下文
        #    context: [batch, heads, seq_len, d_k]
        context = torch.matmul(attn, v)
        return context

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None
               ) -> torch.Tensor:
        """
        前向传播接口
        q, k, v: [batch, seq_len, d_model]
        mask:    [batch, 1, 1, seq_len] 或 [batch, heads, seq_len, seq_len]，True 表示屏蔽
        返回:
          [batch, seq_len, d_model]
        """
        # 1) 线性变换 + 切分头
        q_heads = self._split_heads(self.q_proj(q))
        k_heads = self._split_heads(self.k_proj(k))
        v_heads = self._split_heads(self.v_proj(v))

        # 2) 逐头做 scaled dot‑product attention
        context = self._scaled_dot(q_heads, k_heads, v_heads, mask)

        # 3) 合并头： [batch, heads, seq_len, d_k] → [batch, seq_len, d_model]
        context = context.transpose(1, 2) \
                         .contiguous() \
                         .view(q.size(0), -1, self.n_heads * self.d_k)

        # 4) 最后的线性输出
        output = self.o_proj(context)
        return output


# ---------- Position‑wise Feed‑Forward ----------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = .1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ---------- Encoder / Decoder Layer ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.mha(x, x, x, mask)
        x = self.norm1(x)
        x = x + self.ffn(x)
        return self.norm2(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.src_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask, src_mask):
        x = x + self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x)
        x = x + self.src_mha(x, memory, memory, src_mask)
        x = self.norm2(x)
        x = x + self.ffn(x)
        return self.norm3(x)


# ---------- Encoder / Decoder Stack ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, src, mask):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


# ---------- DeepSC 顶层 ----------
class DeepSC(nn.Module):
    """
    forward(src, n_var, channel, *, return_tx_rx=False)
    """
    def __init__(self, cfg):
        super().__init__()
        self.pad_idx = cfg['pad_idx']
        self.encoder = Encoder(cfg['vocab_size'], cfg['max_len'],
                               cfg['n_layers'], cfg['d_model'],
                               cfg['n_heads'], cfg['d_ff'], cfg['dropout'])
        self.channel_encoder = nn.Sequential(
            nn.Linear(cfg['d_model'], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cfg['latent_dim'])
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(cfg['latent_dim'], cfg['d_model']),
            nn.ReLU(inplace=True),
            nn.Linear(cfg['d_model'], cfg['d_model']),
            nn.LayerNorm(cfg['d_model'])
        )
        self.decoder = Decoder(cfg['vocab_size'], cfg['max_len'],
                               cfg['n_layers'], cfg['d_model'],
                               cfg['n_heads'], cfg['d_ff'], cfg['dropout'])
        self.proj = nn.Linear(cfg['d_model'], cfg['vocab_size'])

    # ----------- 高阶封装 -----------
    def forward(self, src: torch.Tensor, n_var: float,
                channel, *, return_tx_rx: bool = False):
        """
        :param src: [batch, seq_len]  输入同时作为 target（自回归）
        :param n_var: 噪声方差
        :param channel: 已实例化的信道对象
        :return: pred(logits) 或 (pred, tx, rx)
        """
        src_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_inp = src[:, :-1]
        tgt_real = src[:, 1:]
        tgt_mask = (tgt_inp == self.pad_idx).unsqueeze(1).unsqueeze(2)
        look = subsequent_mask(tgt_inp.size(1)).to(src.device)
        tgt_mask = tgt_mask | look  # OR: True 表示禁用注意力

        # ---------- 前向传播 ----------
        enc = self.encoder(src, src_mask)
        tx = power_normalize(self.channel_encoder(enc))
        rx = channel(tx, n_var)
        dec_input = self.channel_decoder(rx)
        dec = self.decoder(tgt_inp, dec_input, tgt_mask, src_mask)
        logits = self.proj(dec)

        return (logits, tx, rx) if return_tx_rx else logits
