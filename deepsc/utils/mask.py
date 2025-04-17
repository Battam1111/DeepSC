# -*- coding: utf-8 -*-
"""
生成 Transformer 所需的各种掩码
"""
import torch

def padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    :param seq: [batch, seq_len] 的 token 下标
    :param pad_idx: <PAD> 的索引
    :return: [batch, 1, 1, seq_len] 形状，True 表示需要 mask
    """
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # bool

def subsequent_mask(size: int) -> torch.Tensor:
    """
    生成下三角矩阵（含对角线为0，右上角为1），用于解码阶段的 look‑ahead mask
    :param size: 序列长度
    """
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return subsequent
