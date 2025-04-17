# -*- coding: utf-8 -*-
"""
Beam Search 解码器
==================================================
基于论文设置：单向自回归、已知起始 <START>，直到生成 <END> 或 max_len
调用示例：
    from deepsc.decoding.beam_search import beam_search
    output_ids = beam_search(model, src, n_var, channel,
                             start_idx, end_idx, pad_idx,
                             beam_size=4, max_len=32)
"""
from __future__ import annotations
import torch
from typing import List

def beam_search(model, src: torch.Tensor, n_var: float, channel,
                start_idx: int, end_idx: int, pad_idx: int,
                beam_size: int = 4, max_len: int = 32) -> List[List[int]]:
    """
    :param model: 已加载权重的 DeepSC (torch.nn.Module)
    :param src:   [batch, seq_len] 源句子
    :return:      List[List[int]]  每个样本最佳输出序列（含 <END>）
    """
    device = src.device
    batch = src.size(0)
    # 初始化 beam：每个样本存 beam_size 个候选 (score, seq_tensor)
    beams = [[(0.0, torch.tensor([start_idx], device=device, dtype=torch.long))]
             for _ in range(batch)]

    with torch.no_grad():
        for _ in range(max_len - 1):
            new_beams = []
            for b in range(batch):
                candidates = []
                for score, seq in beams[b]:
                    if seq[-1].item() == end_idx:      # 已完结
                        candidates.append((score, seq))
                        continue
                    # 生成 logits
                    tgt_mask = (seq.unsqueeze(0) == pad_idx).unsqueeze(1).unsqueeze(2)
                    # 下三角 mask
                    sz = seq.size(0)
                    look = torch.triu(torch.ones((1, sz, sz), dtype=torch.bool, device=device), diagonal=1)
                    tgt_mask = tgt_mask | look
                    # forward
                    memory = model.encoder(src[b:b+1], (src[b:b+1] == pad_idx).unsqueeze(1).unsqueeze(2))
                    tx = model.channel_encoder(memory)
                    tx = model.channel_decoder(channel(tx, n_var))
                    dec_out = model.decoder(seq.unsqueeze(0), tx,
                                             tgt_mask, None)
                    logits = model.proj(dec_out)[:, -1, :]    # 取最后一时刻
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    topk_logprob, topk_idx = torch.topk(log_probs, beam_size)
                    for lp, idx in zip(topk_logprob, topk_idx):
                        candidates.append((score + lp.item(),
                                           torch.cat([seq, idx.unsqueeze(0)])))
                # 取 beam_size 个最优
                candidates.sort(key=lambda x: x[0], reverse=True)
                new_beams.append(candidates[:beam_size])
            beams = new_beams
    # 取分数最高的序列
    results = [max(b, key=lambda x: x[0])[1].tolist() for b in beams]
    return results
