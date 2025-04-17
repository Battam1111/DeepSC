# -*- coding: utf-8 -*-
"""
BLEU-1 评估工具
================

本模块提供 bleu_score 函数，用于计算一个 batch 的
BLEU-1 分数（只考虑 1-grams），返回平均得分。

依赖:
    nltk（请确保已通过 `pip install nltk` 安装）
"""

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 加载平滑函数，避免极端情况下分数为 0
_smooth_fn = SmoothingFunction().method1


def bleu_score(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """
    计算一个 batch 的平均 BLEU-1 分数

    :param pred: torch.LongTensor, 形状 [batch, seq_len]，模型预测的索引序列
    :param ref:  torch.LongTensor, 形状 [batch, seq_len]，参考（真实）索引序列
    :return:     float，平均 BLEU-1 分数（0.0～1.0）

    说明:
    - 本实现只计算 1-gram（unigram）精度，即 BLEU-1。
    - 会自动忽略 <PAD>（假设索引 0）和截断至 <END>（假设索引 2）。
    - 使用 NLTK 的 sentence_bleu 及平滑函数，保证极端情况下不为 0。
    """

    batch_size = pred.size(0)
    scores = []

    for i in range(batch_size):
        # 将第 i 条预测和参考转为 Python 列表
        pred_i = pred[i].tolist()
        ref_i = ref[i].tolist()

        # 内部函数：去除 PAD（0），并在遇到 END（2）时停止
        def _truncate_and_filter(seq: list) -> list:
            out = []
            for tok in seq:
                if tok == 2:  # <END>
                    break
                if tok == 0:  # <PAD>
                    continue
                # 转为 str 方便 NLTK 处理
                out.append(str(tok))
            return out

        hyp = _truncate_and_filter(pred_i)
        ref_clean = _truncate_and_filter(ref_i)

        # 若任一序列为空，则打 0 分
        if len(hyp) == 0 or len(ref_clean) == 0:
            scores.append(0.0)
        else:
            # 计算 BLEU-1: weights=(1,0,0,0)
            score = sentence_bleu(
                [ref_clean],  # 参考列表
                hyp,         # 候选句子
                weights=(1, 0, 0, 0),
                smoothing_function=_smooth_fn
            )
            scores.append(score)

    # 返回 batch 平均
    return float(sum(scores) / batch_size)
