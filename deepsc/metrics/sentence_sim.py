# -*- coding: utf-8 -*-
"""
改动：
1. 使用 bert-base-cased ；2. 取 Encoder-11-FeedForward-Norm 输出并做 sum‑pooling
"""
from __future__ import annotations
import torch, functools
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from typing import List

@functools.lru_cache()
def _load_bert(device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased',
                                      output_hidden_states=True).eval().to(device)
    return tokenizer, model

def _get_sent_vec(hidden_states):
    # hidden_states: tuple(len=13) 取第 12 层 (11 index) -> [batch, seq, 768]
    vec = hidden_states[12]               # Encoder‑11 FFN‑Norm
    vec = vec.sum(dim=1)                  # sum pooling
    return normalize(vec.cpu().numpy())

def sentence_similarity(batch_pred: List[str], batch_ref: List[str],
                        device: str | torch.device = 'cpu') -> List[float]:
    tokenizer, model = _load_bert(device)
    tok = tokenizer(batch_pred + batch_ref,
                    padding=True, truncation=True,
                    max_length=32, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**tok, output_hidden_states=True)
    vec = _get_sent_vec(outputs.hidden_states)
    v_pred, v_ref = vec[:len(batch_pred)], vec[len(batch_pred):]
    score = (v_pred * v_ref).sum(axis=1)
    return score.tolist()
