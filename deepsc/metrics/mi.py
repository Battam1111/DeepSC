# -*- coding: utf-8 -*-
"""
快速评估互信息：前向一次，调用模型内置 mine
"""
import torch
from deepsc.models.mine import MINE

def estimate_mi(model, src, n_var, channel):
    with torch.no_grad():
        _, tx, rx = model(src, n_var, channel, return_tx_rx=True)
        mine = MINE(tx.size(-1)).to(src.device)
        mi = mine(tx.reshape(-1, tx.size(-1)), rx.reshape(-1, rx.size(-1)))
    return mi.item()
