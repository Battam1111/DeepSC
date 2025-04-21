# -*- coding: utf-8 -*-
"""
单文件推理：
    python scripts/inference.py \
        ckpt_path=/path/best.ckpt \
        mode=beam \
        snr=6
"""
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pathlib import Path

import torch, tqdm

from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.decoding.beam_search import beam_search
from deepsc.metrics.bleu import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ---------- 1. 路径解析与校验 ----------
    ckpt_path  = to_absolute_path(cfg.ckpt_path)
    val_pkl    = to_absolute_path(cfg.data.val_pkl)
    vocab_json = to_absolute_path(cfg.data.vocab_json)

    for p in [ckpt_path, val_pkl, vocab_json]:
        if not Path(p).exists():
            raise FileNotFoundError(f"未找到文件：{p}，请检查配置或使用 CLI 覆盖参数")

    # ---------- 2. 加载模型 & 数据 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 兼容性加载模型
    try:
        # 新版加载方式
        lit = LitDeepSC.load_from_checkpoint(ckpt_path, cfg=cfg)
    except Exception as e:
        print(f"使用新方法加载失败，尝试兼容模式: {e}")
        # 加载原始检查点文件
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 手动创建模型并加载状态
        lit = LitDeepSC(cfg)
        lit.load_state_dict(checkpoint['state_dict'])
        
    lit = lit.to(device).eval()
    
    vocab  = Vocab.load(vocab_json)

    loader = make_dataloader(
        val_pkl,
        batch_size = getattr(cfg, 'batch_size', cfg.train.batch_size),
        pad_idx    = vocab.token2idx['<PAD>'],
        shuffle    = False,
    )

    # ---------- 3. 推理参数 ----------
    snr  = cfg.get('snr', 6)
    mode = cfg.get('mode', 'greedy')
    n_var = 1 / (10 ** (snr / 10) * 2) ** 0.5

    bleu_scores = []
    sim_scores  = []

    for batch in tqdm.tqdm(loader, desc=f"Inference SNR={snr}dB"):
        batch = batch.to(device)
        if mode == 'beam':
            outs = beam_search(
                lit.model, batch, n_var, lit.channel,
                vocab.token2idx['<START>'],
                vocab.token2idx['<END>'],
                vocab.token2idx['<PAD>'],
                beam_size=4,
                max_len=cfg.model.max_len
            )
            # pad 到 max_len
            pred = torch.tensor([
                o + [vocab.token2idx['<PAD>']] * (cfg.model.max_len - len(o))
                for o in outs
            ], device=device)
        else:
            logits = lit.model(batch, n_var, lit.channel)
            pred   = logits.argmax(dim=-1)

        # 计算指标
        bleu_scores.append(bleu_score(pred, batch))
        str_pred = [' '.join(vocab.decode(p.tolist())) for p in pred.cpu()]
        str_ref  = [' '.join(vocab.decode(r.tolist())) for r in batch.cpu()]
        sim_scores.append(sentence_similarity(str_pred, str_ref, device=device))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_sim  = sum(sim_scores)  / len(sim_scores)
    print(f"SNR={snr}dB  BLEU‑1={avg_bleu:.4f}  SentenceSim={avg_sim:.4f}")

if __name__ == '__main__':
    main()