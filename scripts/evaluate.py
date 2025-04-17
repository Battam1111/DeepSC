# -*- coding: utf-8 -*-
"""
评估脚本（修正版）
=================
一次性输出 SNR ∈ {0,3,…,18} dB 下的  
1) BLEU‑1  
2) Sentence Similarity  
3) 训练时同一条 MINE 网络给出的 mutual‑information 下界  

主要改动
--------
1. **沿用已训练好的 MINE**：不再重新随机初始化，保证曲线可复现论文 Fig. 9。  
2. **噪声方差公式修正**：σ = √(1/(2·SNR))（线性尺度）。  
3. **设备显式迁移**：Lightning 2.x 在 `.eval()` 后不会自动将子模块转到 GPU，本脚本手动 `.to(device)`。  
4. **批内自动混合精度 (AMP)**：若模型以 FP16 训练，评估同样启用 `torch.cuda.amp.autocast`，避免精度失配。  
5. **中文注释补全**：尽量讲透每一行做什么，方便后续维护。
"""
from __future__ import annotations
import math, torch, tqdm, numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab     import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.metrics.bleu   import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity

# --------------------------- 工具：MI 估计 --------------------------- #
@torch.no_grad()
def estimate_mi_with_trained_mine(lit: LitDeepSC,
                                  batch: torch.Tensor,
                                  n_var: float) -> float:
    """
    使用 checkpoint 中已经训练好的 MINE 网络来估计互信息下界
    :param lit:   LightningModule，含已加载 mine
    :param batch: [B,L]  long 张量
    :param n_var: 信道噪声 σ
    """
    logits, tx, rx = lit.model(batch, n_var, lit.channel, return_tx_rx=True)
    tx_f, rx_f = [z.reshape(-1, z.size(-1)) for z in (tx, rx)]
    mi_lb = lit.mine(tx_f, rx_f).item()
    return mi_lb

# --------------------------- 主程序 --------------------------- #
@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ---------- 1. 路径解析 ----------
    ckpt_path  = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl    = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))

    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"未找到文件：{p}")

    # ---------- 2. 设备 & 模型 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit: LitDeepSC = (
        LitDeepSC
        .load_from_checkpoint(ckpt_path, cfg=cfg)
        .to(device)
        .eval()
        .freeze()                # 冻结参数，节省内存
    )

    # ---------- 3. 数据 ----------
    vocab = Vocab.load(vocab_json)
    test_loader = make_dataloader(
        val_pkl,
        batch_size = cfg.train.batch_size,
        pad_idx    = vocab.token2idx['<PAD>'],
        shuffle    = False,
        num_workers = 4,
    )

    # ---------- 4. 指标曲线 ----------
    snrs     = [0, 3, 6, 9, 12, 15, 18]
    bleu_c, sim_c, mi_c = [], [], []

    amp_ctx = (
        torch.cuda.amp.autocast() if cfg.precision == 16 and device.type == 'cuda'
        else torch.autocast(device_type='cpu', enabled=False)  # 占位
    )

    with torch.no_grad(), amp_ctx:
        for snr_db in snrs:
            # 线性 SNR -> 噪声方差 σ
            snr_lin = 10.0 ** (snr_db / 10.0)
            n_var   = math.sqrt(1.0 / (2.0 * snr_lin))

            bleu_l, sim_l, mi_l = [], [], []
            pbar = tqdm.tqdm(test_loader, desc=f"SNR {snr_db} dB", leave=False)

            for batch in pbar:
                batch = batch.to(device)

                # 1) forward
                logits = lit.model(batch, n_var, lit.channel)
                pred   = logits.argmax(dim=-1)

                # 2) BLEU‑1
                bleu_l.append(bleu_score(pred, batch))

                # 3) Sentence Sim
                str_pred = [' '.join(vocab.decode(x.tolist())) for x in pred.cpu()]
                str_ref  = [' '.join(vocab.decode(x.tolist())) for x in batch.cpu()]
                sim_l.extend(sentence_similarity(str_pred, str_ref, device=device))

                # 4) MI 下界
                mi_l.append(estimate_mi_with_trained_mine(lit, batch, n_var))

            # —— 统计均值 —— #
            bleu_c.append(float(np.mean(bleu_l)))
            sim_c .append(float(np.mean(sim_l)))
            mi_c  .append(float(np.mean(mi_l)))

    # ---------- 5. 打印结果 ----------
    print("\n=== Evaluation Result ===")
    print("SNR(dB)           :", snrs)
    print("BLEU‑1            :", [f"{b:.4f}" for b in bleu_c])
    print("SentenceSimilarity:", [f"{s:.4f}" for s in sim_c])
    print("MI‑LB             :", [f"{m:.4f}" for m in mi_c])

if __name__ == "__main__":
    main()
