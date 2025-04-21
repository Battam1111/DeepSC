# scripts/evaluate.py - 修正后的评估脚本
# -*- coding: utf-8 -*-
"""
评估脚本
==========
一次性输出 SNR ∈ {0,3,…,18} dB 下的  
1) BLEU‑1  
2) Sentence Similarity  
3) 互信息估计结果

完善特性:
1. 使用原模型的MINE网络，保证互信息评估的一致性
2. 支持不同信道类型评估
3. 生成性能曲线图
4. CSV结果保存
"""
from __future__ import annotations
import math, torch, tqdm, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
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
    
    参数:
        lit:   LightningModule，含已加载 mine
        batch: [B,L]  long 张量
        n_var: 信道噪声 σ
        
    返回:
        互信息下界估计值
    """
    logits, tx, rx = lit.model(batch, n_var, lit.channel, return_tx_rx=True)
    tx_f, rx_f = [z.reshape(-1, z.size(-1)) for z in (tx, rx)]
    mi_lb = lit.mine(tx_f, rx_f).item()
    return mi_lb

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    主评估函数
    
    参数:
        cfg: Hydra配置对象
    """
    # ---------- 1. 路径解析 ----------
    ckpt_path  = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl    = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))

    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"未找到文件：{p}")

    # ---------- 2. 设备 & 模型 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载检查点
    try:
        # 首先尝试新版加载方式
        print(f"加载模型: {ckpt_path}")
        lit = LitDeepSC.load_from_checkpoint(ckpt_path, cfg=cfg)
    except Exception as e:
        # 兼容性处理：尝试旧版加载方式
        print(f"使用新方法加载失败，尝试兼容模式: {e}")
        # 加载原始检查点文件
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 手动创建模型并加载状态
        lit = LitDeepSC(cfg)
        lit.load_state_dict(checkpoint['state_dict'])
    
    # 迁移到设备、设置评估模式、冻结参数
    lit = lit.to(device).eval().freeze()
    print(f"信道类型: {cfg.data.channel}")

    # ---------- 3. 数据 ----------
    print(f"加载数据: {val_pkl}")
    vocab = Vocab.load(vocab_json)
    test_loader = make_dataloader(
        val_pkl,
        batch_size = cfg.train.batch_size,
        pad_idx    = vocab.token2idx['<PAD>'],
        shuffle    = False,
        num_workers = 4,
    )
    print(f"测试集大小: {len(test_loader.dataset)} 句")

    # ---------- 4. 指标曲线 ----------
    # SNR设置：可从命令行覆盖
    snrs = cfg.get("eval_snrs", [0, 3, 6, 9, 12, 15, 18])
    
    # 结果收集
    bleu_c, sim_c, mi_c = [], [], []
    
    # 设置AMP上下文（如果启用）
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
            pbar = tqdm.tqdm(test_loader, desc=f"SNR {snr_db} dB", leave=False)

            for batch in pbar:
                batch = batch.to(device)

                # 1) forward
                logits = lit.model(batch, n_var, lit.channel)
                pred   = logits.argmax(dim=-1)

                # 2) BLEU‑1
                batch_bleu = bleu_score(pred, batch)
                bleu_l.append(batch_bleu)

                # 3) Sentence Sim
                str_pred = [' '.join(vocab.decode(x.tolist())) for x in pred.cpu()]
                str_ref  = [' '.join(vocab.decode(x.tolist())) for x in batch.cpu()]
                batch_sim = sentence_similarity(str_pred, str_ref, device=device)
                sim_l.extend(batch_sim)

                # 4) MI 下界
                batch_mi = estimate_mi_with_trained_mine(lit, batch, n_var)
                mi_l.append(batch_mi)
                
                # 更新进度条
                pbar.set_postfix({
                    'BLEU': f"{batch_bleu:.4f}",
                    'Sim': f"{np.mean(batch_sim):.4f}",
                    'MI': f"{batch_mi:.4f}"
                })

            # ---- 统计均值 ---- #
            bleu_c.append(float(np.mean(bleu_l)))
            sim_c .append(float(np.mean(sim_l)))
            mi_c  .append(float(np.mean(mi_l)))

    # ---------- 5. 保存结果 ----------
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # CSV格式保存
    results_df = pd.DataFrame({
        'SNR_dB': snrs,
        'BLEU': bleu_c,
        'SentenceSimilarity': sim_c,
        'MI_LB': mi_c
    })
    
    csv_path = results_dir / f"results_{cfg.data.channel}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # ---------- 6. 绘图 ----------
    plt.figure(figsize=(12, 10))
    
    # BLEU曲线
    plt.subplot(3, 1, 1)
    plt.plot(snrs, bleu_c, 'o-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU-1 Score')
    plt.title('BLEU Score vs. SNR')
    
    # 句子相似度曲线
    plt.subplot(3, 1, 2)
    plt.plot(snrs, sim_c, 's-', color='orange', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sentence Similarity')
    plt.title('Sentence Similarity vs. SNR')
    
    # 互信息曲线
    plt.subplot(3, 1, 3)
    plt.plot(snrs, mi_c, '^-', color='green', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information vs. SNR')
    
    plt.tight_layout()
    plt.savefig(results_dir / f"performance_curves_{cfg.data.channel}.png", dpi=300)
    print(f"性能曲线已保存到: {results_dir}/performance_curves_{cfg.data.channel}.png")

    # ---------- 7. 打印结果 ----------
    print("\n=== 评估结果摘要 ===")
    print("SNR(dB)           :", snrs)
    print("BLEU‑1            :", [f"{b:.4f}" for b in bleu_c])
    print("SentenceSimilarity:", [f"{s:.4f}" for s in sim_c])
    print("MI‑LB             :", [f"{m:.4f}" for m in mi_c])
    
    # 性能指标
    best_snr_idx = np.argmax(sim_c)
    best_snr = snrs[best_snr_idx]
    print(f"\n最佳性能点 (SNR = {best_snr} dB):")
    print(f"  • BLEU-1: {bleu_c[best_snr_idx]:.4f}")
    print(f"  • Sentence Similarity: {sim_c[best_snr_idx]:.4f}")
    print(f"  • Mutual Information: {mi_c[best_snr_idx]:.4f}")

if __name__ == "__main__":
    main()