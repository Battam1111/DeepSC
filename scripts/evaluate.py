# -*- coding: utf-8 -*-
"""
改进版评估脚本
==========
一次性输出 SNR ∈ {0,3,…,18} dB 下的  
1) BLEU‑1  
2) Sentence Similarity  
3) 互信息估计结果

支持严格版本的模型评估。

完善特性:
1. 使用原模型的MINE网络，保证互信息评估的一致性
2. 支持不同信道类型评估
3. 生成性能曲线图
4. CSV结果保存
5. 支持严格和非严格版本模型
"""
from __future__ import annotations
import math, torch, tqdm, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import os
import json

from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab     import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.models.deepsc_strict import DeepSCStrict
from deepsc.models.mine_strict import MINEStrict
from deepsc.metrics.bleu   import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity
from deepsc.models import get_channel

# --------------------------- 工具：MI 估计 --------------------------- #
@torch.no_grad()
def estimate_mi_with_trained_mine(mine, tx, rx) -> float:
    """
    使用已训练的 MINE 网络来估计互信息下界
    
    参数:
        mine: 训练好的MINE网络
        tx: 发送符号
        rx: 接收符号
        
    返回:
        互信息下界估计值
    """
    tx_f = tx.reshape(-1, tx.size(-1))
    rx_f = rx.reshape(-1, rx.size(-1))
    
    # 过滤掉填充符号（功率接近0的符号）
    mask = (tx_f.abs().sum(dim=1) > 1e-6)
    if mask.sum() > 0:
        tx_f = tx_f[mask]
        rx_f = rx_f[mask]
        mi_lb = mine(tx_f, rx_f).item()
    else:
        mi_lb = 0.0
        
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

    # ---------- 2. 设备 & 模型类型确定 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查是否是严格版本模型
    is_strict_version = cfg.get("strict_model", False)
    print(f"模型类型: {'严格版本' if is_strict_version else '标准版本'}")
    
    # ---------- 3. 加载模型 ----------
    # 加载检查点
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if is_strict_version:
        # 严格版本模型加载
        if 'model' in checkpoint and 'mine' in checkpoint:
            # 直接加载严格版本检查点
            model = DeepSCStrict(cfg.model)
            model.load_state_dict(checkpoint['model'])
            
            mine = MINEStrict(latent_dim=cfg.model.latent_dim)
            mine.load_state_dict(checkpoint['mine'])
        else:
            # 尝试从Lightning检查点加载
            print("检查点格式不符合严格版本，尝试从Lightning检查点加载...")
            model = DeepSCStrict(cfg.model)
            
            # 提取模型权重
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                              if k.startswith('model.')}
                model.load_state_dict(state_dict)
                
                # 提取MINE权重
                mine_state_dict = {k.replace('mine.', ''): v for k, v in checkpoint['state_dict'].items() 
                                  if k.startswith('mine.')}
                mine = MINEStrict(latent_dim=cfg.model.latent_dim)
                mine.load_state_dict(mine_state_dict)
            else:
                raise ValueError("无法从检查点加载严格版本模型")
    else:
        # 标准版本模型加载
        try:
            # 首先尝试新版加载方式
            lit = LitDeepSC.load_from_checkpoint(ckpt_path, cfg=cfg)
            model = lit.model
            mine = lit.mine
        except Exception as e:
            # 兼容性处理：尝试旧版加载方式
            print(f"使用新方法加载失败，尝试兼容模式: {e}")
            # 手动创建模型并加载状态
            lit = LitDeepSC(cfg)
            lit.load_state_dict(checkpoint['state_dict'])
            model = lit.model
            mine = lit.mine
    
    # 迁移到设备、设置评估模式、冻结参数
    model = model.to(device).eval()
    mine = mine.to(device).eval()
    
    for param in model.parameters():
        param.requires_grad = False
    for param in mine.parameters():
        param.requires_grad = False
        
    # 创建信道模型
    channel = get_channel(cfg.data.channel)().to(device)
    print(f"信道类型: {cfg.data.channel}")

    # ---------- 4. 数据 ----------
    print(f"加载数据: {val_pkl}")
    vocab = Vocab.load(vocab_json)
    test_loader = make_dataloader(
        str(val_pkl),
        batch_size = cfg.train.batch_size,
        pad_idx    = vocab.token2idx['<PAD>'],
        shuffle    = False,
        num_workers = 4,
    )
    try:
        print(f"测试集大小: {len(test_loader.dataset)} 句")
    except TypeError:
        print("测试集大小: 未知 (可能是IterableDataset)")

    # ---------- 5. 指标曲线 ----------
    # SNR设置：可从命令行覆盖
    snrs = cfg.get("eval_snrs", [0, 3, 6, 9, 12, 15, 18])
    
    # 结果收集
    bleu_c, sim_c, mi_c = [], [], []
    
    # 设置AMP上下文（如果启用）
    amp_enabled = (str(cfg.get('precision', '32')) in ['16', '16-mixed', 16]) and device.type == 'cuda'
    if amp_enabled:
        print("启用自动混合精度(AMP)进行评估")
        amp_ctx = torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext() # 使用无操作上下文

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
                if is_strict_version:
                    logits, tx, rx = model(batch, n_var, channel, return_tx_rx=True)
                else:
                    logits, tx, rx = model(batch, n_var, channel, return_tx_rx=True)
                
                pred = logits.argmax(dim=-1)

                # 2) BLEU‑1
                # 对齐预测和目标，取正确部分比较
                if not is_strict_version:
                    batch_bleu = bleu_score(pred, batch[:, 1:])
                else:
                    # 严格版本比较整句
                    batch_bleu = bleu_score(pred, batch[:, 1:])

                bleu_l.append(batch_bleu)

                # 3) Sentence Sim
                try:
                    str_pred = [' '.join(vocab.decode(x.tolist())) for x in pred.cpu()]
                    str_ref  = [' '.join(vocab.decode(x[1:].tolist())) for x in batch.cpu()]  # 去掉<START>
                    batch_sim = sentence_similarity(str_pred, str_ref, device=device)
                    sim_l.extend(batch_sim)
                except Exception as e:
                    print(f"计算句子相似度时出错: {e}")
                    batch_sim = [0.0] * pred.size(0)  # 使用默认值
                    sim_l.extend(batch_sim)

                # 4) MI 下界
                batch_mi = estimate_mi_with_trained_mine(mine, tx, rx)
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

    # ---------- 6. 保存结果 ----------
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # 构建输出文件名
    model_type = "strict" if is_strict_version else "standard"
    output_prefix = ckpt_path.stem.replace('best-','').replace('.ckpt','')
    if not output_prefix:
        output_prefix = f"deepsc_{model_type}_{cfg.data.channel}"
    
    # CSV格式保存
    results_df = pd.DataFrame({
        'SNR_dB': snrs,
        'BLEU': bleu_c,
        'SentenceSimilarity': sim_c,
        'MI_LB': mi_c
    })
    
    csv_path = results_dir / f"{output_prefix}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # ---------- 7. 绘图 ----------
    plt.figure(figsize=(12, 10))
    
    # BLEU曲线
    plt.subplot(3, 1, 1)
    plt.plot(snrs, bleu_c, 'o-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU-1 Score')
    plt.title(f'BLEU Score vs. SNR - {model_type.capitalize()} Model')
    
    # 句子相似度曲线
    plt.subplot(3, 1, 2)
    plt.plot(snrs, sim_c, 's-', color='orange', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sentence Similarity')
    plt.title(f'Sentence Similarity vs. SNR - {model_type.capitalize()} Model')
    
    # 互信息曲线
    plt.subplot(3, 1, 3)
    plt.plot(snrs, mi_c, '^-', color='green', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mutual Information')
    plt.title(f'Mutual Information vs. SNR - {model_type.capitalize()} Model')
    
    plt.tight_layout()
    
    plot_path = results_dir / f"{output_prefix}_performance_curves.png"
    plt.savefig(plot_path, dpi=300)
    print(f"性能曲线已保存到: {plot_path}")

    # ---------- 8. 打印结果 ----------
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
    
    # 与原论文结果比较(如果可用)
    try:
        paper_results_path = Path("docs/paper_results.json")
        if paper_results_path.exists():
            with open(paper_results_path, 'r') as f:
                paper_results = json.load(f)
            
            if str(snr_db) in paper_results:
                paper_bleu = paper_results[str(snr_db)].get('bleu', None)
                if paper_bleu is not None:
                    print(f"\n与论文结果比较 (SNR={snr_db}dB):")
                    print(f"  • 论文BLEU: {paper_bleu:.4f}")
                    print(f"  • 本实现BLEU: {bleu_c[snrs.index(snr_db)]:.4f}")
                    print(f"  • 差异: {bleu_c[snrs.index(snr_db)] - paper_bleu:.4f}")
    except Exception as e:
        # 忽略任何与论文结果比较的错误
        pass

if __name__ == "__main__":
    main()