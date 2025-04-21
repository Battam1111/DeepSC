# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC/scripts/evaluate.py
# (已修改，解决 InterpolationKeyError, 更新 AMP API, 增强鲁棒性)
# --------------------------------------------------------------------------------
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
6. 解决了配置插值错误
7. 更新了 AMP API 调用
"""
from __future__ import annotations
import math, torch, tqdm, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors # 导入错误类型
from hydra.utils import to_absolute_path
import os
import json
import traceback # 用于打印详细错误堆栈

from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC # 标准版 Lightning Module
from deepsc.models.deepsc_strict import DeepSCStrict # 严格版模型
from deepsc.models.mine_strict import MINEStrict # 严格版 MINE (或标准版 MINE)
from deepsc.models.mine import MINE as StandardMINE # 标准版 MINE
from deepsc.metrics.bleu import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity
from deepsc.models import get_channel
# 导入 AMP 相关 (使用新 API)
from torch.amp import autocast
from contextlib import nullcontext

# --------------------------- 工具：MI 估计 --------------------------- #
@torch.no_grad()
def estimate_mi_with_trained_mine(mine, tx, rx) -> float:
    """
    使用已训练的 MINE 网络来估计互信息下界

    参数:
        mine: 训练好的MINE网络 (可以是 MINEStrict 或 StandardMINE, 或 None)
        tx: 发送符号
        rx: 接收符号

    返回:
        互信息下界估计值 (如果 mine 为 None 或计算出错则返回 0.0)
    """
    if mine is None:
        # print("警告: MINE 网络未提供，无法计算互信息。")
        return 0.0 # 如果没有 MINE 网络，则 MI 为 0

    tx_f = tx.reshape(-1, tx.size(-1))
    rx_f = rx.reshape(-1, rx.size(-1))

    # 过滤掉填充符号（功率接近0的符号）
    mask = (tx_f.abs().sum(dim=1) > 1e-6)
    if mask.sum() > 0:
        tx_f = tx_f[mask]
        rx_f = rx_f[mask]
        try:
            # 确保 MINE 在评估模式
            mine.eval()
            mi_lb = mine(tx_f, rx_f).item()
        except Exception as e:
             print(f"\n警告: MINE 前向计算出错: {e}")
             traceback.print_exc() # 打印详细错误
             mi_lb = 0.0 # 出错时返回 0
    else:
        mi_lb = 0.0 # 如果过滤后没有有效符号

    return mi_lb

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    主评估函数

    参数:
        cfg: Hydra配置对象
    """
    # ---------- 1. 路径解析 ----------
    ckpt_path = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))

    print("--- 检查文件路径 ---")
    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"未找到文件：{p}")
        print(f"  [√] 文件存在: {p}")

    # ---------- 2. 设备 & 模型类型确定 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 设备与模型 ---")
    print(f"使用设备: {device}")

    # 检查是否是严格版本模型 (现在 base.yaml 中有定义)
    is_strict_version = cfg.get("strict_model", False)
    print(f"模型类型: {'严格版本 (DeepSCStrict)' if is_strict_version else '标准版本 (LitDeepSC)'}")

    # ---------- 3. 加载词表 & 动态设置配置 ----------
    print(f"\n--- 加载数据与配置 ---")
    print(f"加载词表: {vocab_json}")
    vocab = Vocab.load(vocab_json)
    actual_vocab_size = len(vocab)
    pad_idx = vocab.token2idx['<PAD>']
    print(f"  实际词表大小: {actual_vocab_size}")
    print(f"  Pad Index: {pad_idx}")

    # --- 解决插值错误的关键步骤 ---
    # 在模型初始化前，确保 cfg.model 和 cfg.data 中的相关配置被正确设置
    print("动态更新配置中的 vocab_size 和 pad_idx...")
    try:
        # 允许临时修改结构
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(cfg.data, False)

        cfg.model.vocab_size = actual_vocab_size
        cfg.model.pad_idx = pad_idx # 更新模型配置中的 pad_idx
        cfg.data.vocab_size = actual_vocab_size # <--- 关键：设置 data.vocab_size
        cfg.data.pad_idx = pad_idx      # 更新数据配置中的 pad_idx

        # 恢复结构模式 (可选)
        OmegaConf.set_struct(cfg.model, True)
        OmegaConf.set_struct(cfg.data, True)
        print("  配置更新完成。")
    except omegaconf_errors.ConfigKeyError as e:
         print(f"  警告：更新配置时出错（可能是因为配置结构不允许添加新键）：{e}。如果后续模型初始化失败，请检查配置文件结构。")
    except Exception as e_cfg:
         print(f"  警告：更新配置时发生未知错误：{e_cfg}")


    # ---------- 4. 加载模型 ----------
    print(f"\n--- 加载模型 ---")
    print(f"加载模型检查点: {ckpt_path}")
    # 推荐：设置 weights_only=True 或处理警告
    # 为了保持行为一致性，暂时不设置，但注意安全风险
    checkpoint = torch.load(ckpt_path, map_location='cpu')#, weights_only=False)
    print(f"  检查点 keys: {list(checkpoint.keys())}")

    model = None
    mine = None # MINE 网络可能不存在或加载失败

    if is_strict_version:
        print("尝试加载严格版本模型 (DeepSCStrict)...")
        if 'model' in checkpoint:
            # 使用更新后的 cfg.model 创建模型
            try:
                model = DeepSCStrict(cfg.model)
                model.load_state_dict(checkpoint['model'], strict=False)
                print("  [√] 成功加载严格版本模型 state_dict。")
            except Exception as e_load_strict:
                 print(f"  [X] 加载严格版本模型 state_dict 失败: {e_load_strict}")
                 raise RuntimeError("无法加载严格版本模型") from e_load_strict

            # 尝试加载 MINE 网络
            if 'mine' in checkpoint:
                 try:
                      latent_dim = cfg.model.get('latent_dim', 16) # 确保 latent_dim 正确
                      mine = MINEStrict(latent_dim=latent_dim)
                      mine.load_state_dict(checkpoint['mine'], strict=False)
                      print("  [√] 成功加载严格版本 MINE state_dict。")
                 except Exception as e_mine:
                      print(f"  [!] 警告：加载严格版本 MINE state_dict 失败: {e_mine}。将无法计算 MI。")
                      mine = None
            else:
                 print("  [!] 警告：严格版本检查点中未找到 'mine' state_dict。将无法计算 MI。")
                 mine = None
        else:
            raise ValueError(f"严格版本检查点 '{ckpt_path}' 格式无效，缺少 'model' 键。")
    else:
        # 标准版本模型加载 (LitDeepSC)
        print("尝试加载标准版本模型 (LitDeepSC)...")
        try:
            # 优先使用 load_from_checkpoint (它会处理配置和模型创建)
            lit = LitDeepSC.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location='cpu', strict=False)
            model = lit.model
            mine = lit.mine # 从 LitModule 获取 MINE (可能是 StandardMINE)
            print("  [√] 成功使用 load_from_checkpoint 加载标准版模型。")
        except Exception as e:
            print(f"  [!] 使用 load_from_checkpoint 加载失败: {e}")
            print("  尝试手动创建并加载 state_dict...")
            try:
                # Fallback: 手动加载 state_dict
                lit = LitDeepSC(cfg) # 使用更新后的 cfg 初始化
                if 'state_dict' not in checkpoint:
                    raise ValueError("检查点缺少 'state_dict'")
                # 加载状态，忽略不匹配的键
                missing_keys, unexpected_keys = lit.load_state_dict(checkpoint['state_dict'], strict=False)
                if missing_keys: print(f"  [!] 手动加载时发现缺失键: {missing_keys}")
                if unexpected_keys: print(f"  [!] 手动加载时发现意外键: {unexpected_keys}")
                model = lit.model
                mine = lit.mine
                print("  [√] 成功手动加载标准版 state_dict (strict=False)。")
            except Exception as e2:
                print(f"  [X] 手动加载标准版 state_dict 也失败: {e2}")
                traceback.print_exc()
                raise RuntimeError(f"无法加载标准版模型检查点: {ckpt_path}") from e2

    # 最终检查模型是否加载成功
    if model is None:
        raise RuntimeError(f"在加载检查点 '{ckpt_path}' 后未能成功初始化模型。")

    # 验证加载后的模型词表大小
    try:
         loaded_vocab_size = model.encoder.embed.num_embeddings if hasattr(model, 'encoder') else model.embedding.num_embeddings
         if loaded_vocab_size != actual_vocab_size:
              print(f"  [!] 警告: 最终加载的模型词表大小 ({loaded_vocab_size}) 与当前词表 ({actual_vocab_size}) 不符。结果可能不准确！")
         else:
              print(f"  [√] 模型词表大小 ({loaded_vocab_size}) 与当前词表匹配。")
    except AttributeError:
         print("  [!] 警告：无法自动检查加载模型的词表大小。")


    # 迁移到设备、设置评估模式、冻结参数
    model = model.to(device).eval()
    if mine: mine = mine.to(device).eval()

    for param in model.parameters(): param.requires_grad = False
    if mine:
        for param in mine.parameters(): param.requires_grad = False
    print("  模型已移至设备并设置为评估模式。")

    # 创建信道模型
    try:
        channel = get_channel(cfg.data.channel)().to(device)
        print(f"  使用信道: {cfg.data.channel}")
    except Exception as e_chan:
         raise RuntimeError(f"无法创建信道模型 '{cfg.data.channel}': {e_chan}") from e_chan


    # ---------- 5. 数据加载器 ----------
    print(f"\n--- 准备数据加载器 ---")
    eval_batch_size = cfg.get('eval_batch_size', 64) # 评估时可以使用稍大的批次
    num_workers = cfg.get("num_workers", 4)
    print(f"  评估批大小: {eval_batch_size}")
    print(f"  数据加载 Workers: {num_workers}")
    test_loader = make_dataloader(
        str(val_pkl),
        batch_size=eval_batch_size,
        pad_idx=pad_idx,
        shuffle=False,
        num_workers=num_workers,
    )
    try:
        print(f"  测试集大小: {len(test_loader.dataset)} 句")
    except TypeError:
        print("  测试集大小: 未知 (可能是IterableDataset)")

    # ---------- 6. 评估循环 ----------
    print(f"\n--- 开始评估 ---")
    snrs = cfg.get("eval_snrs", [0, 3, 6, 9, 12, 15, 18])
    print(f"将在以下 SNR (dB) 上评估: {snrs}")
    results_data = {'SNR_dB': [], 'BLEU': [], 'SentenceSimilarity': [], 'MI_LB': []}

    # 设置AMP上下文（使用更新的 API）
    amp_enabled = str(cfg.get('precision', 32)) in ['16', 16, '16-mixed'] and device.type == 'cuda'
    amp_context = autocast(device_type=device.type, enabled=amp_enabled)
    print(f"自动混合精度 (AMP) 评估: {'启用' if amp_enabled else '禁用'}")

    with torch.no_grad():
        for snr_db in snrs:
            print(f"\n-- 评估 SNR = {snr_db} dB --")
            snr_lin = 10.0 ** (snr_db / 10.0)
            # 注意：模型 forward 通常需要噪声方差 sigma^2
            n_var_squared = 1.0 / (2.0 * max(snr_lin, 1e-10))

            bleu_l, sim_l, mi_l = [], [], []
            pbar = tqdm.tqdm(test_loader, desc=f"SNR {snr_db} dB", leave=False, unit="batch")

            for batch in pbar:
                batch = batch.to(device)
                current_batch_size = batch.size(0) # 获取当前批次的实际大小

                try:
                    with amp_context:
                         # Forward pass
                         # 统一接口，总是尝试获取 tx, rx
                         logits, tx, rx = model(batch, n_var_squared, channel, return_tx_rx=True)

                    # Predictions
                    pred = logits.argmax(dim=-1)

                    # BLEU‑1
                    target_indices = batch[:, 1:] # 参考，去掉 <START>
                    pred_len = pred.size(1)
                    target_len = target_indices.size(1)
                    # 对齐长度
                    if pred_len > target_len:
                        pred_aligned = pred[:, :target_len]
                    elif pred_len < target_len:
                        padding = torch.full((current_batch_size, target_len - pred_len), pad_idx, device=device, dtype=torch.long)
                        pred_aligned = torch.cat([pred, padding], dim=1)
                    else:
                        pred_aligned = pred
                    # 计算 BLEU
                    batch_bleu = bleu_score(pred_aligned.cpu(), target_indices.cpu())
                    bleu_l.append(batch_bleu)

                    # Sentence Sim
                    try:
                        # 使用对齐后的预测进行解码，参考也进行同样处理
                        str_pred = [' '.join(vocab.decode(p.tolist(), stop_at_end=True)) for p in pred_aligned.cpu()]
                        str_ref = [' '.join(vocab.decode(r.tolist(), stop_at_end=True)) for r in target_indices.cpu()]
                        # 过滤掉解码/参考后的空字符串对
                        valid_pairs = [(p, r) for p, r in zip(str_pred, str_ref) if p and r]
                        if valid_pairs:
                             str_pred_valid, str_ref_valid = zip(*valid_pairs)
                             batch_sim = sentence_similarity(list(str_pred_valid), list(str_ref_valid), device=device, max_length=cfg.model.get('max_len', 32))
                             sim_l.extend(batch_sim) # batch_sim is a list
                        # 如果过滤后为空，则不添加任何相似度分数，后续 np.mean 会处理空列表
                    except Exception as e_sim:
                        print(f"\n警告: 计算句子相似度时出错 (SNR={snr_db}): {e_sim}")
                        # 不再添加占位符，让 np.mean 处理可能为空的列表

                    # MI Lower Bound (使用 estimate_mi_with_trained_mine)
                    batch_mi = estimate_mi_with_trained_mine(mine, tx, rx)
                    mi_l.append(batch_mi)

                    # 更新进度条 (显示批次平均值)
                    pbar.set_postfix({
                        'BLEU': f"{batch_bleu:.4f}",
                        'Sim': f"{np.mean(batch_sim):.4f}" if valid_pairs else "N/A", # 仅在有有效对时显示
                        'MI': f"{batch_mi:.4f}"
                    })

                except Exception as e_batch:
                     print(f"\n错误：处理批次时出错 (SNR={snr_db}): {e_batch}")
                     traceback.print_exc()
                     # 跳过此批次的指标记录

            # ---- 统计 SNR 的平均指标 ---- #
            avg_bleu = float(np.mean(bleu_l)) if bleu_l else 0.0
            avg_sim = float(np.mean(sim_l)) if sim_l else 0.0 # sim_l 包含每个样本的分数
            avg_mi = float(np.mean(mi_l)) if mi_l else 0.0

            results_data['SNR_dB'].append(snr_db)
            results_data['BLEU'].append(avg_bleu)
            results_data['SentenceSimilarity'].append(avg_sim)
            results_data['MI_LB'].append(avg_mi)
            print(f"  SNR={snr_db} dB 平均结果: BLEU={avg_bleu:.4f}, Sim={avg_sim:.4f}, MI={avg_mi:.4f}")


    # ---------- 7. 保存结果 ----------
    print(f"\n--- 保存评估结果 ---")
    results_dir = Path(cfg.get("eval_output_dir", "evaluation_results")) # 允许配置输出目录
    results_dir.mkdir(parents=True, exist_ok=True)

    # 构建输出文件名
    model_type_str = "strict" if is_strict_version else "standard"
    # 清理检查点名称以用作文件前缀
    output_prefix = ckpt_path.stem.replace('best-','').replace('.ckpt','').replace('.pt','')
    if not output_prefix: # 如果清理后为空
        output_prefix = f"deepsc_{model_type_str}_{cfg.data.channel}"

    # CSV格式保存
    results_df = pd.DataFrame(results_data)

    csv_path = results_dir / f"{output_prefix}_results.csv"
    try:
         results_df.to_csv(csv_path, index=False, float_format='%.5f')
         print(f"结果已保存到 CSV 文件: {csv_path}")
    except Exception as e_csv:
         print(f"错误：保存 CSV 文件失败: {e_csv}")


    # ---------- 8. 绘图 ----------
    print(f"\n--- 生成性能曲线图 ---")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 12)) # 调整画布大小以容纳三个子图

        # 共享 x 轴
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(results_data['SNR_dB'], results_data['BLEU'], 'o-', linewidth=2, label='BLEU-1')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylabel('BLEU-1 Score', fontsize=11)
        plt.title(f'Performance vs. SNR ({model_type_str.capitalize()} Model, Channel: {cfg.data.channel})', fontsize=13)
        plt.legend()
        plt.xticks(snrs)
        plt.ylim(bottom=-0.05, top=1.05)
        ax1.tick_params(labelbottom=False) # 隐藏 x 轴标签

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(results_data['SNR_dB'], results_data['SentenceSimilarity'], 's-', color='orange', linewidth=2, label='Sentence Similarity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylabel('Sentence Similarity', fontsize=11)
        plt.legend()
        plt.xticks(snrs)
        plt.ylim(bottom=-0.05, top=1.05)
        ax2.tick_params(labelbottom=False) # 隐藏 x 轴标签

        # 仅当有有效 MI 数据时绘制
        if results_data['MI_LB'] and any(m != 0.0 for m in results_data['MI_LB']):
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            plt.plot(results_data['SNR_dB'], results_data['MI_LB'], '^-', color='green', linewidth=2, label='MI Lower Bound')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('SNR (dB)', fontsize=11)
            plt.ylabel('Mutual Information (nats)', fontsize=11)
            plt.legend()
            plt.xticks(snrs)
        else:
             print("未绘制互信息曲线（无有效 MI 数据）。")
             # 如果不绘制 MI，则需要让第二个图显示 x 轴标签
             ax2.tick_params(labelbottom=True)
             plt.xlabel('SNR (dB)', fontsize=11) # 在第二个子图下方添加标签


        plt.tight_layout(h_pad=1.5) # 调整子图间距

        plot_path = results_dir / f"{output_prefix}_performance_curves.png"
        plt.savefig(plot_path, dpi=300)
        print(f"性能曲线已保存到: {plot_path}")
        plt.close() # 关闭图形，释放内存

    except Exception as e_plot:
        print(f"错误：生成性能曲线图失败: {e_plot}")
        traceback.print_exc()

    # ---------- 9. 打印结果摘要 ----------
    print("\n=== 评估结果摘要 ===")
    # 使用 to_markdown() 提供更美观的表格输出 (需要 tabulate 库: pip install tabulate)
    try:
        print(results_df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(results_df.to_string(index=False, float_format="%.4f")) # Fallback

    # 性能指标
    if results_data['SentenceSimilarity']: # 确保列表非空
         try:
             best_sim_idx = np.argmax(results_data['SentenceSimilarity'])
             best_snr_sim = results_data['SNR_dB'][best_sim_idx]
             print(f"\n最佳性能点 (基于最高句子相似度 @ SNR = {best_snr_sim} dB):")
             print(f"  • BLEU-1: {results_data['BLEU'][best_sim_idx]:.4f}")
             print(f"  • Sentence Similarity: {results_data['SentenceSimilarity'][best_sim_idx]:.4f}")
             print(f"  • Mutual Information: {results_data['MI_LB'][best_sim_idx]:.4f}")
         except IndexError:
              print("\n无法确定最佳性能点（结果列表索引错误）。")
         except ValueError: # Handle case where list might be empty after all checks
              print("\n无法确定最佳性能点（无有效的句子相似度结果）。")

    # --- 与原论文结果比较 (改进版) ---
    print("\n--- 与论文结果比较 (如果可用) ---")
    paper_results_path = Path("docs/paper_results.json") # 定义文件路径
    if paper_results_path.exists():
        print(f"加载论文结果文件: {paper_results_path}")
        try:
            with open(paper_results_path, 'r', encoding='utf-8') as f:
                paper_results = json.load(f)

            print("SNR (dB) | 实现 BLEU | 论文 BLEU | 差异")
            print("---------|------------|------------|-------")
            comparison_found = False
            for i, snr_db_val in enumerate(results_data['SNR_dB']):
                snr_key = str(snr_db_val) # JSON key 通常是字符串
                if snr_key in paper_results:
                    paper_bleu = paper_results[snr_key].get('bleu') # 假设 JSON 结构是 {"snr": {"bleu": value}}
                    if paper_bleu is not None:
                        impl_bleu = results_data['BLEU'][i]
                        diff = impl_bleu - paper_bleu
                        print(f"{snr_db_val:<8} | {impl_bleu:<10.4f} | {paper_bleu:<10.4f} | {diff:+.4f}")
                        comparison_found = True
                    else:
                        print(f"{snr_db_val:<8} | {results_data['BLEU'][i]:<10.4f} | N/A        | N/A   (JSON 中缺少 'bleu' 键)")
                else:
                    # 如果 JSON 中没有这个 SNR 点，可以选择性打印信息
                    # print(f"{snr_db_val:<8} | {results_data['BLEU'][i]:<10.4f} | N/A        | N/A   (未在 JSON 中找到)")
                    pass # 或者保持安静

            if not comparison_found:
                 print("未在 JSON 文件中找到与当前评估 SNR 匹配的结果。")

        except json.JSONDecodeError:
            print(f"错误：无法解析 JSON 文件 {paper_results_path}。请检查文件格式。")
        except Exception as e_comp:
            print(f"错误：加载或比较论文结果时出错: {e_comp}")
    else:
        print(f"未找到论文结果文件: {paper_results_path}。跳过比较。")
    # --- 比较结束 ---

if __name__ == "__main__":
    # 可以在这里添加一些预检查，例如检查 torch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
         print(f"CUDA 版本 (PyTorch): {torch.version.cuda}")
         print(f"GPU 数量: {torch.cuda.device_count()}")
         print(f"当前 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    main()