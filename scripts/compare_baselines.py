# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC/scripts/compare_baselines.py
# (已修改，处理 RS 解码器的输入，并修正基线评估逻辑)
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
比较脚本：评估DeepSC与传统方法的性能
==================================================
同时评估不同SNR下的BLEU分数、句子相似度和互信息

特性：
1. 自动适应不同词表大小
2. 支持多种信道类型
3. 全面的传统基线比较 (Huffman/Fixed-Length + Turbo/RS)
4. 结果可视化和保存 (CSV, PNG 图表)
5. 修正了 RS 码的评估流程 (使用硬判决字节)
"""
import sys
import torch
import transformers
# --- 运行时诊断信息 ---
# print("--- Runtime Diagnostics ---")
# print(f"Python Executable: {sys.executable}")
# print(f"Torch Version: {torch.__version__}")
# print(f"Torch File Location: {torch.__file__}")
# print(f"Transformers Version: {transformers.__version__}")
# print("sys.path:")
# for p in sys.path:
#     print(f"  - {p}")
# print("--- End Diagnostics ---")

import math, torch, tqdm, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf  # Hydra 配置管理
from hydra.utils import to_absolute_path     # 将相对路径转为绝对路径
import json
import os
from typing import Tuple

from contextlib import nullcontext # 用于创建无操作的上下文管理器
import traceback # 用于打印更详细的错误信息

# --- 数据处理与模型相关 ---
from deepsc.data.europarl import make_dataloader  # 数据加载器
from deepsc.data.vocab import Vocab                # 词表处理
from deepsc.engine.lit_module import LitDeepSC     # PyTorch Lightning 模块
from deepsc.models import get_channel            # 获取信道模型
from deepsc.metrics.bleu import bleu_score         # BLEU 评估
from deepsc.metrics.sentence_sim import sentence_similarity # 句子相似度评估

# --- 导入基线方法 ---
# 尝试导入基线模型，并设置全局标志位指示是否可用
try:
    from deepsc.baselines.traditional import HuffmanEncoder, FixedLengthEncoder # 源编码
    from deepsc.baselines.channel_coding import TurboCoder, RSCoder            # 信道编码
    _GLOBAL_BASELINES_AVAILABLE = True # 全局标志：基线库已安装
    print("信息: 传统基线方法库加载成功。")
except ImportError as e:
    print(f"警告: 传统基线方法库 (deepsc.baselines) 未找到或导入失败: {e}")
    print("     将仅评估 DeepSC 模型，跳过与传统方法的比较。")
    print("     如需比较，请确保 deepsc/baselines/ 目录及其依赖存在且正确。")
    _GLOBAL_BASELINES_AVAILABLE = False # 全局标志：基线库不可用
except Exception as e_other: # 捕获其他可能的导入时错误
    print(f"警告: 加载传统基线方法库时发生意外错误: {e_other}")
    _GLOBAL_BASELINES_AVAILABLE = False


# --- 辅助函数：模拟传统信道 ---
def simulate_traditional_channel_llr(bits: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    """
    模拟传统通信系统中的 BPSK 调制、AWGN 信道和解调（软判决 LLR）。
    适用于需要 LLR 输入的解码器（如 Turbo）。

    参数:
        bits (np.ndarray): 输入比特序列，形状 [batch_size, bit_length]，元素为 0 或 1。
        snr_db (float): 信噪比 (dB)。

    返回:
        Tuple[np.ndarray, float]:
        - LLR (np.ndarray): 解调后的对数似然比 LLR，形状与输入比特相同。
        - noise_var (float): 计算得到的噪声方差 σ^2。
    """
    # 1. 比特映射到 BPSK 符号 (+1 / -1)
    symbols = 2 * bits.astype(np.float32) - 1

    # 2. 计算噪声方差 σ^2 = 1 / (2 * SNR_lin)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * max(snr_lin, 1e-10)) # 噪声方差 σ^2

    # 3. 生成加性高斯白噪声 N ~ N(0, σ^2)
    noise = np.random.normal(0, np.sqrt(noise_var), symbols.shape) # 标准差是 sqrt(方差)

    # 4. 接收信号 y = x + n
    received = symbols + noise

    # 5. 计算软判决 LLR = 2 * y / σ^2
    llr = 2.0 * received / noise_var

    return llr, noise_var

# --- 辅助函数：比特到字节的打包 ---
def pack_bits_to_bytes(bits_array: np.ndarray) -> np.ndarray:
    """
    将 NumPy 比特数组 (0/1) 打包成字节数组 (uint8)。
    在末尾填充 0 比特以确保总比特数是 8 的倍数。

    参数:
        bits_array (np.ndarray): 输入比特数组，形状 [batch_size, bit_length]。

    返回:
        np.ndarray: 打包后的字节数组，形状 [batch_size, byte_length]，dtype=uint8。
                    如果输入为空，则返回形状 [batch_size, 0] 的空数组。
    """
    batch_size, bit_length = bits_array.shape
    if bit_length == 0:
        return np.zeros((batch_size, 0), dtype=np.uint8)

    num_padding_bits = (8 - (bit_length % 8)) % 8
    # 对每一行进行填充和打包
    byte_batches = []
    for i in range(batch_size):
        current_bits = bits_array[i]
        if len(current_bits) == 0: # 处理空行
             byte_batches.append(np.array([], dtype=np.uint8))
             continue

        padded_bits = np.pad(current_bits, (0, num_padding_bits), constant_values=0).astype(np.uint8)
        try:
            packed_bytes = np.packbits(padded_bits)
            byte_batches.append(packed_bytes)
        except ValueError as e:
            print(f"错误: 打包比特到字节时出错 (Batch={i}, len={len(padded_bits)}): {e}")
            byte_batches.append(np.array([], dtype=np.uint8)) # 添加空数组作为错误处理

    # 找到最大字节长度并填充
    if not byte_batches: return np.array([])
    non_empty_bytes = [b for b in byte_batches if len(b) > 0]
    if not non_empty_bytes: return np.zeros((batch_size, 0), dtype=np.uint8)

    max_byte_len = max(len(b) for b in non_empty_bytes)
    padded_bytes_batches = []
    for b in byte_batches:
         pad_width = max(0, max_byte_len - len(b))
         padded = np.pad(b, (0, pad_width), mode='constant', constant_values=0)
         padded_bytes_batches.append(padded)

    return np.array(padded_bytes_batches, dtype=np.uint8)


# --------------------------- 主程序 (Hydra 配置) --------------------------- #
@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    主函数：加载模型，运行评估，比较DeepSC与传统方法在不同SNR下的性能。

    参数:
        cfg (DictConfig): Hydra 加载的配置对象。
    """
    # ---------- 0. 初始化局部基线可用性标志 ----------
    local_baselines_available = _GLOBAL_BASELINES_AVAILABLE

    # ---------- 1. 路径解析与检查 ----------
    print("--- 1. 解析和检查路径 ---")
    ckpt_path  = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl    = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))
    print(f"  检查点路径: {ckpt_path}")
    print(f"  验证数据: {val_pkl}")
    print(f"  词表文件: {vocab_json}")
    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"错误: 未找到文件 '{p}'。请检查配置。")
    print("  所有必需文件均存在。")

    # ---------- 2. 加载词表并更新配置 ----------
    print("\n--- 2. 加载词表和更新配置 ---")
    print(f"  加载词表: {vocab_json}")
    vocab = Vocab.load(vocab_json)
    actual_vocab_size = len(vocab)
    pad_idx = vocab.token2idx.get('<PAD>', 0)
    start_idx = vocab.token2idx.get('<START>', 1) # 获取 START 索引
    end_idx = vocab.token2idx.get('<END>', 2) # 获取 END 索引
    print(f"  实际词表大小: {actual_vocab_size}")
    print(f"  PAD Token 索引: {pad_idx}")
    print(f"  START Token 索引: {start_idx}")
    print(f"  END Token 索引: {end_idx}")
    # 动态更新配置
    cfg.model.vocab_size = actual_vocab_size
    cfg.data.pad_idx = pad_idx
    if 'pad_idx' in cfg.model:
        cfg.model.pad_idx = pad_idx

    # ---------- 3. 设置设备 & 加载 DeepSC 模型 ----------
    print("\n--- 3. 设置设备和加载 DeepSC 模型 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    print(f"  加载模型检查点: {ckpt_path}")
    try:
        # 优先使用 load_from_checkpoint 并传入 cfg
        lit_model = LitDeepSC.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            map_location='cpu', # 先加载到 CPU，稍后移动
            cfg=cfg,
            strict=False # 增加对旧检查点的兼容性
        )
        print("  成功使用 load_from_checkpoint 加载模型。")
    except Exception as e1:
        print(f"  使用 load_from_checkpoint 加载失败: {e1}")
        print("  尝试手动创建模型并加载 state_dict...")
        try:
            # Fallback: 手动创建 LitDeepSC 实例并加载 state_dict
            lit_model = LitDeepSC(cfg) # 使用当前配置创建模型
            checkpoint = torch.load(ckpt_path, map_location='cpu') # 加载检查点
            if 'state_dict' not in checkpoint:
                 raise ValueError(f"检查点 '{ckpt_path}' 格式无效，缺少 'state_dict'。")

            # 加载状态字典，允许部分不匹配 (strict=False)
            lit_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  成功手动加载 state_dict (strict=False)。")

            # 检查超参数是否匹配 (可选，但建议)
            if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
                loaded_cfg = checkpoint['hyper_parameters']['cfg']
                # 比较关键参数，例如模型结构参数
                # ... (此处可添加更多检查逻辑)
                if loaded_cfg.model.vocab_size != cfg.model.vocab_size:
                     print(f"  警告: 检查点配置的 vocab_size ({loaded_cfg.model.vocab_size}) 与当前 ({cfg.model.vocab_size}) 不符。")
                     # 可以决定是使用加载的配置还是当前配置
                     # lit_model = LitDeepSC(loaded_cfg) # 切换到使用加载的配置
                     # lit_model.load_state_dict(checkpoint['state_dict'], strict=False)

        except Exception as e2:
            print(f"  手动加载 state_dict 也失败: {e2}")
            raise RuntimeError(f"无法加载模型检查点: {ckpt_path}") from e2

    # 验证词表大小一致性
    loaded_vocab_size = lit_model.model.encoder.embed.num_embeddings
    if loaded_vocab_size != actual_vocab_size:
        print(f"  警告: 最终加载的模型词表大小 ({loaded_vocab_size}) 与当前词表 ({actual_vocab_size}) 不符。")
        print("       评估可能不准确。将使用模型内部的词表大小进行后续操作。")
        # 更新配置以匹配实际加载的模型
        cfg.model.vocab_size = loaded_vocab_size
        if hasattr(cfg.data, 'vocab_size'): # 如果数据配置中也有vocab_size，一并更新
            cfg.data.vocab_size = loaded_vocab_size

    # 模型转移至设备并设置为评估模式
    lit_model = lit_model.to(device).eval()
    for param in lit_model.parameters(): # 冻结参数
        param.requires_grad = False
    print(f"  模型已移至 {device} 并设置为评估模式。")
    print(f"  使用的信道模型 (DeepSC): {lit_model.channel.__class__.__name__}")

    # ---------- 4. 加载评估数据 ----------
    print("\n--- 4. 加载评估数据集 ---")
    print(f"  加载验证集数据: {val_pkl}")
    # 使用较小的评估批大小以减少内存占用，特别是对于相似度计算
    eval_batch_size = cfg.get('eval_batch_size', 32) # 从配置获取，默认为 32
    print(f"  评估批大小: {eval_batch_size}")
    num_available_cpus = os.cpu_count()
    num_workers = min(num_available_cpus if num_available_cpus is not None else 1, 4)
    print(f"  数据加载器 Workers: {num_workers}")
    test_loader = make_dataloader(
        str(val_pkl),
        batch_size = eval_batch_size,
        pad_idx    = pad_idx,
        shuffle    = False,
        num_workers = num_workers,
    )
    try:
        dataset_len = len(test_loader.dataset)
        print(f"  测试集样本数量: {dataset_len} 句")
    except TypeError:
        print("  测试集样本数量: 未知 (IterableDataset)")

    # ---------- 5. 初始化传统基线方法 ----------
    print("\n--- 5. 初始化传统基线方法 ---")
    baselines = {}
    baseline_methods_to_compare = []
    if local_baselines_available:
        print("  尝试初始化传统源编码和信道编码器...")
        try:
            # 源编码器
            baselines['huffman'] = HuffmanEncoder(cfg.model.vocab_size)
            # 计算固定长度编码所需的最小比特数
            bits_per_token_fixed = int(np.ceil(np.log2(cfg.model.vocab_size)))
            baselines['fixed'] = FixedLengthEncoder(cfg.model.vocab_size, bits_per_token=bits_per_token_fixed)
            print(f"    FixedLengthEncoder 使用 bits_per_token={bits_per_token_fixed}")

            # 信道编码器
            baselines['turbo'] = TurboCoder(rate=1/3, iterations=5) # 假设码率1/3
            baselines['rs'] = RSCoder(n=255, k=223) # 常用 RS(255, 223)

            baseline_methods_to_compare = [
                'huffman_turbo', 'fixed_turbo', 'huffman_rs', 'fixed_rs'
            ]
            print(f"  成功初始化基线方法: {', '.join(baselines.keys())}")
            print(f"  将比较以下组合: {', '.join(baseline_methods_to_compare)}")
        except Exception as e:
            print(f"  错误: 初始化传统基线方法时发生异常: {e}")
            traceback.print_exc()
            print("       将跳过与传统方法的比较。")
            local_baselines_available = False
    else:
        print("  跳过初始化传统基线方法（库未导入或导入失败）。")

    # ---------- 6. 设置评估参数和结果存储 ----------
    print("\n--- 6. 设置评估参数和结果存储 ---")
    snrs_db = cfg.get("eval_snrs", [0, 3, 6, 9, 12, 15, 18])
    print(f"  将评估以下 SNR (dB) 点: {snrs_db}")
    results = {'snr': snrs_db}
    results['deepsc_bleu'] = []
    results['deepsc_sim'] = []
    results['deepsc_mi'] = []
    if local_baselines_available:
        for method in baseline_methods_to_compare:
            results[f'{method}_bleu'] = []
            results[f'{method}_sim'] = []
    print("  结果存储结构已初始化。")
    # 设置 AMP 上下文
    amp_enabled = (str(cfg.get('precision', '32')) in ['16', '16-mixed', 16]) and device.type == 'cuda'
    if amp_enabled:
        print("  启用自动混合精度 (AMP) 进行评估。")
        amp_ctx = torch.cuda.amp.autocast()
    else:
        print("  不使用自动混合精度评估。")
        amp_ctx = nullcontext() # 使用无操作上下文

    # ---------- 7. 循环评估不同 SNR ----------
    print("\n--- 7. 开始在不同 SNR 下评估性能 ---")
    # 限制评估基线的批次数，因为它们通常比 DeepSC 慢很多
    baseline_eval_max_batches = cfg.get("baseline_eval_max_batches", 5) # 默认只评估前5个批次
    print(f"  传统基线将只评估前 {baseline_eval_max_batches} 个批次以节省时间。")

    for snr_db in snrs_db:
        print(f"\n  评估 SNR = {snr_db} dB")
        # DeepSC 使用的噪声方差 (与 LitModule 中 _sample_noise 逻辑一致)
        snr_lin_deepsc = 10.0 ** (snr_db / 10.0)
        deepsc_noise_var = 1.0 / (2.0 * max(snr_lin_deepsc, 1e-10)) # 这是噪声标准差 sigma 的平方

        # 初始化当前 SNR 的临时结果列表/字典
        current_snr_deepsc_bleu, current_snr_deepsc_sim, current_snr_deepsc_mi = [], [], []
        current_snr_baseline_results = {}
        if local_baselines_available:
            current_snr_baseline_results = {method: {'bleu': [], 'sim': [], 'count': 0} for method in baseline_methods_to_compare}

        # 使用 enumerate 获取批次索引
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc=f"  SNR {snr_db} dB", leave=False, unit="batch")

        for batch_idx, batch in pbar:
            # --- 7.1 评估 DeepSC ---
            with torch.no_grad(), amp_ctx:
                batch_tensor = batch.to(device)
                # 确保模型和 MINE 在正确的设备上 (理论上 load_from_checkpoint 后 to(device) 即可)
                # lit_model.model = lit_model.model.to(device)
                # lit_model.channel = lit_model.channel.to(device)
                # lit_model.mine = lit_model.mine.to(device)

                # DeepSC Forward Pass
                try:
                    logits, tx, rx = lit_model.model(batch_tensor, deepsc_noise_var, lit_model.channel, return_tx_rx=True)
                    pred_indices = logits.argmax(dim=-1)
                except Exception as e_deepsc:
                     print(f"\n错误: DeepSC 前向传播失败 (SNR={snr_db}, Batch={batch_idx}): {e_deepsc}")
                     traceback.print_exc()
                     # 跳过此批次的 DeepSC 评估
                     continue


                # BLEU Score
                target_indices = batch_tensor[:, 1:] # 参考序列，去除 <START>
                pred_len = pred_indices.size(1)
                target_len = target_indices.size(1)

                # 对齐预测和目标长度 (预测长度通常是 target_len)
                if pred_len > target_len:
                    pred_indices_aligned = pred_indices[:, :target_len]
                elif pred_len < target_len:
                    padding = torch.full((pred_indices.size(0), target_len - pred_len), pad_idx, device=device, dtype=torch.long)
                    pred_indices_aligned = torch.cat([pred_indices, padding], dim=1)
                else:
                    pred_indices_aligned = pred_indices

                # 计算 BLEU 分数 (需要 CPU 张量)
                batch_bleu = bleu_score(pred_indices_aligned.cpu(), target_indices.cpu())
                current_snr_deepsc_bleu.append(batch_bleu)

                # Sentence Similarity
                # 解码为字符串，忽略 <PAD>, <START>, 遇到 <END> 停止
                str_pred_list, str_ref_list = [], []
                for p_idx, r_idx in zip(pred_indices_aligned.cpu(), target_indices.cpu()):
                     # 解码预测，不包含<START>，到<END>停止
                     str_pred_list.append(' '.join(vocab.decode(p_idx.tolist(), stop_at_end=True)))
                     # 解码参考，已经去除了<START>，到<END>停止
                     str_ref_list.append(' '.join(vocab.decode(r_idx.tolist(), stop_at_end=True)))

                try:
                    # 传入max_length参数，与模型配置一致
                    batch_sim = sentence_similarity(str_pred_list, str_ref_list, device=device, max_length=cfg.model.max_len)
                    current_snr_deepsc_sim.extend(batch_sim) # batch_sim 是列表
                except Exception as sim_err:
                    print(f"\n警告: 计算 DeepSC 句子相似度时出错 (SNR={snr_db}, Batch={batch_idx}): {sim_err}")
                    batch_sim = [0.0] * len(str_pred_list) # 出错时填 0
                    current_snr_deepsc_sim.extend(batch_sim)

                # Mutual Information (使用模型自带的 MINE)
                tx_flat = tx.reshape(-1, tx.size(-1))
                rx_flat = rx.reshape(-1, rx.size(-1))
                if tx_flat.size(0) > 0: # 确保不为空
                     try:
                         batch_mi = lit_model.mine(tx_flat, rx_flat).item()
                     except Exception as mi_err:
                         print(f"\n警告: 计算 DeepSC 互信息时出错 (SNR={snr_db}, Batch={batch_idx}): {mi_err}")
                         batch_mi = 0.0 # 出错时填 0
                else:
                     batch_mi = 0.0 # 如果输入为空，MI 为 0
                current_snr_deepsc_mi.append(batch_mi)


            # --- 7.2 评估传统基线 (仅评估前 N 个批次) ---
            if local_baselines_available and batch_idx < baseline_eval_max_batches:
                batch_np = batch.numpy() # 使用原始 batch (包含 <START>)

                for method_name in baseline_methods_to_compare:
                    source_coder_name, channel_coder_name = method_name.split('_')
                    source_coder = baselines[source_coder_name]
                    channel_coder = baselines[channel_coder_name]

                    try:
                        # 1. 源编码 (输入是原始句子，包含 <START>)
                        #    返回: (比特数组, 每个句子的比特长度列表)
                        source_encoded_bits, original_source_bit_lengths = source_coder.encode(batch_np)

                        if source_encoded_bits.size == 0:
                            # print(f"\n信息: 源编码器 '{source_coder_name}' 在 Batch={batch_idx} 产生空输出，跳过基线 '{method_name}'。")
                            continue

                        # 2. 信道编码
                        #    返回: Turbo -> 比特; RS -> 符号(字节)
                        channel_coded_output = channel_coder.encode(source_encoded_bits)
                        if channel_coded_output.size == 0:
                            # print(f"\n信息: 信道编码器 '{channel_coder_name}' 在 Batch={batch_idx} 产生空输出，跳过基线 '{method_name}'。")
                            continue

                        # --- 3. 模拟信道 & 4. 信道解码 ---
                        if channel_coder_name == 'turbo':
                            # Turbo 需要 LLR 输入
                            llr, noise_var_turbo = simulate_traditional_channel_llr(channel_coded_output, snr_db)
                            # Turbo 解码器需要 noise_var (虽然当前简化版不用)
                            channel_decoded_bits = channel_coder.decode(llr, noise_var=noise_var_turbo)
                        elif channel_coder_name == 'rs':
                            # RS 需要硬判决符号 (字节) 输入
                            # 先通过 LLR 通道模拟
                            llr, _ = simulate_traditional_channel_llr(channel_coded_output, snr_db) # RS编码输出是符号，但通道作用在调制后的信号上，此处简化为作用在bit上
                            # 进行硬判决 (LLR -> bits)
                            hard_bits = (llr < 0).astype(np.uint8) # LLR<0 -> 1, LLR>=0 -> 0

                            # 将硬判决比特打包成字节 (模拟解调后的字节流)
                            received_symbols = pack_bits_to_bytes(hard_bits)

                            # RS 解码器需要原始比特长度信息以正确处理解码后的比特流截断
                            # 注意：这里需要的是源编码后的原始比特长度，而不是源输入长度
                            # channel_coder.encode 中已经保存了原始信息比特长度 (_current_original_bit_lengths)
                            # decode 方法内部会使用这个保存的长度
                            # 我们需要将 encode 保存的长度传递给 decode
                            # 这里假设 RSCoder.decode 内部能访问到 self._current_original_bit_lengths
                            # 这是 RSCoder 实现的一部分（见上面修改后的 RSCoder）
                            channel_decoded_bits = channel_coder.decode(received_symbols) # 不需要 noise_var

                        else:
                            print(f"警告: 未知的信道编码器类型 '{channel_coder_name}'，无法进行解码。")
                            continue

                        if channel_decoded_bits.size == 0:
                            # print(f"\n信息: 信道解码器 '{channel_coder_name}' 在 Batch={batch_idx} 产生空输出，跳过基线 '{method_name}'。")
                            continue

                        # 5. 源解码
                        #    需要传入源编码后的比特长度列表
                        decoded_sentences_indices = source_coder.decode(channel_decoded_bits, original_source_bit_lengths)

                        # 6. 计算指标
                        batch_baseline_bleu = []
                        batch_baseline_sim_list = [] # 存储每个样本的 sim 分数
                        for i, (decoded_indices, ref_indices_np) in enumerate(zip(decoded_sentences_indices, batch_np)):
                            # decoded_indices 是一个句子的索引列表 [idx1, idx2, ...]
                            # ref_indices_np 是一整个原始句子 (含 <START>)
                            if not decoded_indices: continue # 跳过空解码结果

                            # BLEU: 比较解码结果和目标（去除 <START>）
                            # 将解码结果和参考转换为 Tensor 进行比较
                            dec_tensor = torch.tensor([decoded_indices], dtype=torch.long)
                            # 参考序列去掉 <START>
                            ref_tensor = torch.from_numpy(np.array([ref_indices_np[1:]])).long()

                            # 对齐解码和参考序列长度
                            pred_len_b = dec_tensor.size(1)
                            target_len_b = ref_tensor.size(1)
                            if pred_len_b == 0 or target_len_b == 0: continue # 跳过空序列

                            if pred_len_b > target_len_b:
                                dec_tensor_aligned = dec_tensor[:, :target_len_b]
                            elif pred_len_b < target_len_b:
                                padding_b = torch.full((1, target_len_b - pred_len_b), pad_idx, dtype=torch.long)
                                dec_tensor_aligned = torch.cat([dec_tensor, padding_b], dim=1)
                            else:
                                dec_tensor_aligned = dec_tensor

                            batch_baseline_bleu.append(bleu_score(dec_tensor_aligned, ref_tensor))

                            # Sentence Similarity
                            # 解码时忽略 <PAD>, <START>, 遇到 <END> 停止
                            dec_str = ' '.join(vocab.decode(decoded_indices, stop_at_end=True)) # 基线解码结果通常不含 <START>
                            ref_str = ' '.join(vocab.decode(ref_indices_np[1:].tolist(), stop_at_end=True)) # 参考去掉 <START>
                            if not dec_str or not ref_str: continue # 跳过空字符串

                            try:
                                sim_score = sentence_similarity([dec_str], [ref_str], device=device, max_length=cfg.model.max_len)
                                batch_baseline_sim_list.extend(sim_score) # sim_score is a list with one element
                            except Exception as sim_err_b:
                                print(f"\n警告: 计算基线 '{method_name}' 句子相似度时出错 (SNR={snr_db}, Batch={batch_idx}, Idx={i}): {sim_err_b}")
                                batch_baseline_sim_list.append(0.0) # 添加占位符

                        # 累加当前批次的平均 BLEU 和所有样本的 Sim 分数
                        if batch_baseline_bleu:
                            current_snr_baseline_results[method_name]['bleu'].append(np.mean(batch_baseline_bleu))
                        if batch_baseline_sim_list:
                            current_snr_baseline_results[method_name]['sim'].extend(batch_baseline_sim_list)
                        current_snr_baseline_results[method_name]['count'] += 1 # 记录成功评估的批次数


                    except Exception as e_baseline:
                        print(f"\n警告: 在 SNR={snr_db} dB, Batch={batch_idx} 评估基线 '{method_name}' 时出错: {e_baseline}")
                        traceback.print_exc() # 打印详细错误堆栈


            # --- 7.3 更新进度条 ---
            pbar_metrics = {
                'BLEU': f"{batch_bleu:.3f}",
                'Sim': f"{np.mean(batch_sim):.3f}" if batch_sim else "N/A",
                'MI': f"{batch_mi:.3f}"
            }
            if local_baselines_available and batch_idx < baseline_eval_max_batches:
                 # 显示第一个基线的 BLEU 作为参考
                 first_baseline = baseline_methods_to_compare[0]
                 if current_snr_baseline_results[first_baseline]['bleu']:
                      pbar_metrics[f"{first_baseline[:3]}BLEU"] = f"{current_snr_baseline_results[first_baseline]['bleu'][-1]:.3f}"

            pbar.set_postfix(pbar_metrics)

        # --- 单个 SNR 评估结束 ---
        pbar.close() # 关闭当前 SNR 的进度条

        # 计算并记录 DeepSC 平均指标
        avg_deepsc_bleu = np.mean(current_snr_deepsc_bleu) if current_snr_deepsc_bleu else 0.0
        avg_deepsc_sim = np.mean(current_snr_deepsc_sim) if current_snr_deepsc_sim else 0.0
        avg_deepsc_mi = np.mean(current_snr_deepsc_mi) if current_snr_deepsc_mi else 0.0
        results['deepsc_bleu'].append(avg_deepsc_bleu)
        results['deepsc_sim'].append(avg_deepsc_sim)
        results['deepsc_mi'].append(avg_deepsc_mi)

        # 计算并记录基线平均指标
        if local_baselines_available:
            for method in baseline_methods_to_compare:
                # BLEU: 对每个批次的平均 BLEU 再求平均
                avg_baseline_bleu = np.mean(current_snr_baseline_results[method]['bleu']) if current_snr_baseline_results[method]['bleu'] else 0.0
                # Sim: 对所有样本的 Sim 分数求平均
                avg_baseline_sim = np.mean(current_snr_baseline_results[method]['sim']) if current_snr_baseline_results[method]['sim'] else 0.0
                results[f'{method}_bleu'].append(avg_baseline_bleu)
                results[f'{method}_sim'].append(avg_baseline_sim)
                # 如果 count 为 0，表示该基线在此 SNR 下未能成功评估
                if current_snr_baseline_results[method]['count'] == 0:
                     print(f"  信息: 基线 '{method}' 在 SNR={snr_db} dB 下未能成功评估任何批次。")


        # 打印当前 SNR 汇总
        print(f"  SNR={snr_db} dB 汇总:")
        print(f"    DeepSC: BLEU={avg_deepsc_bleu:.4f}, Sim={avg_deepsc_sim:.4f}, MI={avg_deepsc_mi:.4f}")
        if local_baselines_available:
            for method in baseline_methods_to_compare:
                bleu_val = results[f'{method}_bleu'][-1]
                sim_val = results[f'{method}_sim'][-1]
                # 只有当成功评估了至少一个批次时才打印结果
                if current_snr_baseline_results[method]['count'] > 0:
                    print(f"    {method.replace('_','+').upper()}: BLEU={bleu_val:.4f}, Sim={sim_val:.4f} (基于 {current_snr_baseline_results[method]['count']} 批次)")
                else:
                     print(f"    {method.replace('_','+').upper()}: (未成功评估)")


    # ---------- 8. 保存结果与可视化 ----------
    print("\n--- 8. 保存结果并生成图表 ---")
    # 使用 Hydra 的输出目录，如果可用
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) if hydra.core.hydra_config.HydraConfig.initialized() else Path('evaluation_results_compare')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  结果将保存到: {output_dir.resolve()}")

    # 保存 CSV
    results_df = pd.DataFrame(results)
    # 使用检查点名称（不含扩展名）作为文件前缀
    output_prefix = ckpt_path.stem.replace('best-','').replace('.ckpt','') # 清理文件名
    csv_filename = output_dir / f'{output_prefix}_baseline_comparison.csv'
    try:
        results_df.to_csv(csv_filename, index=False, float_format='%.5f')
        print(f"  详细结果已保存到 CSV 文件: {csv_filename}")
    except Exception as e_csv:
        print(f"  错误: 保存 CSV 文件失败: {e_csv}")

    # --- 绘图 ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid') # 使用兼容的样式
        plot_channel_name = lit_model.channel.__class__.__name__ # 获取信道名称用于标题

        # 绘制 BLEU 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(results['snr'], results['deepsc_bleu'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC')
        if local_baselines_available:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(baseline_methods_to_compare)))
            markers = ['s', '^', 'x', 'd'] # 为4个基线定义不同标记
            for i, method in enumerate(baseline_methods_to_compare):
                # 仅当该方法有有效数据点时才绘制
                if any(results[f'{method}_bleu']):
                    plt.plot(results['snr'], results[f'{method}_bleu'],
                             marker=markers[i % len(markers)], linestyle='--', linewidth=1.5, markersize=5,
                             color=colors[i], label=method.replace('_', '+').upper())
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('BLEU-1 Score', fontsize=12)
        plt.title(f'BLEU Score vs. SNR Comparison ({plot_channel_name})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.6) # 使用 both major and minor ticks
        plt.legend(fontsize=10)
        plt.xticks(snrs_db)
        plt.ylim(bottom=0) # BLEU 最低为 0
        plt.tight_layout()
        bleu_plot_filename = output_dir / f'{output_prefix}_bleu_comparison.png'
        plt.savefig(bleu_plot_filename, dpi=300, bbox_inches='tight')
        print(f"  BLEU 比较图已保存: {bleu_plot_filename}")
        plt.close()

        # 绘制句子相似度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(results['snr'], results['deepsc_sim'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC')
        if local_baselines_available:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(baseline_methods_to_compare)))
            markers = ['s', '^', 'x', 'd']
            for i, method in enumerate(baseline_methods_to_compare):
                 if any(results[f'{method}_sim']):
                    plt.plot(results['snr'], results[f'{method}_sim'],
                             marker=markers[i % len(markers)], linestyle='--', linewidth=1.5, markersize=5,
                             color=colors[i], label=method.replace('_', '+').upper())
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Sentence Similarity', fontsize=12)
        plt.title(f'Sentence Similarity vs. SNR Comparison ({plot_channel_name})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.xticks(snrs_db)
        plt.ylim(bottom=0, top=1.05) # 相似度范围 [0, 1]
        plt.tight_layout()
        sim_plot_filename = output_dir / f'{output_prefix}_similarity_comparison.png'
        plt.savefig(sim_plot_filename, dpi=300, bbox_inches='tight')
        print(f"  句子相似度比较图已保存: {sim_plot_filename}")
        plt.close()

        # 绘制互信息曲线 (仅 DeepSC)
        plt.figure(figsize=(10, 6))
        plt.plot(results['snr'], results['deepsc_mi'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC MI Lower Bound', color='purple')
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Mutual Information Lower Bound (nats)', fontsize=12)
        plt.title(f'DeepSC Mutual Information vs. SNR ({plot_channel_name})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.xticks(snrs_db)
        plt.tight_layout()
        mi_plot_filename = output_dir / f'{output_prefix}_mi_curve.png'
        plt.savefig(mi_plot_filename, dpi=300, bbox_inches='tight')
        print(f"  DeepSC 互信息曲线图已保存: {mi_plot_filename}")
        plt.close()

    except Exception as e_plot:
        print(f"  错误: 生成图表时失败: {e_plot}")


    print(f"\n评估完成！结果已保存到目录: {output_dir.resolve()}")

# ---------- 程序入口点 ----------
if __name__ == '__main__':
    # 可以在这里添加一些环境检查或预处理
    main()