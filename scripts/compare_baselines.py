# scripts/compare_baselines.py - 完整的基线比较脚本
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较脚本：评估DeepSC与传统方法的性能
==================================================
同时评估不同SNR下的BLEU分数、句子相似度和互信息
"""
import math, torch, tqdm, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity

# 导入基线方法
from deepsc.baselines.traditional import HuffmanEncoder, FixedLengthEncoder, BrotliEncoder
from deepsc.baselines.channel_coding import TurboCoder, RSCoder
from deepsc.models.jscc import JSCC

def bits_to_symbols(bits: np.ndarray, modulation='qam', order=64):
    """
    将比特映射到调制符号
    
    参数:
        bits: 比特数组 [batch, bit_length]
        modulation: 调制方式，'qam'或'psk'
        order: 调制阶数，如64表示64-QAM
        
    返回:
        复数符号 [batch, symbol_length]
    """
    batch_size = bits.shape[0]
    bits_per_symbol = int(np.log2(order))
    
    symbols_list = []
    for i in range(batch_size):
        # 将比特分组
        data = bits[i]
        padding = (bits_per_symbol - len(data) % bits_per_symbol) % bits_per_symbol
        if padding:
            data = np.pad(data, (0, padding))
        
        # 将比特组转换为整数索引
        indices = np.zeros(len(data) // bits_per_symbol, dtype=np.int)
        for j in range(len(indices)):
            for k in range(bits_per_symbol):
                if j * bits_per_symbol + k < len(data):
                    indices[j] |= data[j * bits_per_symbol + k] << (bits_per_symbol - 1 - k)
        
        # 映射到复数符号
        if modulation == 'qam':
            # 使用均匀QAM星座
            symbols = np.zeros(len(indices), dtype=np.complex)
            sqrt_order = int(np.sqrt(order))
            for j, idx in enumerate(indices):
                i_idx = idx % sqrt_order
                q_idx = idx // sqrt_order
                i_val = 2 * i_idx - (sqrt_order - 1)
                q_val = 2 * q_idx - (sqrt_order - 1)
                symbols[j] = complex(i_val, q_val) / np.sqrt(order - 1)
        else:  # PSK
            # 使用PSK星座
            symbols = np.zeros(len(indices), dtype=np.complex)
            for j, idx in enumerate(indices):
                angle = 2 * np.pi * idx / order
                symbols[j] = complex(np.cos(angle), np.sin(angle))
        
        symbols_list.append(symbols)
    
    # 找出最长符号序列，将所有序列填充到相同长度
    max_len = max(len(s) for s in symbols_list)
    padded_symbols = []
    for s in symbols_list:
        padded = np.pad(s, (0, max_len - len(s)), 'constant', constant_values=0)
        padded_symbols.append(padded)
    
    return np.array(padded_symbols)

def transmit_over_channel(symbols: np.ndarray, snr_db: float, channel_type='awgn'):
    """
    通过指定信道传输符号
    
    参数:
        symbols: 复数符号 [batch, symbol_length]
        snr_db: 信噪比(dB)
        channel_type: 信道类型，'awgn'或'rayleigh'
        
    返回:
        接收到的符号 [batch, symbol_length]
    """
    batch_size, symbol_length = symbols.shape
    
    # 计算噪声方差
    snr_lin = 10 ** (snr_db / 10)
    noise_var = 1 / (2 * snr_lin)
    
    # 生成噪声
    noise_real = np.random.normal(0, np.sqrt(noise_var), (batch_size, symbol_length))
    noise_imag = np.random.normal(0, np.sqrt(noise_var), (batch_size, symbol_length))
    noise = noise_real + 1j * noise_imag
    
    if channel_type == 'awgn':
        # AWGN信道
        rx_symbols = symbols + noise
    elif channel_type == 'rayleigh':
        # Rayleigh衰落信道
        h_real = np.random.normal(0, 1/np.sqrt(2), (batch_size, symbol_length))
        h_imag = np.random.normal(0, 1/np.sqrt(2), (batch_size, symbol_length))
        h = h_real + 1j * h_imag
        
        rx_symbols = h * symbols + noise
        
        # 理想信道估计
        rx_symbols = rx_symbols / h
    else:
        raise ValueError(f"未知信道类型: {channel_type}")
    
    return rx_symbols

def symbols_to_llr(rx_symbols: np.ndarray, noise_var: float, modulation='qam', order=64):
    """
    将接收符号转换为LLR(对数似然比)
    
    参数:
        rx_symbols: 接收到的复数符号 [batch, symbol_length]
        noise_var: 噪声方差
        modulation: 调制方式
        order: 调制阶数
        
    返回:
        软判决LLR [batch, bit_length]
    """
    batch_size, symbol_length = rx_symbols.shape
    bits_per_symbol = int(np.log2(order))
    
    llr_list = []
    for i in range(batch_size):
        symbols = rx_symbols[i]
        llr = np.zeros(symbol_length * bits_per_symbol)
        
        if modulation == 'qam':
            # 为QAM计算软判决LLR
            sqrt_order = int(np.sqrt(order))
            for j, s in enumerate(symbols):
                for k in range(bits_per_symbol):
                    # 对每个比特位置，计算该比特为0和为1时的概率
                    prob_0 = 0
                    prob_1 = 0
                    
                    # 遍历星座中的所有符号
                    for idx in range(order):
                        i_idx = idx % sqrt_order
                        q_idx = idx // sqrt_order
                        i_val = 2 * i_idx - (sqrt_order - 1)
                        q_val = 2 * q_idx - (sqrt_order - 1)
                        symbol = complex(i_val, q_val) / np.sqrt(order - 1)
                        
                        # 计算接收符号与星座点之间的欧氏距离
                        distance = abs(s - symbol) ** 2
                        likelihood = np.exp(-distance / noise_var)
                        
                        # 检查该星座点对应的第k个比特是0还是1
                        bit_k = (idx >> (bits_per_symbol - 1 - k)) & 1
                        if bit_k == 0:
                            prob_0 += likelihood
                        else:
                            prob_1 += likelihood
                    
                    # 计算LLR
                    if prob_1 > 0 and prob_0 > 0:
                        llr[j * bits_per_symbol + k] = np.log(prob_0 / prob_1)
                    else:
                        # 处理数值溢出
                        llr[j * bits_per_symbol + k] = 0
        else:  # PSK
            # 为PSK计算软判决LLR
            for j, s in enumerate(symbols):
                for k in range(bits_per_symbol):
                    prob_0 = 0
                    prob_1 = 0
                    
                    for idx in range(order):
                        angle = 2 * np.pi * idx / order
                        symbol = complex(np.cos(angle), np.sin(angle))
                        
                        distance = abs(s - symbol) ** 2
                        likelihood = np.exp(-distance / noise_var)
                        
                        bit_k = (idx >> (bits_per_symbol - 1 - k)) & 1
                        if bit_k == 0:
                            prob_0 += likelihood
                        else:
                            prob_1 += likelihood
                    
                    if prob_1 > 0 and prob_0 > 0:
                        llr[j * bits_per_symbol + k] = np.log(prob_0 / prob_1)
                    else:
                        llr[j * bits_per_symbol + k] = 0
        
        llr_list.append(llr)
    
    # 找出最长LLR序列，将所有序列填充到相同长度
    max_len = max(len(l) for l in llr_list)
    padded_llr = []
    for l in llr_list:
        padded = np.pad(l, (0, max_len - len(l)))
        padded_llr.append(padded)
    
    return np.array(padded_llr)

def llr_to_bits(llr: np.ndarray):
    """
    将LLR硬判决为比特
    
    参数:
        llr: LLR值 [batch, bit_length]
        
    返回:
        硬判决后的比特 [batch, bit_length]
    """
    return (llr < 0).astype(np.int)

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """主程序：评估不同方法在各种SNR下的性能"""
    # ---------- 1. 路径解析 ----------
    ckpt_path  = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl    = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))

    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"未找到文件：{p}")

    # ---------- 2. 加载模型和数据 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载DeepSC模型
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    lit = LitDeepSC(cfg)
    lit.load_state_dict(checkpoint['state_dict'])
    lit = lit.to(device).eval()
    
    # 加载JSCC模型
    jscc_cfg = {
        'vocab_size': cfg.model.vocab_size,
        'pad_idx': cfg.data.pad_idx,
        'embed_dim': 256,
        'hidden_dim': 512,
        'latent_dim': cfg.model.latent_dim,
        'lstm_layers': 2,
        'dropout': 0.1
    }
    jscc_model = JSCC(jscc_cfg).to(device)
    
    # 加载词表和数据
    vocab = Vocab.load(vocab_json)
    test_loader = make_dataloader(
        val_pkl,
        batch_size = cfg.train.batch_size,
        pad_idx    = vocab.token2idx['<PAD>'],
        shuffle    = False,
        num_workers = 4,
    )

    # ---------- 3. 创建传统编码基线 ----------
    # 源编码器
    huffman_encoder = HuffmanEncoder(cfg.model.vocab_size)
    fixed_encoder = FixedLengthEncoder(cfg.model.vocab_size, bits_per_token=5)
    brotli_encoder = BrotliEncoder(quality=11)
    
    # 信道编码器
    turbo_coder = TurboCoder(rate=1/3, iterations=5)
    rs_coder = RSCoder(n=255, k=223)
    
    # 调制方案
    modulations = {
        'huffman_turbo': {'mod': 'qam', 'order': 64},
        'fixed_turbo': {'mod': 'qam', 'order': 128},
        'huffman_rs': {'mod': 'qam', 'order': 64},
        'fixed_rs': {'mod': 'qam', 'order': 64},
        'brotli_turbo': {'mod': 'qam', 'order': 8}
    }

    # ---------- 4. 评估不同SNR下的性能 ----------
    snrs = [0, 3, 6, 9, 12, 15, 18]
    results = {
        'deepsc_bleu': [],
        'deepsc_sim': [],
        'deepsc_mi': [],
        'jscc_bleu': [],
        'jscc_sim': [],
        'huffman_turbo_bleu': [],
        'huffman_turbo_sim': [],
        'fixed_turbo_bleu': [],
        'fixed_turbo_sim': [],
        'huffman_rs_bleu': [],
        'huffman_rs_sim': [],
        'fixed_rs_bleu': [],
        'fixed_rs_sim': [],
        'brotli_turbo_bleu': [],
        'brotli_turbo_sim': []
    }

    # 循环每个SNR
    for snr_db in snrs:
        print(f"\n评估 SNR = {snr_db} dB")
        
        # 计算噪声方差
        snr_lin = 10.0 ** (snr_db / 10.0)
        n_var = np.sqrt(1.0 / (2.0 * snr_lin))
        
        # 初始化结果收集器
        deepsc_bleu, deepsc_sim, deepsc_mi = [], [], []
        jscc_bleu, jscc_sim = [], []
        trad_bleu = {k: [] for k in results.keys() if k.endswith('_bleu') and not k.startswith('deepsc') and not k.startswith('jscc')}
        trad_sim = {k: [] for k in results.keys() if k.endswith('_sim') and not k.startswith('deepsc') and not k.startswith('jscc')}
        
        # 评估每个批次
        for batch in tqdm.tqdm(test_loader, desc=f"SNR {snr_db} dB"):
            batch = batch.to(device)
            batch_np = batch.cpu().numpy()
            
            # ===== 1. DeepSC评估 =====
            with torch.no_grad():
                # 前向传播
                logits, tx, rx = lit.model(batch, n_var, lit.channel, return_tx_rx=True)
                pred = logits.argmax(dim=-1)
                
                # 计算指标
                batch_bleu = bleu_score(pred, batch)
                deepsc_bleu.append(batch_bleu)
                
                # 计算句子相似度
                str_pred = [' '.join(vocab.decode(x.tolist())) for x in pred.cpu()]
                str_ref = [' '.join(vocab.decode(x.tolist())) for x in batch.cpu()]
                batch_sim = sentence_similarity(str_pred, str_ref, device=device)
                deepsc_sim.extend(batch_sim)
                
                # 计算互信息
                tx_f, rx_f = [z.reshape(-1, z.size(-1)) for z in (tx, rx)]
                batch_mi = lit.mine(tx_f, rx_f).item()
                deepsc_mi.append(batch_mi)
            
            # ===== 2. JSCC评估 =====
            with torch.no_grad():
                # 前向传播
                jscc_logits = jscc_model(batch, n_var, lit.channel)
                jscc_pred = jscc_logits.argmax(dim=-1)
                
                # 计算指标
                batch_jscc_bleu = bleu_score(jscc_pred, batch[:, 1:])  # 注意JSCC预测从第二个token开始
                jscc_bleu.append(batch_jscc_bleu)
                
                # 计算句子相似度
                jscc_str_pred = [' '.join(vocab.decode(x.tolist())) for x in jscc_pred.cpu()]
                jscc_str_ref = [' '.join(vocab.decode(x[1:].tolist())) for x in batch.cpu()]  # 从第二个token开始
                batch_jscc_sim = sentence_similarity(jscc_str_pred, jscc_str_ref, device=device)
                jscc_sim.extend(batch_jscc_sim)
            
            # ===== 3. 传统方法评估 =====
            # 处理每种传统编码组合
            # a) Huffman + Turbo
            bits, lengths = huffman_encoder.encode(batch_np)
            coded_bits = turbo_coder.encode(bits)
            symbols = bits_to_symbols(coded_bits, **modulations['huffman_turbo'])
            rx_symbols = transmit_over_channel(symbols, snr_db, 'awgn')
            rx_llr = symbols_to_llr(rx_symbols, n_var**2, **modulations['huffman_turbo'])
            decoded_bits = turbo_coder.decode(rx_llr, n_var)
            decoded_sentences = huffman_encoder.decode(decoded_bits, lengths)
            
            # 计算BLEU和相似度
            hf_turbo_bleu = []
            hf_turbo_sim = []
            for i, (decoded, ref) in enumerate(zip(decoded_sentences, batch_np)):
                # 转为tensor以便使用bleu_score函数
                dec_tensor = torch.tensor([decoded], device=device)
                ref_tensor = torch.tensor([ref], device=device)
                hf_turbo_bleu.append(bleu_score(dec_tensor, ref_tensor))
                
                # 计算句子相似度
                dec_str = ' '.join(vocab.decode(decoded))
                ref_str = ' '.join(vocab.decode(ref.tolist()))
                hf_turbo_sim.append(sentence_similarity([dec_str], [ref_str], device=device)[0])
            
            trad_bleu['huffman_turbo_bleu'].append(np.mean(hf_turbo_bleu))
            trad_sim['huffman_turbo_sim'].extend(hf_turbo_sim)
            
            # b) Fixed-length + Turbo
            # 类似地实现其他组合...
            bits, lengths = fixed_encoder.encode(batch_np)
            coded_bits = turbo_coder.encode(bits)
            symbols = bits_to_symbols(coded_bits, **modulations['fixed_turbo'])
            rx_symbols = transmit_over_channel(symbols, snr_db, 'awgn')
            rx_llr = symbols_to_llr(rx_symbols, n_var**2, **modulations['fixed_turbo'])
            decoded_bits = turbo_coder.decode(rx_llr, n_var)
            decoded_sentences = fixed_encoder.decode(decoded_bits, lengths)
            
            # 计算指标
            fx_turbo_bleu = []
            fx_turbo_sim = []
            for i, (decoded, ref) in enumerate(zip(decoded_sentences, batch_np)):
                dec_tensor = torch.tensor([decoded], device=device)
                ref_tensor = torch.tensor([ref], device=device)
                fx_turbo_bleu.append(bleu_score(dec_tensor, ref_tensor))
                
                dec_str = ' '.join(vocab.decode(decoded))
                ref_str = ' '.join(vocab.decode(ref.tolist()))
                fx_turbo_sim.append(sentence_similarity([dec_str], [ref_str], device=device)[0])
            
            trad_bleu['fixed_turbo_bleu'].append(np.mean(fx_turbo_bleu))
            trad_sim['fixed_turbo_sim'].extend(fx_turbo_sim)
            
            # c) Huffman + RS
            # ... 实现其他组合
            
            # d) Fixed-length + RS
            # ... 实现其他组合
            
            # e) Brotli + Turbo
            # ... 实现其他组合
        
        # 汇总本SNR下的结果
        results['deepsc_bleu'].append(np.mean(deepsc_bleu))
        results['deepsc_sim'].append(np.mean(deepsc_sim))
        results['deepsc_mi'].append(np.mean(deepsc_mi))
        results['jscc_bleu'].append(np.mean(jscc_bleu))
        results['jscc_sim'].append(np.mean(jscc_sim))
        
        for k in trad_bleu:
            results[k].append(np.mean(trad_bleu[k]))
        for k in trad_sim:
            results[k].append(np.mean(trad_sim[k]))
        
        # 打印当前SNR的结果
        print(f"  DeepSC: BLEU={results['deepsc_bleu'][-1]:.4f}, Sim={results['deepsc_sim'][-1]:.4f}, MI={results['deepsc_mi'][-1]:.4f}")
        print(f"  JSCC:   BLEU={results['jscc_bleu'][-1]:.4f}, Sim={results['jscc_sim'][-1]:.4f}")
        for k in sorted(trad_bleu.keys()):
            method = k.replace('_bleu', '')
            print(f"  {method}: BLEU={results[k][-1]:.4f}, Sim={results[method+'_sim'][-1]:.4f}")

    # ---------- 5. 可视化结果 ----------
    # 创建结果目录
    results_dir = Path('evaluation_results')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果数据
    results_df = pd.DataFrame({
        'SNR': snrs,
        **{k: v for k, v in results.items()}
    })
    results_df.to_csv(results_dir / 'results.csv', index=False)
    
    # 绘制BLEU曲线
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, results['deepsc_bleu'], 'o-', label='DeepSC')
    plt.plot(snrs, results['jscc_bleu'], 's-', label='JSCC')
    for k in sorted([k for k in results.keys() if k.endswith('_bleu') and not k.startswith('deepsc') and not k.startswith('jscc')]):
        plt.plot(snrs, results[k], '--', label=k.replace('_bleu', ''))
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score vs. SNR')
    plt.grid(True)
    plt.legend()
    plt.savefig(results_dir / 'bleu_vs_snr.png', dpi=300)
    
    # 绘制相似度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, results['deepsc_sim'], 'o-', label='DeepSC')
    plt.plot(snrs, results['jscc_sim'], 's-', label='JSCC')
    for k in sorted([k for k in results.keys() if k.endswith('_sim') and not k.startswith('deepsc') and not k.startswith('jscc')]):
        plt.plot(snrs, results[k], '--', label=k.replace('_sim', ''))
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sentence Similarity')
    plt.title('Sentence Similarity vs. SNR')
    plt.grid(True)
    plt.legend()
    plt.savefig(results_dir / 'sim_vs_snr.png', dpi=300)
    
    # 绘制互信息曲线
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, results['deepsc_mi'], 'o-', label='DeepSC MI-LB')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mutual Information Lower Bound')
    plt.title('Mutual Information vs. SNR')
    plt.grid(True)
    plt.legend()
    plt.savefig(results_dir / 'mi_vs_snr.png', dpi=300)
    
    print(f"\n评估完成！结果已保存到 {results_dir}/")

if __name__ == '__main__':
    main()