# -*- coding: utf-8 -*-
"""
数据预处理脚本
=============
处理欧洲议会数据集，生成训练集、测试集和词表

功能:
1. 读取原始文本数据
2. 清洗和标准化文本
3. 构建词表
4. 生成训练集和测试集
5. 保存为pickle和json格式
"""

import os
import re
import json
import pickle
import random
from pathlib import Path
from collections import Counter
import tqdm
import argparse
from typing import List, Dict, Tuple

# 特殊标记
SPECIAL_TOKENS = {
    '<PAD>': 0,  # 填充符
    '<START>': 1,  # 起始符
    '<END>': 2,  # 结束符
    '<UNK>': 3,  # 未知词
}

def clean_text(text: str) -> str:
    """
    清洗文本
    
    参数:
        text: 输入文本
        
    返回:
        清洗后的文本
    """
    # 转换为小写
    text = text.lower()
    
    # 移除特殊字符和多余空格
    text = re.sub(r'[^a-z0-9,.?!\'"\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize(text: str) -> List[str]:
    """
    分词
    
    参数:
        text: 输入文本
        
    返回:
        分词列表
    """
    # 简单按空格分词
    return text.split()

def build_vocab(texts: List[str], min_freq: int = 5, max_vocab: int = 30000) -> Dict[str, int]:
    """
    构建词表
    
    参数:
        texts: 文本列表
        min_freq: 最小词频
        max_vocab: 最大词表大小
        
    返回:
        word2idx: 词到索引的映射
    """
    counter = Counter()
    
    # 计数所有词
    for text in tqdm.tqdm(texts, desc="构建词表"):
        tokens = tokenize(clean_text(text))
        counter.update(tokens)
    
    # 过滤低频词，并限制词表大小
    words = [word for word, count in counter.most_common() if count >= min_freq]
    words = words[:max_vocab - len(SPECIAL_TOKENS)]
    
    # 构建词表
    word2idx = {word: idx + len(SPECIAL_TOKENS) for idx, word in enumerate(words)}
    
    # 添加特殊标记
    word2idx.update(SPECIAL_TOKENS)
    
    return word2idx

def text_to_indices(text: str, word2idx: Dict[str, int]) -> List[int]:
    """
    将文本转换为索引序列
    
    参数:
        text: 输入文本
        word2idx: 词表
        
    返回:
        索引序列
    """
    tokens = tokenize(clean_text(text))
    indices = [SPECIAL_TOKENS['<START>']]
    
    for token in tokens:
        if token in word2idx:
            indices.append(word2idx[token])
        else:
            indices.append(SPECIAL_TOKENS['<UNK>'])
    
    indices.append(SPECIAL_TOKENS['<END>'])
    
    return indices

def prepare_data(input_file: str, output_dir: str, max_len: int = 30, test_ratio: float = 0.1, seed: int = 42):
    """
    准备数据集
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出目录
        max_len: 最大句子长度
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取原始数据
    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 清洗和过滤数据
    print("清洗和过滤数据...")
    filtered_texts = []
    for line in tqdm.tqdm(lines):
        text = clean_text(line)
        tokens = tokenize(text)
        
        # 过滤太短或太长的句子
        if 4 <= len(tokens) <= max_len:
            filtered_texts.append(text)
    
    # 构建词表
    print("构建词表...")
    word2idx = build_vocab(filtered_texts)
    
    # 保存词表
    vocab_file = os.path.join(output_dir, 'vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({'token_to_idx': word2idx}, f, ensure_ascii=False, indent=2)
    
    print(f"词表已保存: {vocab_file} (共 {len(word2idx)} 个词)")
    
    # 划分训练集和测试集
    random.shuffle(filtered_texts)
    split_idx = int(len(filtered_texts) * (1 - test_ratio))
    train_texts = filtered_texts[:split_idx]
    test_texts = filtered_texts[split_idx:]
    
    # 转换为索引序列
    print("转换训练集为索引序列...")
    train_indices = []
    for text in tqdm.tqdm(train_texts):
        indices = text_to_indices(text, word2idx)
        train_indices.append(indices)
    
    print("转换测试集为索引序列...")
    test_indices = []
    for text in tqdm.tqdm(test_texts):
        indices = text_to_indices(text, word2idx)
        test_indices.append(indices)
    
    # 保存训练集和测试集
    train_file = os.path.join(output_dir, 'train_data.pkl')
    test_file = os.path.join(output_dir, 'test_data.pkl')
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_indices, f)
    
    with open(test_file, 'wb') as f:
        pickle.dump(test_indices, f)
    
    print(f"训练集已保存: {train_file} (共 {len(train_indices)} 个样本)")
    print(f"测试集已保存: {test_file} (共 {len(test_indices)} 个样本)")

def main():
    parser = argparse.ArgumentParser(description='处理欧洲议会数据集')
    parser.add_argument('--input', default='data/europarl/europarl-v7.fr-en.en',
                        help='输入文件路径')
    parser.add_argument('--output', default='data/europarl',
                        help='输出目录')
    parser.add_argument('--max-len', type=int, default=30,
                        help='最大句子长度')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.max_len, args.test_ratio, args.seed)

if __name__ == '__main__':
    main()