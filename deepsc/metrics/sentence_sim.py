# deepsc/metrics/sentence_sim.py - 改进的句子相似度计算
from __future__ import annotations
import torch
import functools # 用于 LRU 缓存
from transformers import AutoTokenizer, AutoModel # 从 Hugging Face 加载预训练模型和分词器
from sklearn.preprocessing import normalize # 用于 L2 范数归一化
from typing import List, Optional
import numpy as np

# 使用 LRU 缓存装饰器来避免重复加载模型和分词器，提高效率
# maxsize=None 表示缓存大小无限制 (或根据系统资源调整)
@functools.lru_cache(maxsize=None)
def _load_bert(device: str | torch.device, model_name: str = 'bert-base-uncased'):
    """
    加载并缓存（Cache）指定的预训练 BERT 模型和对应的分词器。

    使用 Hugging Face 的 Transformers 库自动下载或加载本地缓存的模型。
    模型加载后设置为评估模式 (`eval()`) 并移动到指定设备。

    参数:
        device (str | torch.device): 模型加载的目标设备 (例如 'cpu', 'cuda', 'cuda:0')。
        model_name (str):              要加载的 BERT 模型的名称 (例如 'bert-base-uncased', 'bert-large-uncased')。
                                       默认为 'bert-base-uncased'，与 DeepSC 原论文设置保持一致。

    返回:
        tuple: 包含加载好的 tokenizer 和 model 的元组 (tokenizer, model)。
               tokenizer: 用于将文本转换为模型输入的 ID。
               model: 加载的 BERT 模型实例。
    """
    print(f"  [Sentence Similarity] 正在加载或缓存 '{model_name}' 分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  [Sentence Similarity] 正在加载或缓存 '{model_name}' 模型 (用于评估)...")
    # 加载模型，并指定输出所有隐藏状态 (output_hidden_states=True) 以便后续选择层
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).eval().to(device) # 设置为评估模式并移动到设备
    print(f"  [Sentence Similarity] 模型和分词器 '{model_name}' 已加载到 {device}。")
    return tokenizer, model

def _get_sent_vec(hidden_states: tuple, attention_mask: torch.Tensor, pooling: str = 'mean') -> np.ndarray:
    """
    从 BERT 模型输出的隐藏状态中提取句子的向量表示。

    可以选择不同的池化（Pooling）策略来聚合词元（Token）级别的向量，得到句子级别的向量。
    通常使用倒数第二层（索引为 -2 或 11，对于 bert-base）的隐藏状态，因为它被认为包含丰富的语义信息。

    参数:
        hidden_states (tuple): BERT 模型输出的隐藏状态元组。元组的每个元素对应一层transformer的输出，
                               形状为 [batch_size, sequence_length, hidden_size]。
        attention_mask (torch.Tensor): 输入分词时产生的注意力掩码，形状 [batch_size, sequence_length]。
                                       用于在平均池化时忽略填充（Padding）词元。值为 1 表示有效词元，0 表示填充。
        pooling (str): 池化策略。可选：
                       'mean': 平均池化。计算所有非填充词元向量的平均值。 (常用且效果较好)
                       'cls': 使用 [CLS] 词元（通常是序列的第一个词元）的向量作为整个句子的表示。

    返回:
        np.ndarray: L2 归一化后的句子向量 NumPy 数组，形状 [batch_size, hidden_size]。
                    归一化是为了方便后续计算余弦相似度。
    """
    # 选择倒数第二层的隐藏状态 (索引 -2)。bert-base-uncased 有 12 层 + embedding 层，所以索引 11 或 -2 对应倒数第二 transformer 层。
    # hidden_states[0] 是 embedding 输出, hidden_states[1] 是第一层输出, ..., hidden_states[12] 是最后一层输出。
    # 因此 hidden_states[11] 或 hidden_states[-2] 是倒数第二层。
    last_but_one_layer = hidden_states[-2] # 形状: [batch_size, seq_len, hidden_size]

    if pooling == 'mean':
        # 平均池化：只考虑非填充词元 (attention_mask == 1)
        # 1. 将填充位置的向量置零：乘以 mask (需要扩展 mask 维度)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_but_one_layer.size()).float()
        sum_embeddings = torch.sum(last_but_one_layer * input_mask_expanded, 1)
        # 2. 计算每个句子实际词元的数量 (防止除以零)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # 3. 计算平均值
        sentence_vectors = sum_embeddings / sum_mask
    elif pooling == 'cls':
        # CLS 池化：直接取第一个词元 ([CLS]) 的向量
        sentence_vectors = last_but_one_layer[:, 0, :] # 形状: [batch_size, hidden_size]
    else:
        raise ValueError(f"不支持的池化方法: {pooling}。请选择 'mean' 或 'cls'。")

    # 将向量移到 CPU，转为 NumPy 数组，并进行 L2 归一化
    sentence_vectors_np = sentence_vectors.detach().cpu().numpy()
    # normalize 函数默认按行 (axis=1) 进行 L2 归一化
    normalized_vectors = normalize(sentence_vectors_np, axis=1, norm='l2')

    return normalized_vectors # 返回 NumPy 数组

def sentence_similarity(
    batch_pred: List[str],
    batch_ref: List[str],
    device: str | torch.device = 'cpu',
    model_name: str = 'bert-base-uncased',
    pooling: str = 'mean',
    max_length: int = 128
) -> List[float]:
    """
    计算两组句子（预测句和参考句）之间的批量语义相似度。

    该函数利用预训练的 BERT 模型提取每个句子的语义向量，然后计算预测句向量和
    对应参考句向量之间的余弦相似度。最终返回的相似度得分范围在 [0, 1] 之间。

    参数:
        batch_pred (List[str]): 预测的句子列表 (纯文本字符串)。
        batch_ref (List[str]):  对应的参考句子列表 (纯文本字符串)。数量必须与 batch_pred 相同。
        device (str | torch.device): 计算设备 ('cpu', 'cuda', etc.)。
        model_name (str): 使用的预训练 BERT 模型名称 (例如 'bert-base-uncased')。
        pooling (str): 从词元向量生成句子向量的池化策略 ('mean' 或 'cls')。
        max_length (int): 输入 BERT 模型前的最大序列长度，超过则截断。

    返回:
        List[float]: 包含每对预测句和参考句之间相似度得分的列表。得分范围 [0, 1]，越高表示越相似。
    """
    # 检查输入列表长度是否一致
    if len(batch_pred) != len(batch_ref):
        raise ValueError(f"预测句子数 ({len(batch_pred)}) 必须等于参考句子数 ({len(batch_ref)})")

    # 处理空输入的情况
    if not batch_pred: # 如果列表为空
        return []

    # 加载或获取缓存的 BERT 模型和分词器
    # 注意：_load_bert 使用了缓存，模型只会在第一次调用时真正加载
    tokenizer, model = _load_bert(device, model_name)

    # --- 文本编码 ---
    # 将所有句子（预测句 + 参考句）合并，一次性进行分词和编码
    # padding=True: 将批次内所有序列填充到最长序列的长度
    # truncation=True: 将超过 max_length 的序列截断
    # return_tensors='pt': 返回 PyTorch 张量
    inputs = tokenizer(
        batch_pred + batch_ref, # 合并列表
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device) # 将编码后的输入移动到指定设备

    # --- 特征提取 ---
    # 使用 BERT 模型进行前向传播，获取隐藏状态
    with torch.no_grad(): # 禁用梯度计算，因为只是在做推理
        outputs = model(**inputs, output_hidden_states=True) # 需要获取 hidden_states

    # --- 句子向量提取与归一化 ---
    # 从隐藏状态中提取句子向量 (使用指定的池化方法)
    # attention_mask 也需要传入以正确处理 mean pooling
    sentence_vectors = _get_sent_vec(outputs.hidden_states, inputs['attention_mask'], pooling)
    # sentence_vectors 是 L2 归一化后的 NumPy 数组

    # --- 相似度计算 ---
    # 将合并的向量拆分为预测句向量和参考句向量
    num_sentences = len(batch_pred)
    v_pred = sentence_vectors[:num_sentences] # 前半部分是预测句向量
    v_ref = sentence_vectors[num_sentences:]  # 后半部分是参考句向量

    # 计算余弦相似度
    # 对于 L2 归一化的向量 u 和 v，其点积 u·v 等于 cos(θ)
    # 这里直接计算对应向量的点积 (逐元素相乘后求和)
    # v_pred * v_ref 实现了逐元素相乘
    # .sum(axis=1) 按行求和，得到每个句子对的点积（即余弦相似度）
    cosine_similarities = (v_pred * v_ref).sum(axis=1) # 范围 [-1, 1]

    # 将余弦相似度从 [-1, 1] 线性映射到 [0, 1]
    # sim = (cos(θ) + 1) / 2
    # 这样做可以使得结果更直观，0 表示完全不相似（向量相反），1 表示完全相似（向量相同）
    similarity_scores = (cosine_similarities + 1) / 2

    # 返回每个句子对的相似度得分列表
    return similarity_scores.tolist()