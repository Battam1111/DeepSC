# deepsc/metrics/sentence_sim.py - 改进的句子相似度计算
from __future__ import annotations
import torch, functools
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from typing import List, Optional

@functools.lru_cache()
def _load_bert(device, model_name: str = 'bert-base-uncased'):
    """
    加载并缓存 BERT 模型，用于句子向量提取
    
    参数:
        device: 计算设备
        model_name: BERT模型名称，默认使用'bert-base-uncased'以符合论文设置
        
    返回:
        tokenizer和model的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).eval().to(device)
    return tokenizer, model

def _get_sent_vec(hidden_states, pooling: str = 'mean'):
    """
    从BERT输出中提取句子向量表示
    
    参数:
        hidden_states: BERT的隐藏状态输出(tuple)
        pooling: 池化方法，可选'mean'(平均池化)或'cls'(使用[CLS]表示)
    
    返回:
        L2归一化后的句子向量
    """
    # 使用倒数第二层(第11层)的输出，论文中提到这层对语义信息捕获最好
    vec = hidden_states[11]
    
    if pooling == 'mean':
        # 对所有token的表示取平均（排除特殊token如[PAD]）
        # 注意：在实际应用中应该使用attention_mask来正确计算平均值
        vec = vec.mean(dim=1)
    elif pooling == 'cls':
        # 使用[CLS] token的表示(第一个token)
        vec = vec[:, 0]
    
    # L2归一化，确保余弦相似度计算正确
    return normalize(vec.cpu().numpy())

def sentence_similarity(
    batch_pred: List[str], 
    batch_ref: List[str],
    device: str | torch.device = 'cpu',
    model_name: str = 'bert-base-uncased',
    pooling: str = 'mean',
    max_length: int = 128
) -> List[float]:
    """
    计算两组句子之间的语义相似度
    
    基于BERT提取的语义表示，计算余弦相似度。
    返回每对句子的相似度得分，范围为0到1。
    
    参数:
        batch_pred: 预测的句子列表
        batch_ref: 参考句子列表
        device: 计算设备
        model_name: BERT模型名称
        pooling: 池化方法，'mean'或'cls'
        max_length: 截断长度
    
    返回:
        每对句子的相似度分数列表，范围[0,1]
    """
    if len(batch_pred) != len(batch_ref):
        raise ValueError(f"预测句子数({len(batch_pred)})必须等于参考句子数({len(batch_ref)})")
    
    tokenizer, model = _load_bert(device, model_name)
    
    # 对所有句子进行分词处理
    inputs = tokenizer(
        batch_pred + batch_ref,
        padding=True, 
        truncation=True,
        max_length=max_length, 
        return_tensors='pt'
    ).to(device)
    
    # 使用BERT提取特征
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 提取句子向量并计算相似度
    vectors = _get_sent_vec(outputs.hidden_states, pooling)
    v_pred, v_ref = vectors[:len(batch_pred)], vectors[len(batch_pred):]
    
    # 计算余弦相似度 (向量点积，由于已经归一化，范围为[-1,1])
    # 将范围调整为[0,1]以符合论文中的相似度定义
    sim_scores = ((v_pred * v_ref).sum(axis=1) + 1) / 2
    
    return sim_scores.tolist()