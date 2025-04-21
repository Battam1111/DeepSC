# -*- coding: utf-8 -*-
"""
传统源编码与解码方法
==========================================
实现了以下编码方法：
1. HuffmanEncoder: 霍夫曼编码
2. FixedLengthEncoder: 固定长度编码

这些方法用于与DeepSC性能比较。
"""
import numpy as np
from typing import List, Tuple, Dict, Any
import collections

class BaselineEncoder:
    """传统编码方法基类"""
    def encode(self, sentences: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        将整数序列编码为比特
        
        参数:
            sentences: 整数序列，形状 [batch_size, seq_len]
            
        返回:
            (编码后的比特, 每个句子的比特长度)
        """
        raise NotImplementedError
    
    def decode(self, bits: np.ndarray, lengths: List[int]) -> List[List[int]]:
        """
        将比特解码为整数序列
        
        参数:
            bits: 比特序列，形状 [batch_size, bit_length]
            lengths: 每个句子的实际比特长度
            
        返回:
            解码后的整数序列列表
        """
        raise NotImplementedError

class HuffmanNode:
    """霍夫曼树的节点"""
    def __init__(self, freq, token=None, left=None, right=None):
        self.freq = freq
        self.token = token
        self.left = left
        self.right = right
        
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(token_freqs):
    """构建霍夫曼树"""
    nodes = [HuffmanNode(freq, token) for token, freq in token_freqs.items()]
    
    while len(nodes) > 1:
        nodes.sort()  # 按频率排序
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
        nodes.append(parent)
    
    return nodes[0] if nodes else None

def build_huffman_codes(root):
    """从霍夫曼树构建编码表"""
    codes = {}
    
    def _traverse(node, code):
        if node.token is not None:  # 叶子节点
            codes[node.token] = code
            return
        
        if node.left:
            _traverse(node.left, code + "0")
        if node.right:
            _traverse(node.right, code + "1")
    
    _traverse(root, "")
    return codes

class HuffmanEncoder(BaselineEncoder):
    """
    霍夫曼编码实现
    
    参数:
        vocab_size: 词表大小
        token_freqs: 词元频率字典，如果为None则使用均匀分布
    """
    def __init__(self, vocab_size: int, token_freqs: Dict[int, int] = None):
        self.vocab_size = vocab_size
        
        # 如果没有提供频率，则使用均匀分布
        if token_freqs is None:
            token_freqs = {i: 1 for i in range(vocab_size)}
            
        # 构建霍夫曼树和编码表
        self.tree = build_huffman_tree(token_freqs)
        self.codebook = build_huffman_codes(self.tree)
        
        # 构建解码表（反向映射）
        self.reverse_codebook = {}
        for token, code in self.codebook.items():
            self.reverse_codebook[code] = token
        
        # 为未登录词设置默认编码（四位码，比如1010）
        self.default_code = "1010"
    
    def encode(self, sentences: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        霍夫曼编码
        
        参数:
            sentences: 整数序列，形状 [batch_size, seq_len]
            
        返回:
            (编码后的比特数组, 每个句子的比特长度)
        """
        encoded_bits = []
        lengths = []
        
        for sentence in sentences:
            # 对每个句子的每个词进行编码
            bits = []
            for token in sentence:
                if token < self.vocab_size and token in self.codebook:
                    token_bits = [int(b) for b in self.codebook[token]]
                    bits.extend(token_bits)
                else:
                    # 对于词表外的词，使用默认编码
                    bits.extend([int(b) for b in self.default_code])
            
            encoded_bits.append(bits)
            lengths.append(len(bits))
        
        # 找出最长编码，将所有编码填充到相同长度
        max_len = max(lengths)
        padded_bits = []
        for bits in encoded_bits:
            padded = bits + [0] * (max_len - len(bits))
            padded_bits.append(padded)
        
        return np.array(padded_bits), lengths
    
    def decode(self, bits: np.ndarray, lengths: List[int]) -> List[List[int]]:
        """
        霍夫曼解码
        
        参数:
            bits: 编码后的比特数组，形状 [batch_size, bit_length]
            lengths: 每个句子的实际比特长度
            
        返回:
            解码后的整数序列列表
        """
        decoded_sentences = []
        
        for i, (sentence_bits, length) in enumerate(zip(bits, lengths)):
            # 只使用有效的比特
            valid_bits = sentence_bits[:length]
            
            # 霍夫曼解码
            current_code = ""
            decoded = []
            
            for bit in valid_bits:
                current_code += str(int(bit))
                if current_code in self.reverse_codebook:
                    decoded.append(self.reverse_codebook[current_code])
                    current_code = ""
            
            decoded_sentences.append(decoded)
        
        return decoded_sentences

class FixedLengthEncoder(BaselineEncoder):
    """
    固定长度编码实现
    
    参数:
        vocab_size: 词表大小
        bits_per_token: 每个词元使用的比特数
    """
    def __init__(self, vocab_size: int, bits_per_token: int = 5):
        self.vocab_size = vocab_size
        self.bits_per_token = bits_per_token
        
    def _int_to_bits(self, n: int) -> List[int]:
        """将整数转换为定长比特表示"""
        bits = [int(b) for b in bin(n)[2:]]
        # 填充到固定长度
        padding = [0] * (self.bits_per_token - len(bits))
        return padding + bits[-self.bits_per_token:]  # 如果太长，截断
    
    def _bits_to_int(self, bits: List[int]) -> int:
        """将比特表示转换回整数"""
        return int(''.join(str(b) for b in bits), 2)
    
    def encode(self, sentences: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        固定长度编码
        
        参数:
            sentences: 整数序列，形状 [batch_size, seq_len]
            
        返回:
            (编码后的比特数组, 每个句子的比特长度)
        """
        encoded_bits = []
        lengths = []
        
        for sentence in sentences:
            # 对每个句子的每个词进行定长编码
            bits = []
            for token in sentence:
                # 截断到词表大小范围内
                token = min(token, self.vocab_size - 1)
                token_bits = self._int_to_bits(token)
                bits.extend(token_bits)
            
            encoded_bits.append(bits)
            lengths.append(len(bits))
        
        # 找出最长编码，将所有编码填充到相同长度
        max_len = max(lengths)
        padded_bits = []
        for bits in encoded_bits:
            padded = bits + [0] * (max_len - len(bits))
            padded_bits.append(padded)
        
        return np.array(padded_bits), lengths
    
    def decode(self, bits: np.ndarray, lengths: List[int]) -> List[List[int]]:
        """
        固定长度解码
        
        参数:
            bits: 编码后的比特数组，形状 [batch_size, bit_length]
            lengths: 每个句子的实际比特长度
            
        返回:
            解码后的整数序列列表
        """
        decoded_sentences = []
        
        for i, (sentence_bits, length) in enumerate(zip(bits, lengths)):
            # 只使用有效的比特
            valid_bits = sentence_bits[:length]
            
            # 按固定长度分组并解码
            decoded = []
            for j in range(0, len(valid_bits), self.bits_per_token):
                if j + self.bits_per_token <= len(valid_bits):
                    token_bits = valid_bits[j:j + self.bits_per_token]
                    token = self._bits_to_int(token_bits)
                    decoded.append(token)
            
            decoded_sentences.append(decoded)
        
        return decoded_sentences