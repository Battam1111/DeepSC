# deepsc/baselines/traditional.py - 传统编码方法接口
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import huffman
import brotli

class BaselineEncoder:
    """传统编码方法基类"""
    def encode(self, sentences: List[List[int]]) -> np.ndarray:
        """将整数序列编码为比特"""
        raise NotImplementedError
    
    def decode(self, bits: np.ndarray, lengths: List[int]) -> List[List[int]]:
        """将比特解码为整数序列"""
        raise NotImplementedError

class HuffmanEncoder(BaselineEncoder):
    """霍夫曼编码实现"""
    def __init__(self, vocab_size: int, token_freqs: Dict[int, int] = None):
        self.vocab_size = vocab_size
        
        # 如果没有提供频率，则使用均匀分布
        if token_freqs is None:
            token_freqs = {i: 1 for i in range(vocab_size)}
            
        # 构建霍夫曼码表
        self.codebook = huffman.codebook(token_freqs)
        self.reverse_codebook = {code: token for token, code in self.codebook.items()}
    
    def encode(self, sentences: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
        """
        霍夫曼编码
        
        参数:
            sentences: 整数序列列表
            
        返回:
            (编码后的比特, 每个句子的比特长度)
        """
        encoded_bits = []
        lengths = []
        
        for sentence in sentences:
            # 对每个句子的每个词进行编码
            bits = []
            for token in sentence:
                if token in self.codebook:
                    token_bits = [int(b) for b in self.codebook[token]]
                    bits.extend(token_bits)
                else:
                    # 对于词表外的词，使用固定编码
                    bits.extend([1, 0, 1, 0])
            
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
            bits: 编码后的比特
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
    """固定长度编码实现"""
    def __init__(self, vocab_size: int, bits_per_token: int = 5):
        self.vocab_size = vocab_size
        self.bits_per_token = bits_per_token
        
    def _int_to_bits(self, n: int) -> List[int]:
        """将整数转换为定长比特表示"""
        bits = [int(b) for b in bin(n)[2:]]
        # 填充到固定长度
        padding = [0] * (self.bits_per_token - len(bits))
        return padding + bits
    
    def _bits_to_int(self, bits: List[int]) -> int:
        """将比特表示转换回整数"""
        return int(''.join(str(b) for b in bits), 2)
    
    def encode(self, sentences: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
        """
        固定长度编码
        
        参数:
            sentences: 整数序列列表
            
        返回:
            (编码后的比特, 每个句子的比特长度)
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
            bits: 编码后的比特
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

class BrotliEncoder(BaselineEncoder):
    """Brotli压缩算法封装"""
    def __init__(self, quality: int = 11):
        self.quality = quality  # 压缩质量，1-11，越高压缩比越高但越慢
    
    def encode(self, sentences: List[List[int]]) -> Tuple[bytes, List[int]]:
        """
        Brotli编码
        
        参数:
            sentences: 整数序列列表
            
        返回:
            (压缩后的字节, 原始句子长度)
        """
        # 将整数列表转为字节
        byte_data = []
        lengths = []
        
        for sentence in sentences:
            # 每个整数使用变长编码
            sentence_bytes = bytearray()
            for token in sentence:
                # 简单的变长编码：小整数用较少字节
                if token < 128:
                    sentence_bytes.append(token)
                else:
                    high = (token >> 8) | 0x80  # 设置高位标志
                    low = token & 0xFF
                    sentence_bytes.extend([high, low])
            
            byte_data.append(sentence_bytes)
            lengths.append(len(sentence))
        
        # 将所有句子连接并压缩
        all_bytes = b''.join(byte_data)
        compressed = brotli.compress(all_bytes, quality=self.quality)
        
        return compressed, lengths
    
    def decode(self, compressed: bytes, lengths: List[int]) -> List[List[int]]:
        """
        Brotli解码
        
        参数:
            compressed: 压缩后的字节
            lengths: 原始句子长度列表
            
        返回:
            解码后的整数序列列表
        """
        try:
            # 解压缩
            decompressed = brotli.decompress(compressed)
            
            # 解析回整数列表
            tokens = []
            i = 0
            while i < len(decompressed):
                if decompressed[i] & 0x80:  # 检查高位标志
                    high = decompressed[i] & 0x7F
                    low = decompressed[i+1]
                    tokens.append((high << 8) | low)
                    i += 2
                else:
                    tokens.append(decompressed[i])
                    i += 1
            
            # 根据原始长度分割回句子
            sentences = []
            start = 0
            for length in lengths:
                if start + length <= len(tokens):
                    sentences.append(tokens[start:start+length])
                else:
                    # 处理解码错误：填充零
                    sentences.append([0] * length)
                start += length
            
            return sentences
        except Exception as e:
            # 解码错误时返回零序列
            return [[0] * length for length in lengths]