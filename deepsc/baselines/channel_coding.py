# deepsc/baselines/channel_coding.py - 传统信道编码实现
import numpy as np
import torch
from typing import Tuple, List
from commpy.channelcoding import Trellis, turbo_encode, turbo_decode
from reedsolo import RSCodec

class ChannelCoder:
    """信道编码器基类"""
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """对比特序列进行信道编码"""
        raise NotImplementedError
    
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """对编码后的比特序列进行解码"""
        raise NotImplementedError

class TurboCoder(ChannelCoder):
    """Turbo编码实现"""
    def __init__(self, rate: float = 1/3, interleaver_size: int = 128, iterations: int = 5):
        """
        参数:
            rate: 编码率，默认1/3
            interleaver_size: 交织器大小
            iterations: 解码迭代次数
        """
        # 定义卷积码(PCCC)的网格结构
        self.trellis1 = Trellis(
            memory=[2], 
            g_matrix=[[0o7, 0o5]]
        )
        self.trellis2 = Trellis(
            memory=[2], 
            g_matrix=[[0o7, 0o5]]
        )
        self.rate = rate
        self.interleaver_size = interleaver_size
        self.iterations = iterations
        
        # 创建随机交织器
        np.random.seed(42)  # 固定种子以确保复现性
        self.interleaver = np.random.permutation(interleaver_size)
    
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Turbo编码
        
        参数:
            bits: 输入比特，形状 [batch_size, bit_length]
            
        返回:
            编码后比特，形状 [batch_size, encoded_bit_length]
        """
        batch_size = bits.shape[0]
        encoded_batches = []
        
        for i in range(batch_size):
            msg = bits[i].astype(np.int)
            
            # 分块编码
            encoded_blocks = []
            for j in range(0, len(msg), self.interleaver_size):
                block = msg[j:j+self.interleaver_size]
                # 如果不足交织器大小，则填充
                if len(block) < self.interleaver_size:
                    block = np.pad(block, (0, self.interleaver_size - len(block)))
                
                # Turbo编码
                encoded = turbo_encode(block, self.trellis1, self.trellis2, self.interleaver)
                encoded_blocks.append(encoded)
            
            # 合并所有块
            encoded_msg = np.concatenate(encoded_blocks)
            encoded_batches.append(encoded_msg)
        
        # 找出最长编码，将所有编码填充到相同长度
        max_len = max(len(enc) for enc in encoded_batches)
        padded_encoded = []
        for enc in encoded_batches:
            padded = np.pad(enc, (0, max_len - len(enc)))
            padded_encoded.append(padded)
        
        return np.array(padded_encoded)
    
    def decode(self, noisy_bits: np.ndarray, sigma: float) -> np.ndarray:
        """
        Turbo解码
        
        参数:
            noisy_bits: 含噪声的LLR (对数似然比)，形状 [batch_size, encoded_bit_length]
            sigma: 噪声标准差
            
        返回:
            解码后比特，形状 [batch_size, original_bit_length]
        """
        batch_size = noisy_bits.shape[0]
        decoded_batches = []
        
        for i in range(batch_size):
            llr = noisy_bits[i]
            
            # 分块解码
            decoded_blocks = []
            for j in range(0, len(llr), self.interleaver_size * 3):  # *3 因为码率为1/3
                if j + self.interleaver_size * 3 <= len(llr):
                    block_llr = llr[j:j+self.interleaver_size*3]
                    
                    # Turbo解码 (Log-MAP算法)
                    decoded = turbo_decode(block_llr, self.trellis1, self.trellis2, 
                                          self.interleaver, self.iterations, sigma)
                    
                    # 硬判决
                    decoded = (decoded > 0).astype(np.int)
                    decoded_blocks.append(decoded)
                else:
                    # 处理不完整的块
                    remaining = len(llr) - j
                    remaining_input_size = remaining // 3
                    if remaining_input_size > 0:
                        # 填充到完整块大小
                        padded_llr = np.pad(llr[j:], (0, self.interleaver_size*3 - remaining))
                        decoded = turbo_decode(padded_llr, self.trellis1, self.trellis2, 
                                              self.interleaver, self.iterations, sigma)
                        decoded = (decoded > 0).astype(np.int)
                        # 只保留有效部分
                        decoded = decoded[:remaining_input_size]
                        decoded_blocks.append(decoded)
            
            # 合并所有块
            if decoded_blocks:
                decoded_msg = np.concatenate(decoded_blocks)
                decoded_batches.append(decoded_msg)
            else:
                # 处理边缘情况
                decoded_batches.append(np.zeros(1))
        
        # 找出最长解码结果，将所有结果填充到相同长度
        max_len = max(len(dec) for dec in decoded_batches)
        padded_decoded = []
        for dec in decoded_batches:
            padded = np.pad(dec, (0, max_len - len(dec)))
            padded_decoded.append(padded)
        
        return np.array(padded_decoded)

class RSCoder(ChannelCoder):
    """Reed-Solomon编码实现"""
    def __init__(self, n: int = 255, k: int = 223):
        """
        参数:
            n: 码字长度(必须是2^m-1，最大为255)
            k: 信息长度(必须小于n)
        """
        self.n = n
        self.k = k
        self.rs = RSCodec(n - k)
    
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Reed-Solomon编码
        
        参数:
            bits: 输入比特，形状 [batch_size, bit_length]
            
        返回:
            编码后字节，形状 [batch_size, encoded_byte_length]
        """
        batch_size = bits.shape[0]
        encoded_batches = []
        
        for i in range(batch_size):
            # 将比特转换为字节
            msg_bytes = self._bits_to_bytes(bits[i])
            
            # 分块编码
            encoded_blocks = []
            for j in range(0, len(msg_bytes), self.k):
                block = msg_bytes[j:j+self.k]
                # 如果不足k，则填充
                if len(block) < self.k:
                    block = np.pad(block, (0, self.k - len(block)))
                
                # RS编码
                encoded = self.rs.encode(block.tobytes())
                encoded_blocks.append(np.frombuffer(encoded, dtype=np.uint8))
            
            # 合并所有块
            if encoded_blocks:
                encoded_msg = np.concatenate(encoded_blocks)
                encoded_batches.append(encoded_msg)
            else:
                # 处理空消息
                encoded_batches.append(np.zeros(self.n, dtype=np.uint8))
        
        # 找出最长编码，将所有编码填充到相同长度
        max_len = max(len(enc) for enc in encoded_batches)
        padded_encoded = []
        for enc in encoded_batches:
            padded = np.pad(enc, (0, max_len - len(enc)))
            padded_encoded.append(padded)
        
        return np.array(padded_encoded)
    
    def decode(self, noisy_bytes: np.ndarray) -> np.ndarray:
        """
        Reed-Solomon解码
        
        参数:
            noisy_bytes: 含噪声的字节，形状 [batch_size, encoded_byte_length]
            
        返回:
            解码后比特，形状 [batch_size, original_bit_length]
        """
        batch_size = noisy_bytes.shape[0]
        decoded_batches = []
        
        for i in range(batch_size):
            noisy_msg = noisy_bytes[i]
            
            # 分块解码
            decoded_blocks = []
            for j in range(0, len(noisy_msg), self.n):
                if j + self.n <= len(noisy_msg):
                    block = noisy_msg[j:j+self.n]
                    
                    try:
                        # RS解码
                        decoded, _ = self.rs.decode(block.tobytes())
                        # 只保留信息部分
                        decoded = np.frombuffer(decoded, dtype=np.uint8)[:self.k]
                        decoded_blocks.append(decoded)
                    except Exception:
                        # 解码失败时返回零序列
                        decoded_blocks.append(np.zeros(self.k, dtype=np.uint8))
                else:
                    # 处理不完整的块
                    remaining = len(noisy_msg) - j
                    if remaining > self.n - self.k:  # 至少能恢复一些信息
                        try:
                            # 填充到完整块大小
                            padded = np.pad(noisy_msg[j:], (0, self.n - remaining))
                            decoded, _ = self.rs.decode(padded.tobytes())
                            # 计算可恢复的信息长度
                            info_len = max(0, remaining - (self.n - self.k))
                            decoded = np.frombuffer(decoded, dtype=np.uint8)[:info_len]
                            decoded_blocks.append(decoded)
                        except Exception:
                            # 解码失败返回零序列
                            info_len = max(0, remaining - (self.n - self.k))
                            decoded_blocks.append(np.zeros(info_len, dtype=np.uint8))
            
            # 合并所有块并转回比特
            if decoded_blocks:
                decoded_bytes = np.concatenate(decoded_blocks)
                decoded_bits = self._bytes_to_bits(decoded_bytes)
                decoded_batches.append(decoded_bits)
            else:
                # 处理边缘情况
                decoded_batches.append(np.zeros(8, dtype=np.int))
        
        # 找出最长解码结果，将所有结果填充到相同长度
        max_len = max(len(dec) for dec in decoded_batches)
        padded_decoded = []
        for dec in decoded_batches:
            padded = np.pad(dec, (0, max_len - len(dec)))
            padded_decoded.append(padded)
        
        return np.array(padded_decoded)
    
    def _bits_to_bytes(self, bits: np.ndarray) -> np.ndarray:
        """将比特序列转换为字节序列"""
        # 确保比特长度是8的倍数
        padding = (8 - len(bits) % 8) % 8
        if padding:
            bits = np.pad(bits, (0, padding))
        
        # 每8位转换为1个字节
        bytes_array = np.zeros(len(bits) // 8, dtype=np.uint8)
        for i in range(0, len(bits), 8):
            for j in range(8):
                if i + j < len(bits):
                    bytes_array[i // 8] |= (bits[i + j] << (7 - j))
        
        return bytes_array
    
    def _bytes_to_bits(self, bytes_array: np.ndarray) -> np.ndarray:
        """将字节序列转换为比特序列"""
        bits = np.zeros(len(bytes_array) * 8, dtype=np.int)
        
        for i, byte in enumerate(bytes_array):
            for j in range(8):
                bits[i * 8 + j] = (byte >> (7 - j)) & 1
        
        return bits