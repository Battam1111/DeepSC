# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC/deepsc/baselines/channel_coding.py
# (已修改，添加 galois 导入，修改 RSCoder.decode)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
传统信道编码与解码方法
==========================================
实现了以下信道编码方法：
1. TurboCoder: 涡轮码编解码（近似实现）
2. RSCoder: Reed-Solomon编解码（使用 galois 库或简化占位符）

注意：这些是简化实现，主要用于概念验证和效果比较。
"""
import numpy as np
from typing import Tuple, List

# --- 新增: 导入 galois 库 ---
# 尝试导入，即使在 __init__ 中有 try-except，顶层导入确保
# galois.errors 等命名空间在整个文件中可用 (如果库存在)。
try:
    import galois
    _GALOIS_INSTALLED = True
except ImportError:
    _GALOIS_INSTALLED = False
    # 在后续代码中，我们将使用 self._galois_available 来判断实例是否能用Galois，
    # 但 _GALOIS_INSTALLED 可用于静态检查或避免 NameError。

class ChannelCoder:
    """信道编码器基类"""
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        对比特序列进行信道编码

        参数:
            bits: 输入比特，形状 [batch_size, bit_length]

        返回:
            编码后比特或符号，取决于编码器类型。
            TurboCoder 返回比特 [batch_size, encoded_bit_length]。
            RSCoder 返回符号(字节) [batch_size, encoded_symbol_length]。
        """
        raise NotImplementedError

    def decode(self, received_signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        对接收到的信号进行解码

        参数:
            received_signal: 接收到的信号。
                             对于 TurboCoder，这是 LLR 值 [batch_size, encoded_bit_length]。
                             对于 RSCoder，这是硬判决符号(字节) [batch_size, encoded_symbol_length]。
            **kwargs: 其他解码器可能需要的参数 (如 noise_var for Turbo, original_bit_lengths for RS)。

        返回:
            解码后比特，形状 [batch_size, original_bit_length] (填充到批次内最大长度)。
        """
        raise NotImplementedError

class TurboCoder(ChannelCoder):
    """
    涡轮码编解码（简化实现）

    参数:
        rate: 编码率，默认1/3
        iterations: 解码迭代次数
    """
    def __init__(self, rate: float = 1/3, iterations: int = 5):
        self.rate = rate
        self.iterations = iterations
        # 确保交织/解交织索引只生成一次
        self._interleaver_indices = None
        self._deinterleaver_indices = None
        self._current_original_lengths = [] # 用于存储编码时的原始信息比特长度

    def _get_interleaver_indices(self, n: int) -> np.ndarray:
        """获取或生成交织器索引 (固定种子)"""
        if self._interleaver_indices is None or len(self._interleaver_indices) != n:
            # 使用固定的随机状态生成，确保每次运行相同
            rng = np.random.RandomState(42)
            self._interleaver_indices = rng.permutation(n)
        return self._interleaver_indices

    def _get_deinterleaver_indices(self, n: int) -> np.ndarray:
        """获取或生成解交织器索引 (与交织器对应)"""
        if self._deinterleaver_indices is None or len(self._deinterleaver_indices) != n:
            interleaver_indices = self._get_interleaver_indices(n)
            # 解交织索引是交织索引的反向映射
            self._deinterleaver_indices = np.argsort(interleaver_indices)
        return self._deinterleaver_indices

    def _simple_convolutional_encode(self, bits: np.ndarray) -> np.ndarray:
        """简单的卷积编码 (Rate 1/2, G=[1, (1+D^2)/(1+D+D^2)] - 只是示意)"""
        # 这是一个非常简化的示例，并非标准卷积码
        n = len(bits)
        # 输出长度是输入的两倍 (系统位 + 校验位)
        encoded = np.zeros(n * 2, dtype=int) # 使用内建 int
        state = 0 # 简化的状态寄存器
        for i in range(n):
            bit = bits[i]
            encoded[2*i] = bit # 系统位
            encoded[2*i+1] = (bit + state) % 2 # 简化的校验位计算
            state = bit # 更新状态 (非常简化的反馈)
        return encoded

    def _interleave(self, bits: np.ndarray) -> np.ndarray:
        """简单的交织器"""
        n = len(bits)
        indices = self._get_interleaver_indices(n)
        # 确保索引不会超出数组边界
        if n > 0:
            return bits[indices]
        else:
            return np.array([], dtype=bits.dtype)


    def _deinterleave(self, bits: np.ndarray) -> np.ndarray:
        """解交织器"""
        n = len(bits)
        indices = self._get_deinterleaver_indices(n)
         # 确保索引不会超出数组边界
        if n > 0:
            return bits[indices]
        else:
            return np.array([], dtype=bits.dtype)


    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        涡轮码编码 (简化版)

        参数:
            bits: 输入比特，形状 [batch_size, bit_length]

        返回:
            编码后比特，形状 [batch_size, encoded_bit_length] (填充到批次内最大长度)
        """
        batch_size = bits.shape[0]
        encoded_batches = []
        original_lengths = [] # 记录原始信息比特长度，用于解码时恢复

        self._current_original_lengths = [] # 清空旧记录

        for i in range(batch_size):
            # 获取当前批次的输入比特，确保是整数类型
            current_bits = bits[i].astype(int) # 使用 int
            current_len = len(current_bits)
            original_lengths.append(current_len) # 记录原始长度

            if current_len == 0: # 处理空输入
                 encoded_batches.append(np.array([], dtype=int))
                 continue

            # --- 第一个分量编码器 ---
            systematic = current_bits
            encoded1 = self._simple_convolutional_encode(current_bits)
            parity1 = encoded1[1::2] # 只取校验位

            # --- 第二个分量编码器 ---
            interleaved_bits = self._interleave(current_bits)
            encoded2 = self._simple_convolutional_encode(interleaved_bits)
            parity2 = encoded2[1::2] # 只取校验位

            # --- 组合输出 ---
            if self.rate == 1/3:
                encoded = np.concatenate([systematic, parity1, parity2])
            elif self.rate == 1/2:
                half_len = len(parity1) // 2
                encoded = np.concatenate([systematic, parity1[:half_len], parity2[:half_len]])
            else:
                print(f"警告: TurboCoder 不支持 rate={self.rate} 的精确简化，将按 1/3 处理。")
                encoded = np.concatenate([systematic, parity1, parity2])

            encoded_batches.append(encoded)

        # --- 填充到最大长度 ---
        if not encoded_batches: return np.array([])
        # 过滤掉空数组，如果所有都为空则返回空批次
        non_empty_batches = [enc for enc in encoded_batches if len(enc) > 0]
        if not non_empty_batches: return np.zeros((batch_size, 0), dtype=int)

        max_len = max(len(enc) for enc in non_empty_batches)

        padded_encoded = []
        for enc in encoded_batches:
            pad_width = max_len - len(enc)
            padded = np.pad(enc, (0, pad_width), mode='constant', constant_values=0)
            padded_encoded.append(padded)

        # 将原始长度信息存储，解码时可能需要
        self._current_original_lengths = original_lengths

        return np.array(padded_encoded, dtype=int) # 确保输出是整数

    def _logmap_decode_step(self, llr_channel_sys, llr_channel_parity, llr_apriori):
        """单次 BCJR/Log-MAP 解码步骤（极度简化版 - 仅作示意）"""
        n = len(llr_channel_sys)
        if n == 0: # 处理空输入
            return np.array([]), np.array([])

        total_llr_input = llr_apriori + llr_channel_sys
        # 简化模型：后验 LLR 是输入 LLR 和校验 LLR 的加权组合
        llr_posterior = total_llr_input + 0.7 * llr_channel_parity
        llr_extrinsic = llr_posterior - total_llr_input
        decoded_bits = (llr_posterior < 0).astype(int) # LLR < 0 判为 1，否则为 0

        return llr_extrinsic, decoded_bits

    def decode(self, received_signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        涡轮码解码 (简化版) - 接收 LLR

        参数:
            received_signal: 接收到的对数似然比 LLR，形状 [batch_size, encoded_bit_length]
            **kwargs: 预期包含 noise_var (当前简化版未使用)

        返回:
            解码后比特，形状 [batch_size, original_bit_length] (填充到批次内最大长度)
        """
        llr = received_signal # 输入是 LLR
        noise_var = kwargs.get('noise_var', 0.1) # 获取噪声方差，提供默认值
        batch_size = llr.shape[0]
        decoded_batches = []

        # 获取编码时记录的原始长度信息
        original_lengths = getattr(self, '_current_original_lengths', [])
        if len(original_lengths) != batch_size:
            print(f"警告: Turbo解码时原始长度信息丢失或不匹配 ({len(original_lengths)} vs batch {batch_size})。尝试基于码率估计。")
            # 尝试基于码率估计原始长度
            if llr.shape[1] > 0:
                est_len_func = lambda l: int(round(l * self.rate)) # 粗略估计
                original_lengths = [est_len_func(llr.shape[1])] * batch_size
            else:
                 original_lengths = [0] * batch_size # 如果 LLR 长度为 0

        for i in range(batch_size):
            current_llr = llr[i]
            n_original = original_lengths[i] if i < len(original_lengths) else 0 # 安全访问

            if n_original <= 0: # 如果原始长度为0或无效
                decoded_batches.append(np.array([], dtype=int))
                continue

            # --- 检查 LLR 长度 ---
            # 理论上的 LLR 长度应该大约是 n_original / rate
            # 因为有填充，实际长度可能 >= 理论长度
            expected_len_approx = int(round(n_original / self.rate))
            actual_len = len(current_llr)

            if actual_len < expected_len_approx:
                 print(f"警告: Batch {i}, 解码 LLR 长度 {actual_len} 小于预期近似长度 {expected_len_approx} (基于原始长度 {n_original} 和码率 {self.rate:.2f})。解码可能出错。")
                 # 尝试填充 LLR 到预期长度？这比较危险。或者截断 n_original？
                 # 暂时按实际 LLR 长度处理，后续切片需要更鲁棒
                 # pass

            # --- 分离 LLR ---
            # 需要根据编码时的结构精确分离，同时处理长度不匹配问题
            len_sys = n_original
            len_par1 = n_original
            len_par2 = n_original

            # 实际切片，使用 min 避免越界
            llr_sys = current_llr[:min(len_sys, actual_len)]
            llr_par1 = current_llr[min(len_sys, actual_len) : min(len_sys + len_par1, actual_len)]
            llr_par2 = current_llr[min(len_sys + len_par1, actual_len) : min(len_sys + len_par1 + len_par2, actual_len)]

            # 如果切片得到的长度不足 n_original，用 0 填充（表示无信息）
            llr_sys = np.pad(llr_sys, (0, max(0, n_original - len(llr_sys))))
            llr_par1 = np.pad(llr_par1, (0, max(0, n_original - len(llr_par1))))
            llr_par2 = np.pad(llr_par2, (0, max(0, n_original - len(llr_par2))))

            # 处理码率 1/2 的特殊情况（如果之前简化编码是这样做的）
            if self.rate == 1/2:
                 # 假设编码时只用了部分校验位，解码时需要将对应 LLR 填回
                 # 这是一个非常粗糙的模拟，实际 Turbo 码打孔模式需要精确对应
                 n_parity_half = n_original // 2
                 # 将提取到的 (可能不完整的) LLR 填充到零数组的前半部分
                 temp_par1 = np.zeros(n_original)
                 temp_par2 = np.zeros(n_original)
                 len_par1_half = min(n_parity_half, len(llr_par1)) # 实际提取到的长度
                 len_par2_half = min(n_parity_half, len(llr_par2))
                 temp_par1[:len_par1_half] = llr_par1[:len_par1_half]
                 temp_par2[:len_par2_half] = llr_par2[:len_par2_half]
                 llr_par1 = temp_par1
                 llr_par2 = temp_par2

            # --- 迭代解码 ---
            llr_apriori_2 = np.zeros(n_original) # 初始化第二次解码器的先验信息
            decoded_bits = np.zeros(n_original, dtype=int)

            for _ in range(self.iterations):
                # --- 第一解码器 ---
                # 输入 = 系统信道LLR + 来自第二解码器的先验信息 (解交织后)
                apriori_1 = self._deinterleave(llr_apriori_2)
                # 使用简化解码步骤获取外信息
                llr_extrinsic_1, _ = self._logmap_decode_step(llr_sys, llr_par1, apriori_1)

                # --- 第二解码器 ---
                # 输入 = 交织后的系统信道LLR + 来自第一解码器的先验信息 (交织后)
                apriori_2_interleaved = self._interleave(llr_extrinsic_1) # 交织第一解码器的外信息作为先验
                llr_sys_interleaved = self._interleave(llr_sys)
                llr_par2_interleaved = self._interleave(llr_par2) # 也需要对应交织 (虽然简化模型没严格用)
                # 使用简化解码步骤获取外信息
                llr_extrinsic_2_interleaved, _ = self._logmap_decode_step(llr_sys_interleaved, llr_par2_interleaved, apriori_2_interleaved)
                # 更新用于下一次迭代的先验信息 (需要解交织)
                llr_apriori_2 = self._deinterleave(llr_extrinsic_2_interleaved) # 外信息成为下一次迭代的先验

            # --- 最终判决 ---
            # 基于最后一次迭代的后验信息（近似）
            final_llr = llr_sys + self._deinterleave(llr_apriori_2) # 近似：系统 + 第二解码器的最后输出外信息(解交织后)
            decoded_bits = (final_llr < 0).astype(int) # 硬判决

            decoded_batches.append(decoded_bits)

        # --- 填充到最大原始长度 ---
        if not decoded_batches: return np.array([])
        # 处理可能的空数组
        non_empty_decoded = [dec for dec in decoded_batches if len(dec) > 0]
        if not non_empty_decoded: return np.zeros((batch_size, 0), dtype=int)

        max_original_len = max(len(dec) for dec in non_empty_decoded)

        padded_decoded = []
        for dec in decoded_batches:
            pad_width = max_original_len - len(dec)
            padded = np.pad(dec, (0, pad_width), mode='constant', constant_values=0)
            padded_decoded.append(padded)

        return np.array(padded_decoded, dtype=int)

class RSCoder(ChannelCoder):
    """
    Reed-Solomon编解码（使用 galois 库或简化实现）

    参数:
        n: 码字长度(GF(2^8)下，n <= 255)
        k: 信息符号长度(k < n)
    """
    def __init__(self, n: int = 255, k: int = 223):
        if n > 255 or k >= n:
            raise ValueError("对于 GF(2^8) 的 RS 码, 必须满足 k < n <= 255")
        self.n = n # 码字长度 (符号数)
        self.k = k # 信息符号数
        self.t = (n - k) // 2 # 可纠正的符号错误数
        self._current_original_bit_lengths = [] # 存储编码时的原始比特长度

        # 尝试导入并初始化 galois 库
        self.gf = None
        self.rs_coder = None
        self._galois_available = False
        if _GALOIS_INSTALLED: # 使用顶层检查结果
            try:
                self.gf = galois.GF(2**8) # 定义有限域 GF(2^8)
                # 创建 RS 编码器实例
                self.rs_coder = galois.ReedSolomon(n, k, field=self.gf)
                self._galois_available = True
                print("信息: Galois 库已找到，将使用 Galois 实现 RS 编解码。")
            except Exception as e: # 捕捉可能的初始化错误
                print(f"警告: 初始化 Galois RS 编码器失败: {e}")
                print("     RS 编解码将退回到极度简化的占位符实现。")
        else:
            print("警告: Galois 库未安装。RS 编解码将使用极度简化的占位符实现，无法纠错。")
            print("     请运行 'pip install galois' 来安装 Galois 库以获得真实的 RS 编解码功能。")

    # ... （_gf_mul, _polynomial_division, _reed_solomon_encode_simplified 省略，因为它们仅用于占位符）
    def _reed_solomon_encode_simplified(self, data: np.ndarray) -> np.ndarray:
         """Reed-Solomon 编码（极度简化 - 仅用于占位符，无纠错能力）"""
         codeword = np.zeros(self.n, dtype=np.uint8)
         data_len = len(data)
         if data_len > self.k:
             print(f"警告: RS 简化编码输入数据长度 {data_len} > k={self.k}，将截断。")
             data = data[:self.k]
             data_len = self.k
         codeword[:data_len] = data # 系统码部分
         return codeword

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Reed-Solomon 编码 (比特 -> 符号)

        参数:
            bits: 输入比特，形状 [batch_size, bit_length]

        返回:
            编码后的字节（符号），形状 [batch_size, encoded_symbol_length] (填充到批次内最大长度)
        """
        batch_size = bits.shape[0]
        encoded_symbol_batches = []
        original_bit_lengths = [] # 记录原始比特长度
        self._current_original_bit_lengths = [] # 清空旧记录

        for i in range(batch_size):
            current_bits = bits[i]
            original_bit_lengths.append(len(current_bits))

            # 1. 将比特流转换为字节流 (GF(2^8) 符号)
            num_bits = len(current_bits)
            if num_bits == 0:
                encoded_symbol_batches.append(np.array([], dtype=np.uint8))
                continue

            num_padding_bits = (8 - (num_bits % 8)) % 8
            # 确保填充后能被8整除
            padded_bits = np.pad(current_bits, (0, num_padding_bits), constant_values=0).astype(np.uint8)

            # 将比特重新塑形为字节 (使用 np.packbits)
            # np.packbits 需要一个扁平数组或最后一维是8
            # 先 reshape 成 [N/8, 8]，再 packbits
            try:
                 # np.packbits 需要明确的 uint8 类型
                 symbols_packed = np.packbits(padded_bits.reshape(-1, 8), axis=1).flatten()
                 int_symbols = symbols_packed
            except ValueError as e:
                 print(f"错误: Batch {i}, 比特流长度 {len(padded_bits)} 无法整形为 N/8 x 8. Error: {e}")
                 # 可能输入 current_bits 就有问题？或者填充逻辑错误？
                 # 暂时跳过此样本
                 encoded_symbol_batches.append(np.array([], dtype=np.uint8))
                 continue


            # 2. 分段进行 RS 编码
            encoded_segments = []
            num_symbols = len(int_symbols)
            for j in range(0, num_symbols, self.k):
                segment = int_symbols[j : j + self.k]
                # 如果段长度小于 k，需要填充 0 符号 (确保输入给编码器的是 k 个符号)
                if len(segment) < self.k:
                    segment = np.pad(segment, (0, self.k - len(segment)), constant_values=0)

                # 使用 Galois 库编码或简化编码
                if self._galois_available:
                    try:
                        # 使用 Galois 库进行编码
                        # 确保输入类型正确
                        encoded_segment = self.rs_coder.encode(segment.astype(self.gf.dtypes[0]))
                    except Exception as e_galois_enc:
                        print(f"警告: Galois RS 编码失败 at Batch={i}, Segment={j//self.k}: {e_galois_enc}")
                        # 退化到简化编码
                        encoded_segment = self._reed_solomon_encode_simplified(segment)
                else:
                    # 使用极度简化的占位符编码
                    encoded_segment = self._reed_solomon_encode_simplified(segment)
                encoded_segments.append(encoded_segment)

            # 3. 合并所有编码段
            if encoded_segments:
                encoded_symbols = np.concatenate(encoded_segments)
                encoded_symbol_batches.append(encoded_symbols)
            else:
                # 如果输入比特流为空，则输出空数组
                 encoded_symbol_batches.append(np.array([], dtype=np.uint8))

        # --- 填充到最大符号长度 ---
        if not encoded_symbol_batches: return np.array([])
        non_empty_batches = [enc for enc in encoded_symbol_batches if len(enc) > 0]
        if not non_empty_batches: return np.zeros((batch_size, 0), dtype=np.uint8)

        max_len = max(len(enc) for enc in non_empty_batches)

        padded_encoded_symbols = []
        for enc_sym in encoded_symbol_batches:
            pad_width = max_len - len(enc_sym)
            padded = np.pad(enc_sym, (0, pad_width), mode='constant', constant_values=0)
            padded_encoded_symbols.append(padded)

        # 存储原始比特长度信息，用于解码恢复
        self._current_original_bit_lengths = original_bit_lengths

        return np.array(padded_encoded_symbols, dtype=np.uint8) # 输出是符号(字节)

    def _reed_solomon_decode_simplified(self, received_symbols: np.ndarray) -> Tuple[np.ndarray, bool]:
        """极度简化的RS解码 (占位符) - 仅返回信息部分，无法纠错"""
        decoded_symbols = received_symbols[:self.k] # 假设信息在前 k 个符号
        decode_successful = True # 简化版总是“成功”
        return decoded_symbols, decode_successful

    def decode(self, received_signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Reed-Solomon 解码 (符号 -> 比特) - 接收硬判决符号

        参数:
            received_signal: 接收到的硬判决符号（字节），形状 [batch_size, encoded_symbol_length]。
            **kwargs: 预期包含 'original_bit_lengths' 列表，指示每个样本原始的比特数。

        返回:
            解码后的比特流，形状 [batch_size, original_bit_length] (填充到批次内最大长度)
        """
        received_symbols = received_signal # 输入是硬判决符号(字节)
        batch_size = received_symbols.shape[0]
        decoded_bit_batches = []

        # 获取编码时记录的原始比特长度
        original_bit_lengths = getattr(self, '_current_original_bit_lengths', [])
        if len(original_bit_lengths) != batch_size:
            print(f"警告: RS解码时原始比特长度信息丢失或不匹配 ({len(original_bit_lengths)} vs batch {batch_size})。")
            # 尝试基于 n, k 粗略估计，但这通常不准确，可能导致末尾比特错误
            num_received_syms = received_symbols.shape[1]
            if num_received_syms > 0 and self.n > 0:
                num_blocks = int(np.ceil(num_received_syms / self.n))
                estimated_info_symbols = num_blocks * self.k
                estimated_info_bits = estimated_info_symbols * 8
                original_bit_lengths = [estimated_info_bits] * batch_size
                print(f"     使用估计的原始比特长度: {estimated_info_bits} (可能不准确)")
            else:
                original_bit_lengths = [0] * batch_size # 如果无法估计

        for i in range(batch_size):
            current_received_symbols = received_symbols[i].astype(np.uint8) # 确保是 uint8
            original_length_bits = original_bit_lengths[i] if i < len(original_bit_lengths) else 0

            # --- 分段解码 ---
            decoded_symbol_segments = []
            total_decode_successful = True # 跟踪整个序列是否解码成功

            num_symbols_received = len(current_received_symbols)
            if num_symbols_received == 0 and original_length_bits > 0:
                print(f"警告: Batch {i}, 收到空符号序列，但预期原始长度 {original_length_bits} > 0。输出空比特序列。")
                decoded_bit_batches.append(np.array([], dtype=int))
                continue
            elif num_symbols_received == 0 and original_length_bits == 0:
                decoded_bit_batches.append(np.array([], dtype=int))
                continue

            for j in range(0, num_symbols_received, self.n):
                segment = current_received_symbols[j : j + self.n]
                segment_len = len(segment)

                # 如果段长度不足 n，可能是最后一个不完整的块
                if segment_len == 0: continue
                if segment_len < self.n:
                    # RS 解码器通常需要固定长度 n 的输入
                    # 用 0 填充到长度 n
                    segment = np.pad(segment, (0, self.n - segment_len), constant_values=0)

                # 使用 Galois 库解码或简化解码
                decoded_segment = np.array([], dtype=np.uint8) # 初始化
                decode_successful = False

                if self._galois_available:
                    try:
                        # Galois 解码，返回解码后的信息符号 (k个)
                        # 确保输入类型正确
                        decoded_segment = self.rs_coder.decode(segment.astype(self.gf.dtypes[0]))
                        decode_successful = True
                    except galois.errors.ReedSolomonError as e:
                        # 解码失败 (错误数超过 t)
                        # print(f"信息: RS解码失败 (Galois) at Batch={i}, Segment={j//self.n}. Error: {e}")
                        # 解码失败，输出 k 个零符号作为占位符
                        decoded_segment = np.zeros(self.k, dtype=np.uint8)
                        decode_successful = False
                        total_decode_successful = False
                    except TypeError as te: # 捕获可能的类型错误
                        print(f"错误: RS解码时发生TypeError (Galois) at Batch={i}, Segment={j//self.n}: {te}")
                        decoded_segment = np.zeros(self.k, dtype=np.uint8)
                        decode_successful = False
                        total_decode_successful = False
                    except Exception as e_other: # 捕获其他可能的 Galois 错误
                        print(f"错误: RS解码时发生未知错误 (Galois) at Batch={i}, Segment={j//self.n}: {e_other}")
                        decoded_segment = np.zeros(self.k, dtype=np.uint8)
                        decode_successful = False
                        total_decode_successful = False
                else:
                    # 使用简化占位符解码
                    decoded_segment, decode_successful = self._reed_solomon_decode_simplified(segment)

                decoded_symbol_segments.append(decoded_segment)

            # --- 合并解码段并转换回比特 ---
            if decoded_symbol_segments:
                # 合并所有解码出的信息符号段 (每个段长 k)
                decoded_symbols = np.concatenate(decoded_symbol_segments)

                # 将解码后的符号（字节 uint8）转换回比特流
                try:
                    # 修复: 先将 Galois 字段数组转换为标准 NumPy 数组
                    if self._galois_available:
                        # 检查是否是 Galois 数组类型
                        if hasattr(decoded_symbols, 'view') and callable(getattr(decoded_symbols, 'view')):
                            decoded_symbols = decoded_symbols.view(np.ndarray)
                        
                    # np.unpackbits 要求输入是 uint8
                    # 输出是 [N_symbols * 8] 的比特数组
                    decoded_bits = np.unpackbits(decoded_symbols.astype(np.uint8)[:, np.newaxis], axis=1).flatten()
                except Exception as e_unpack:
                    print(f"错误: Batch {i}, 解码后符号转换回比特时出错: {e_unpack}")
                    # 发生错误，生成一个全零的比特序列作为后备
                    decoded_bits = np.zeros(original_length_bits, dtype=int)

                # --- 截断到原始比特长度 ---
                # 这是关键一步，因为编码时可能填充了比特和符号
                actual_decoded_bits_len = len(decoded_bits)
                if actual_decoded_bits_len < original_length_bits:
                    print(f"警告: RS解码后比特数 {actual_decoded_bits_len} 小于原始长度 {original_length_bits} (Batch={i})。末尾可能丢失数据。")
                    # 填充 0 到原始长度 (这可能不正确，但保持长度一致性)
                    decoded_bits = np.pad(decoded_bits, (0, original_length_bits - actual_decoded_bits_len))
                elif actual_decoded_bits_len > original_length_bits:
                    # 截断超出原始长度的多余比特 (这些通常是由于块填充产生的)
                    decoded_bits = decoded_bits[:original_length_bits]

                decoded_bit_batches.append(decoded_bits.astype(int)) # 确保是 int 类型
            else:
                # 如果没有解码出任何段（例如输入符号流为空）
                # 输出一个长度为 original_length_bits 的全零数组
                decoded_bit_batches.append(np.zeros(original_length_bits, dtype=int))


        # --- 填充到批次内最大原始比特长度 ---
        if not decoded_bit_batches: return np.array([])
        non_empty_decoded = [dec for dec in decoded_bit_batches if len(dec) > 0]
        if not non_empty_decoded: return np.zeros((batch_size, 0), dtype=int)

        max_original_bit_len = max(len(dec_bits) for dec_bits in non_empty_decoded)

        padded_decoded_bits = []
        for dec_bits in decoded_bit_batches:
            pad_width = max(0, max_original_bit_len - len(dec_bits)) # 确保 pad_width 非负
            padded = np.pad(dec_bits, (0, pad_width), mode='constant', constant_values=0)
            padded_decoded_bits.append(padded)

        return np.array(padded_decoded_bits, dtype=int) # 确保输出是整数比特