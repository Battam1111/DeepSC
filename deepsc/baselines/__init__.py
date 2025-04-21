# deepsc/baselines/__init__.py
"""
传统基线方法包
=======================
包含与DeepSC比较的传统通信方法。
"""
# 导入注册，确保所有基线可用
from .traditional import HuffmanEncoder, FixedLengthEncoder
from .channel_coding import TurboCoder, RSCoder