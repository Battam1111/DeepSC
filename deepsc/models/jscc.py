# deepsc/models/jscc.py - 完整的 JSCC 基线实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.power_norm import power_normalize

class BiLSTMModule(nn.Module):
    """
    双向 LSTM 模块，实现基于 JSCC[22] 的双向语言模型组件
    
    参数:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏状态维度
        num_layers: LSTM层数
        dropout: Dropout概率
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        参数:
            x: [batch, seq_len, input_size] 输入序列
            lengths: 可选的序列长度，用于打包
            
        返回:
            [batch, seq_len, hidden_size*2] 双向输出
        """
        if lengths is not None:
            # 打包填充的序列以提高效率
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            self.lstm.flatten_parameters()
            output_packed, _ = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            self.lstm.flatten_parameters()
            output, _ = self.lstm(x)
            
        return self.dropout(output)

class JSCC(nn.Module):
    """
    联合源-信道编码 (JSCC) 基线
    
    基于论文 [22] 的实现，使用BiLSTM进行文本处理
    
    参数:
        cfg: 配置字典，包含模型参数
    """
    def __init__(self, cfg):
        super().__init__()
        # 配置参数
        self.vocab_size = cfg['vocab_size']
        self.pad_idx = cfg['pad_idx']
        self.embed_dim = cfg.get('embed_dim', 256)
        self.hidden_dim = cfg.get('hidden_dim', 512)
        self.latent_dim = cfg['latent_dim']
        self.lstm_layers = cfg.get('lstm_layers', 2)
        self.dropout = cfg['dropout']
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_dim, padding_idx=self.pad_idx
        )
        
        # BiLSTM 编码器
        self.encoder = BiLSTMModule(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=self.dropout
        )
        
        # 信道编码器
        self.channel_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.latent_dim)
        )
        
        # 信道解码器
        self.channel_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2)
        )
        
        # BiLSTM 解码器
        self.decoder = BiLSTMModule(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=self.dropout
        )
        
        # 输出投影
        self.proj = nn.Linear(self.hidden_dim * 2, self.vocab_size)
        
    def _get_lengths_and_mask(self, x):
        """
        获取序列长度和掩码
        
        参数:
            x: [batch, seq_len] 输入序列
            
        返回:
            (lengths, mask) 长度和布尔掩码
        """
        # 计算每个序列的实际长度(非pad部分)
        mask = (x != self.pad_idx)
        lengths = mask.sum(dim=1)
        return lengths, mask
    
    def forward(self, src, n_var, channel, *, return_tx_rx=False):
        """
        端到端前向传播
        
        参数:
            src: [batch, seq_len] 输入文本序列
            n_var: 噪声方差
            channel: 信道模型
            return_tx_rx: 是否返回发送/接收的信道符号
            
        返回:
            词汇分布logits，或元组 (logits, 发送符号, 接收符号)
        """
        # 获取序列长度和掩码
        lengths, mask = self._get_lengths_and_mask(src)
        
        # 词嵌入
        embedded = self.embedding(src) * (self.embed_dim ** 0.5)  # 缩放嵌入
        
        # BiLSTM编码
        enc_output = self.encoder(embedded, lengths)
        
        # 信道编码
        tx = power_normalize(self.channel_encoder(enc_output))
        
        # 信道传输
        rx = channel(tx, n_var)
        
        # 信道解码
        dec_input = self.channel_decoder(rx)
        
        # BiLSTM解码
        dec_output = self.decoder(dec_input)
        
        # 投影到词汇表
        logits = self.proj(dec_output)
        
        # 只计算 <START> 之后的词预测
        # 对应于训练中，目标是预测除第一个token外的所有token
        logits = logits[:, :-1, :]  # 截去最后一个时间步
        
        if return_tx_rx:
            return logits, tx, rx
        else:
            return logits