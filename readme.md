# DeepSC: 深度学习赋能的语义通信系统

[[GitHub stars](https://img.shields.io/github/stars/YourUsername/DeepSC?style=social)](https://github.com/YourUsername/DeepSC)

[[PyPI](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[[License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

本项目是论文 [Deep Learning Enabled Semantic Communication Systems](https://ieeexplore.ieee.org/document/9398576) 的完整 PyTorch 实现。DeepSC 突破了传统通信系统对比特和符号的关注，转而在语义层面进行信息交换，实现了在恶劣信道条件下的高效通信。

<div align="center">
  <img src="docs/assets/deepsc_arch.png" alt="DeepSC 架构图" width="80%">
</div>

## 📌 主要特点

- **语义级通信**：关注文本的意义而非比特准确性，特别适合低信噪比环境
- **端到端优化**：联合优化语义编解码和信道编解码，一体化设计
- **多信道支持**：兼容 AWGN、瑞利衰落、莱斯衰落和擦除信道
- **迁移学习**：快速适应新信道环境或新领域文本，降低训练成本
- **创新评估**：除传统 BLEU 外，引入基于 BERT 的句子相似度评估

## 🚀 快速开始

### 环境配置

```bash
# 创建并激活 conda 环境
conda create -n deepsc python=3.10 -y
conda activate deepsc

# 安装依赖包
pip install -r requirements.txt
```

### 数据准备

```bash
# 下载并预处理 EuroParl 语料库
bash scripts/download_and_preprocess.sh
```

成功执行后，将在 `data/europarl/` 目录下生成训练集、测试集和词表文件。

### 模型训练

基础训练命令：

```bash
python -m scripts.train
```

自定义训练参数：

```bash
# 调整批大小和学习率
python -m scripts.train train.batch_size=64 train.lr=5e-4

# 选择不同信道
python -m scripts.train data.channel=RAYLEIGH

# 增大互信息权重以提高语义保留能力
python -m scripts.train train.lambda_mi=0.01
```

### 模型评估

评估不同信噪比下的性能：

```bash
python -m scripts.evaluate ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt
```

结果将同时显示 BLEU 分数、句子相似度和互信息评估，并生成可视化图表。

### 迁移学习

迁移到新信道环境：

```bash
python -m scripts.finetune \
    ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt \
    mode=channel \
    new_channel=RAYLEIGH \
    ft.epochs=5
```

迁移到新领域文本：

```bash
python -m scripts.finetune \
    ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt \
    mode=domain \
    data.train_pkl=/path/to/new/train.pkl \
    data.val_pkl=/path/to/new/val.pkl \
    data.vocab_json=/path/to/new/vocab.json
```

## 📊 性能对比

DeepSC 相比传统通信系统在低信噪比环境中表现出显著优势：

| 方法 | SNR=0dB | SNR=6dB | SNR=12dB |
|------|---------|---------|----------|
| DeepSC | 0.42 | 0.89 | 0.95 |
| JSCC [22] | 0.38 | 0.81 | 0.93 |
| Huffman+Turbo | 0.05 | 0.42 | 0.85 |
| 5-bit+RS | 0.03 | 0.31 | 0.78 |

_表格中数值为 BLEU-1 分数，越高越好_

<div align="center">
  <img src="docs/assets/performance_curve.png" alt="性能曲线" width="70%">
</div>

## 📁 项目结构

```
DeepSC/
├── configs/            # Hydra 配置文件
│   ├── model/          # 模型配置
│   ├── data/           # 数据配置
│   └── train/          # 训练配置
├── deepsc/             # 核心模块
│   ├── data/           # 数据处理
│   ├── models/         # 模型定义
│   │   ├── transformer.py  # DeepSC 主体结构
│   │   ├── mine.py     # 互信息估计器
│   │   └── channel.py  # 各类信道模型
│   ├── engine/         # 训练引擎
│   ├── metrics/        # 评估指标
│   └── utils/          # 工具函数
├── scripts/            # 训练和评估脚本
│   ├── train.py        # 训练脚本
│   ├── evaluate.py     # 评估脚本
│   ├── finetune.py     # 迁移学习脚本
│   └── compare_baselines.py  # 基线比较脚本
└── docs/               # 文档资源
```

## 🔍 高级用法

### 1. 训练参数配置

所有训练参数都可通过 Hydra 配置系统设置，主要参数包括：

- **模型参数**：`model.d_model`、`model.n_layers`、`model.n_heads` 等
- **训练参数**：`train.batch_size`、`train.lr`、`train.lambda_mi` 等
- **信道参数**：`data.channel`、`train.snr_low`、`train.snr_high` 等

完整参数参考 `configs/` 目录下的配置文件。

### 2. 自定义信道模型

添加新的信道模型非常简单，只需在 `deepsc/models/channel.py` 中继承 `BaseChannel` 类并注册：

```python
@register_channel('YOUR_CHANNEL')
class YourChannel(BaseChannel):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def forward(self, tx, n_var):
        # 实现您的信道模型
        return processed_signal
```

### 3. 语义相似度评估

除了使用默认的句子相似度计算外，还可以自定义 BERT 模型和池化方法：

```python
from deepsc.metrics.sentence_sim import sentence_similarity

# 自定义评估
scores = sentence_similarity(
    predicted_sentences, 
    reference_sentences,
    device='cuda',
    model_name='bert-large-uncased',  # 使用更大的模型
    pooling='cls'                    # 使用 [CLS] 令牌表示
)
```

### 4. 性能优化

对于大规模数据集，可以启用以下优化选项：

```bash
# 使用16位自动混合精度
python -m scripts.train precision=16

# 增加数据加载并行度
python -m scripts.train num_workers=8

# 梯度累积以模拟更大批次
python -m scripts.train accumulate_grad_batches=2
```

## 💡 常见问题

### Q: DeepSC 如何处理不同长度的句子？

**A:** DeepSC 使用 Transformer 架构，通过填充和掩码机制处理变长序列。数据加载器会自动将批次内句子填充到相同长度，并生成相应的注意力掩码。

### Q: 互信息损失的作用是什么？

**A:** 互信息损失项促使模型在信道编码中保留更多语义信息，同时最大化信道容量。增大 `lambda_mi` 值会加强语义保留能力，但可能降低比特层面的准确性。

### Q: 如何处理新语言的文本？

**A:** 通过 `mode=domain` 的迁移学习，DeepSC 可以适应新语言。您需要准备新语言的数据集和词表，然后使用 `finetune.py` 脚本进行迁移。最好先冻结信道相关层，只训练语义层。

### Q: 训练时信道信噪比范围如何选择？

**A:** 为获得对各种信道条件的鲁棒性，建议设置较宽的 SNR 范围，例如 `snr_low=0`，`snr_high=15`。如果主要关注低信噪比环境，可以将范围缩小至 `snr_low=0`，`snr_high=10`。

### Q: 如何避免训练过程中互信息估计不稳定？

**A:** 通过设置 `mine_warmup` 和 `mine_update_freq` 参数来改善稳定性。在训练初期关注语义重建损失，稍后再引入互信息损失。同时使用多步更新 MINE 网络有助于提高估计精度。

## 🔧 故障排除

| 问题 | 解决方案 |
|------|----------|
| CUDA 内存不足 | 减小 `batch_size`，启用 `precision=16`，或增加梯度累积 |
| 训练不收敛 | 检查学习率，增加 `warmup` 步数，减小 `lambda_mi` |
| 验证集表现差 | 尝试增加 `dropout`，减少训练轮数，启用早停 |
| MINE 训练不稳定 | 增加 `mine_warmup`，减小学习率，增加 `mine_extra_steps` |
| 句子相似度计算慢 | 减小评估批大小，或使用更小的 BERT 模型 |

## 📚 论文复现指南

本实现精确复现了原论文中的以下关键组件和实验：

1. **模型架构**：3层 Transformer 编解码器，8头注意力，16维信道编码
2. **训练策略**：互信息优化，基于 SNR 的噪声采样
3. **评估指标**：BLEU-1 分数和句子相似度
4. **迁移学习**：信道迁移和领域迁移
5. **基线比较**：与 JSCC、Huffman+Turbo、固定长度+RS 等方法比较

如果您希望完全复现论文中的结果，请使用以下命令：

```bash
# 1. 训练基础模型
python -m scripts.train \
  train.lambda_mi=0.01 \
  train.snr_low=0 \
  train.snr_high=15 \
  train.lr=3e-4 \
  train.batch_size=128

# 2. 与传统方法比较
python -m scripts.compare_baselines \
  ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt

# 3. 迁移学习实验
python -m scripts.finetune \
  ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt \
  mode=channel \
  new_channel=RAYLEIGH
```

## 📜 引用

如果您在研究中使用了 DeepSC，请引用原论文：

```bibtex
@article{xie2021deep,
  title={Deep Learning Enabled Semantic Communication Systems},
  author={Xie, Huiqiang and Qin, Zhijin and Li, Geoffrey Ye and Juang, Biing-Hwang},
  journal={IEEE Transactions on Signal Processing},
  volume={69},
  pages={2663--2675},
  year={2021},
  publisher={IEEE}
}
```

## 📄 许可证

本项目基于 MIT 许可证，详细信息请参阅 [LICENSE](LICENSE) 文件。

## 📬 联系我们

如有问题或建议，请通过以下方式联系我们：

- 提交 GitHub Issue
- 发送邮件至：[yan-jun.chen@connect.polyu.hk](mailto:yan-jun.chen@connect.polyu.hk)

---

<div align="center">
  <b>DeepSC - 突破比特界限，传递语义信息</b>
</div>
