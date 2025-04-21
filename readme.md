
-----

# DeepSC: 深度学习赋能的语义通信系统

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://www.google.com/search?q=LICENSE)

本项目提供了论文 [Deep Learning Enabled Semantic Communication Systems](https://ieeexplore.ieee.org/document/9398576) 的 PyTorch 实现。DeepSC 旨在突破传统通信系统关注比特传输的限制，转向在语义层面进行信息交换，尤其在低信噪比（SNR）等恶劣信道条件下展现出高效和鲁棒的通信能力。

\<div align="center"\>
\<img src="docs/assets/deepsc\_arch.png" alt="DeepSC 架构图" width="80%"\>
\</div\>

## 📌 主要特点

  - **语义通信**: 直接传输文本的语义信息，而非精确的比特序列，显著提升低信噪比下的通信效果。
  - **端到端优化**: 联合优化语义编解码器和信道编解码器。
  - **两种训练模式**:
      - **标准版 (Lightning)**: 使用 PyTorch Lightning 实现，采用灵活的交替训练策略。
      - **严格版 (Two-Phase)**: 严格遵循论文提出的两阶段训练流程（先训练 MINE，再训练主网络），用于精确复现。
  - **多信道支持**: 内置支持 AWGN、瑞利衰落、莱斯衰落和擦除信道，并易于扩展。
  - **迁移学习**: 支持快速将预训练模型适应到新的信道环境或新的文本领域。
  - **先进评估指标**: 除了传统的 BLEU 分数，还引入基于 BERT 的句子相似度来更准确地评估语义保真度。

## 🚀 快速开始

### 1\. 环境配置

```bash
# 创建并激活 conda 环境 (建议 Python 3.8+)
conda create -n deepsc python=3.10 -y
conda activate deepsc

# 安装依赖
pip install -r requirements.txt
```

### 2\. 数据准备

```bash
# 下载并预处理 EuroParl 语料库
bash scripts/download_and_preprocess.sh
```

成功执行后，将在 `data/europarl/` 目录下生成 `train_data.pkl`, `test_data.pkl`, 和 `vocab.json` 文件。

### 3\. 模型训练

本项目提供两种训练方式：

#### a) 标准版本 (Lightning 交替训练)

使用 PyTorch Lightning 进行训练，每个批次交替训练 MINE 和主网络。

```bash
# 使用默认配置 (AWGN信道) 开始训练
python -m scripts.train

# 自定义训练参数 (示例)
# 调整批大小和学习率
python -m scripts.train train.batch_size=64 train.lr=5e-4
# 选择不同信道
python -m scripts.train data.channel=RAYLEIGH train.channel.name=RAYLEIGH # 同时修改数据和训练配置中的信道
# 增大互信息权重
python -m scripts.train train.lambda_mi=0.01
```

训练日志和模型检查点将保存在 `lightning_logs/` 目录下。

#### b) 严格版本 (论文两阶段训练)

严格按照论文描述的两阶段流程进行训练。

```bash
# 使用两阶段训练方法 (默认AWGN信道)
python -m scripts.train_phase

# 自定义训练参数 (示例)
# 调整 MINE 预训练轮数和主网络训练轮数
python -m scripts.train_phase train.mine_epochs=10 train.epochs=30
# 选择不同信道
python -m scripts.train_phase data.channel=RAYLEIGH train.channel.name=RAYLEIGH
# 调整互信息权重
python -m scripts.train_phase train.lambda_mi=0.01
```

此方法训练的模型检查点默认保存在 `checkpoints/` 目录下。

### 4\. 模型评估

评估模型在不同信噪比下的性能。

```bash
# 评估标准版本 (Lightning) 训练的模型
# 将 version_X 替换为实际的日志版本号
python -m scripts.evaluate ckpt_path=lightning_logs/version_X/checkpoints/best*.ckpt

# 评估严格版本 (Two-Phase) 训练的模型
# 将 epochX 替换为实际的最佳模型轮数
python -m scripts.evaluate ckpt_path=checkpoints/best_model_epochX.pt train.strict_model=True # 使用 train.strict_model=True 标志
```

评估脚本会计算并显示 BLEU 分数、句子相似度，并可选地生成性能曲线图。

### 5\. 迁移学习

#### a) 迁移到新信道

```bash
# 假设从 AWGN 迁移到 Rayleigh 信道
python -m scripts.finetune \
    ckpt_path=/path/to/pretrained/checkpoint.ckpt \
    mode=channel \
    train.new_channel=RAYLEIGH \
    ft.epochs=5 # 迁移学习的轮数

# 注意：如果原始模型是严格版本，也需添加 train.strict_model=True
```

#### b) 迁移到新领域文本

```bash
python -m scripts.finetune \
    ckpt_path=/path/to/pretrained/checkpoint.ckpt \
    mode=domain \
    data.train_pkl=/path/to/new/domain/train.pkl \
    data.val_pkl=/path/to/new/domain/val.pkl \
    data.vocab_json=/path/to/new/domain/vocab.json \
    ft.epochs=10 # 迁移学习的轮数

# 注意：如果原始模型是严格版本，也需添加 train.strict_model=True
```

## 📊 性能对比

DeepSC 在低信噪比环境下相较于传统方法（如 Huffman+Turbo 编码）和一些基于深度学习的联合信源信道编码（JSCC）方法表现出显著优势。

| 方法          | SNR=0dB | SNR=6dB | SNR=12dB |
| ------------- | ------- | ------- | -------- |
| DeepSC (本文实现) | 0.42    | 0.89    | 0.95     |
| JSCC \[22]     | 0.38    | 0.81    | 0.93     |
| Huffman+Turbo | 0.05    | 0.42    | 0.85     |
| 5-bit+RS      | 0.03    | 0.31    | 0.78     |

*表格中数值为 BLEU-1 分数，越高越好。结果基于论文报告，实际复现可能略有差异。*

\<div align="center"\>
\<img src="docs/assets/performance\_curve.png" alt="性能曲线" width="70%"\>
\</div\>
*性能曲线示意图，展示了 DeepSC 在不同 SNR 下的 BLEU 分数优势。*

## 📁 项目结构

```
DeepSC/
├── configs/                # Hydra 配置文件目录
│   ├── base.yaml           # 基础配置
│   ├── data/               # 数据配置
│   │   └── europarl.yaml   # EuroParl数据集配置
│   ├── ft/                 # 迁移学习配置
│   │   └── base.yaml       # 迁移学习基础配置
│   ├── infer/              # 推理配置
│   │   └── default.yaml    # 默认推理配置
│   ├── model/              # 模型配置
│   │   ├── deepsc.yaml     # DeepSC配置 (标准版)
│   │   ├── deepsc_s.yaml   # DeepSC语音版配置 (若支持)
│   │   └── jscc.yaml       # JSCC基线配置
│   └── train/              # 训练配置
│       ├── awgn.yaml       # AWGN信道训练配置
│       ├── base.yaml       # 训练基础配置 (包含strict_model等)
│       └── channel/        # 信道参数配置 (AWGN, Rayleigh等)
│           └── awgn.yaml
├── data/                   # 数据目录 (自动下载或处理后存放)
│   └── europarl/           # EuroParl数据集
│       ├── train_data.pkl  # 训练集 (处理后)
│       ├── test_data.pkl   # 测试集 (处理后)
│       └── vocab.json      # 词表
├── deepsc/                 # 核心代码目录
│   ├── baselines/          # 传统基线方法实现
│   │   ├── channel_coding.py # 信道编码 (Turbo, RS)
│   │   └── traditional.py    # 传统源编码 (Huffman)
│   ├── data/               # 数据加载与处理
│   │   ├── europarl.py     # EuroParl数据加载器
│   │   └── vocab.py        # 词表处理
│   ├── decoding/           # 解码策略
│   │   └── beam_search.py  # 束搜索解码
│   ├── engine/             # 训练与评估引擎
│   │   ├── callbacks.py    # Pytorch Lightning 回调
│   │   ├── lit_module.py   # Lightning 核心模块 (标准版)
│   │   └── trainer.py      # 两阶段训练器 (严格版)
│   ├── metrics/            # 评估指标
│   │   ├── bleu.py         # BLEU 评分计算
│   │   ├── mi.py           # 互信息计算 (使用 MINE)
│   │   └── sentence_sim.py # 基于 BERT 的句子相似度
│   ├── models/             # 模型定义
│   │   ├── channel.py      # 信道模型 (AWGN, Rayleigh等)
│   │   ├── deepsc_s.py     # 语音版DeepSC模型 (若支持)
│   │   ├── deepsc_strict.py# 严格版DeepSC模型架构
│   │   ├── jscc.py         # JSCC基线模型
│   │   ├── mine.py         # MINE网络 (用于标准版)
│   │   ├── mine_strict.py  # 严格版 MINE 网络
│   │   ├── registry.py     # 模型/信道注册器
│   │   └── transformer.py  # DeepSC 主体 Transformer 结构 (标准版共享)
│   └── utils/              # 工具函数
│       ├── freeze.py       # 参数冻结/解冻工具
│       ├── mask.py         # Transformer 掩码生成
│       ├── power_norm.py   # 发射信号功率归一化
│       └── seed.py         # 随机种子设置
├── docs/                   # 文档与资源
│   └── assets/             # 图片等资源
│       ├── deepsc_arch.png     # 架构图
│       └── performance_curve.png # 性能曲线图
├── scripts/                # 可执行脚本
│   ├── compare_baselines.py # 与基线方法比较性能
│   ├── evaluate.py         # 模型评估脚本
│   ├── finetune.py         # 迁移学习脚本
│   ├── inference.py        # 模型推理脚本
│   ├── train.py            # 标准版 (Lightning) 训练脚本
│   ├── train_phase.py      # 严格版 (Two-Phase) 训练脚本
│   ├── train_pt.py         # (可选) PyTorch 原生训练示例脚本
│   └── download_and_preprocess.sh # 数据下载与预处理脚本
├── tests/                  # 测试代码
│   ├── test_channels.py    # 信道模型测试
│   ├── test_dataset.py     # 数据加载测试
│   ├── test_freeze.py      # 参数冻结测试
│   └── test_models.py      # 模型构建与前向传播测试
├── LICENSE                 # 项目许可证 (MIT)
├── readme.md               # 本文档
└── requirements.txt        # Python 依赖库列表
```

## 🔧 进阶用法

### 1\. 配置系统 (Hydra)

本项目使用 [Hydra](https://hydra.cc/) 进行配置管理。所有参数（模型结构、训练参数、数据路径、信道类型等）都定义在 `configs/` 目录下的 `.yaml` 文件中。

  - **覆盖参数**: 你可以在命令行中轻松覆盖任何配置项。
    ```bash
    # 示例：修改模型层数和训练时的 SNR 范围
    python -m scripts.train model.n_layers=4 train.snr_low=0 train.snr_high=10
    ```
  - **选择配置**: 可以通过 `+` 号添加或切换配置文件片段。
    ```bash
    # 示例：切换到瑞利信道训练配置 (如果存在 configs/train/rayleigh.yaml)
    # python -m scripts.train +train=rayleigh
    ```
  - **严格版本参数**: 严格版本的特定参数（如 `mine_epochs`, `mine_lr`）位于 `configs/train/base.yaml` 或相关训练配置文件中，可通过 `train.` 前缀访问。

### 2\. 自定义信道模型

在 `deepsc/models/channel.py` 中添加新的信道模型：

1.  继承 `BaseChannel` 类。
2.  实现 `__init__` 和 `forward` 方法。
3.  使用 `@register_channel('YOUR_CHANNEL_NAME')` 装饰器注册。
4.  在 `configs/train/channel/` 目录下添加对应的配置文件 `your_channel_name.yaml`。
5.  训练时通过 `train.channel.name=YOUR_CHANNEL_NAME` 来选择。

### 3\. 自定义句子相似度评估

可以修改 `scripts/evaluate.py` 或直接调用 `deepsc.metrics.sentence_sim.sentence_similarity` 函数，并指定不同的 `model_name` (Hugging Face 模型) 或 `pooling` 方法 (`'mean'`, `'cls'`)。

### 4\. 性能优化

对于大规模训练，考虑：

  - **混合精度训练**: `python -m scripts.train trainer.precision=16` (需要 PyTorch 支持 AMP)
  - **增加数据加载进程**: `python -m scripts.train data.num_workers=8`
  - **调整批大小**: 根据 GPU 显存调整 `train.batch_size`。

## 💡 复现论文结果的最佳实践

为确保完全复现论文中报告的结果，请严格按照以下步骤操作：

### 1. 数据准备

首先下载并预处理数据集：

```bash
# 下载并预处理欧洲议会数据集
bash scripts/download_and_preprocess.sh
```

### 2. 严格版两阶段训练（AWGN信道）

使用严格版本的两阶段训练方法，完全按照论文描述的流程：

```bash
# 严格两阶段训练 - AWGN信道
python -m scripts.train_phase
```

训练完成后，将在 `checkpoints/` 目录生成最佳模型检查点，形如 `best_model_epochX.pt`。

### 3. 在AWGN信道上评估模型

```bash
python -m scripts.evaluate \
    ckpt_path=checkpoints_phase/best_model.pt \
    strict_model=True \
    data.channel=AWGN
```

### 4. 迁移到瑞利信道（论文第二个实验）

使用在AWGN上训练的模型，迁移到瑞利信道：

```bash
python -m scripts.finetune \
    ckpt_path=checkpoints_phase/best_model.pt \
    mode=channel \
    new_channel=RAYLEIGH \
    strict_model=True \
    ft.epochs=5
```

迁移学习完成后，将生成新的检查点，如 `finetune_channel_RAYLEIGH_ckpts/ft-epochY-bleuZ.pt`。

### 5. 在瑞利信道上评估迁移模型

```bash
# 将 Y 和 Z 替换为实际的迁移模型轮数和BLEU分数
python -m scripts.evaluate \
    ckpt_path=finetune_channel_RAYLEIGH_ckpts/ft-epochY-bleuZ.pt \
    strict_model=True \
    data.channel=RAYLEIGH
```

### 6. 与传统方法进行比较

为完全复现论文中的对比实验，执行：

```bash
# 比较 AWGN 信道上的性能
python -m scripts.compare_baselines \
    ckpt_path=checkpoints_phase/best_model.pt \
    strict_model=True \
    data.channel=AWGN

# 比较 Rayleigh 信道上的性能
python -m scripts.compare_baselines \
    ckpt_path=finetune_channel_RAYLEIGH_ckpts/ft-epochY-bleuZ.pt \
    strict_model=True \
    data.channel=RAYLEIGH
```

### 7. 验证结果

检查生成的评估结果CSV文件和性能曲线，确认SNR=6dB时BLEU分数约为0.89，SNR=12dB时约为0.95，与论文报告一致。曲线应该显示DeepSC相比传统方法在低SNR区域（0-9dB）具有明显优势。

### 8. 域迁移实验（可选，复现论文第三个实验）

如需复现论文中提到的域迁移实验，请准备新的领域数据集，然后执行：

```bash
# 将 X 替换为AWGN上训练的最佳模型轮数
python -m scripts.finetune \
    ckpt_path=checkpoints/best_model_epochX.pt \
    mode=domain \
    data.train_pkl=/path/to/new/domain/train.pkl \
    data.val_pkl=/path/to/new/domain/val.pkl \
    data.vocab_json=/path/to/new/domain/vocab.json \
    strict_model=True \
    ft.epochs=10
```

以上所有命令都可以通过添加 `trainer.precision=16` 参数来启用混合精度训练，加速训练过程。


## ❓ 常见问题 (FAQ)

  * **Q: 严格版本 (train\_phase.py) 和标准版本 (train.py) 有什么核心区别？**
      * **A:** 严格版本严格按论文分为两个阶段：先独立训练 MINE 网络，然后冻结 MINE 并训练 DeepSC 主网络。标准版本使用 Lightning，在每个训练步骤中交替更新 MINE 和主网络，代码更简洁，训练可能更稳定，但与论文流程有差异。复现论文结果推荐使用严格版本。
  * **Q: 如何确保 MINE 估计的互信息准确？**
      * **A:** 在严格版本中，通过 `train.mine_epochs` 参数控制 MINE 的预训练轮数，确保其充分收敛。在标准版本中，调整 MINE 的学习率 (`train.mine_lr`，如果与主网络不同) 和更新频率可能有助于提高稳定性。监控训练过程中的互信息估计值也是一个方法。
  * **Q: 如何用于新语言或不同领域的文本？**
      * **A:** 使用迁移学习 (`scripts/finetune.py`) 的 `mode=domain` 功能。你需要准备新语言/领域的数据集（`.pkl` 格式）和词表（`.json` 格式），然后运行脚本进行微调。通常会冻结部分底层网络（如信道编译码器），只训练与语义相关的层。
  * **Q: 训练时的信噪比 (SNR) 范围如何选择？**
      * **A:** 为了让模型对变化的信道条件具有鲁棒性，建议使用较宽的范围，如 `train.snr_low=0`, `train.snr_high=15` (单位 dB)。如果特别关注低信噪比性能，可以将范围下移，如 `train.snr_low=-5`, `train.snr_high=10`。

## 🔧 故障排除

| 问题                     | 可能的解决方案                                                                                                                               |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| CUDA 内存不足 (OOM)      | 减小 `train.batch_size`；启用混合精度 `trainer.precision=16`；减小模型维度 `model.d_model` 或序列长度 `data.max_len` (如果可配置)。                 |
| 训练不收敛/损失爆炸       | 检查学习率 `train.lr` 是否过大；尝试增加学习率预热步数 `train.warmup_steps`；检查数据预处理是否正确；对于严格版本，确保 MINE 预训练充分 (`train.mine_epochs`)。 |
| MINE 训练不稳定/损失为 NaN | 尝试降低 MINE 的学习率 (`train.mine_lr`)；检查 MINE 网络结构；确保批大小不是太小；检查输入 MINE 的数据范围是否合适。                           |
| 句子相似度计算缓慢       | 评估时减小批大小 `infer.batch_size`；在 `scripts/evaluate.py` 中选择更轻量级的 BERT 模型 (`model_name`)；使用 GPU 进行评估。                      |
| `ModuleNotFoundError`    | 确保所有依赖项已通过 `pip install -r requirements.txt` 安装；检查 Python 环境是否正确激活 (`conda activate deepsc`)。                         |
| Hydra 配置错误           | 仔细检查命令行覆盖的参数名是否正确；确认 YAML 文件格式无误（缩进等）；检查 `+` 语法是否用于存在的配置文件。                                            |

## 📚 引用

如果你在研究中使用了本项目的代码或 DeepSC 的思想，请引用原始论文：

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

本项目基于 MIT 许可证。详情请参阅 [LICENSE](https://www.google.com/search?q=LICENSE) 文件。

-----