# DeepSC 重现与使用指南

*Deep Learning Enabled Semantic Communication Systems*  
论文原作者：Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, Biing‑Hwang Juang  
本仓库：对论文所有实验的 **完整 PyTorch 复现**（含迁移学习）。

---

## 目录

1. [环境依赖](#环境依赖)  
2. [数据准备](#数据准备)  
3. [快速上手](#快速上手)  
   * 3.1 训练（AWGN）  
   * 3.2 评估（BLEU / SentenceSim / MI 曲线）  
   * 3.3 迁移学习  
   * 3.4 单文件推理  
4. [项目结构](#项目结构)  
5. [常见问题](#常见问题)  
6. [引用](#引用)  

---

## 环境依赖

| 组件 | 版本（或以上） |
|------|---------------|
| Python | 3.9 / 3.10 |
| PyTorch | 2.0 |
| PyTorch‑Lightning | 2.2 |
| CUDA | 11.7+ (若使用 GPU) |

> ### 一键安装脚本
> ```bash
> # 创建隔离环境（可改成 mamba / venv）
> conda create -n deepsc python=3.10 -y
> conda activate deepsc
>
> # 安装依赖
> pip install -r requirements.txt
> ```

依赖列表见 `requirements.txt`。如需 CPU 版本，请将 `torch>=2.0` 改为官方发布的 `torch==2.0.1+cpu` 等对应包名。

---

## 数据准备

论文使用 **EuroParl 英语语料**（约 2 M 句）。以下脚本将自动下载、切分、构建词表并生成二进制 `pkl`：

```bash
bash scripts/download_and_preprocess.sh
```

执行后目录结构应类似：

```
data/
└── europarl/
    ├── train_data.pkl   # 训练集合 (pickle)
    ├── test_data.pkl    # 验证/测试集合
    └── vocab.json       # 词表 (含 <PAD>/<START>/<END>/<UNK>)
```

如要使用自己的语料，将对应文件路径写入 `configs/data/*.yaml`，或在 CLI 中覆盖。

---

## 快速上手

### 3.1 训练（默认 AWGN 信道）

```bash
python -m scripts.train
```

* 重要参数均可在 CLI 覆盖，例如：
  ```bash
  python -m scripts.train train.batch_size=64 model.d_model=512
  ```
* 训练日志与模型权重保存在 `lightning_logs/`。  
* 最优模型以 `best-epoch=xx-val_bleu=xxx.ckpt` 命名。

### 3.2 评估

生成 SNR ∈ {0, 3, 6, 9, 12, 15, 18} dB 下的三条曲线：

```bash
python -m scripts.evaluate \
    ckpt_path=lightning_logs/version_0/checkpoints/best*.ckpt
```

输出示例：

```
=== Evaluation Result ===
SNR(dB)           : [0, 3, 6, 9, 12, 15, 18]
BLEU‑1            : ['0.4200', '0.7235', '0.8852', ...]
SentenceSimilarity: ['0.5102', '0.9013', '0.9431', ...]
MI‑LB             : ['0.3314', '0.8325', '1.2280', ...]
```

### 3.3 迁移学习

#### a) 换信道（AWGN → Rician）

```bash
python -m scripts.finetune \
    ckpt_path=lightning_logs/.../best.ckpt \
    mode=channel \
    new_channel=Rician \
    ft.epochs=5
```

#### b) 换领域语料

```bash
python -m scripts.finetune \
    ckpt_path=lightning_logs/.../best.ckpt \
    mode=domain \
    data.train_pkl=/path/med/train.pkl \
    data.val_pkl=/path/med/val.pkl \
    data.vocab_json=/path/med/vocab.json
```

### 3.4 单文件推理

```bash
python -m scripts.inference \
    ckpt_path=lightning_logs/.../best.ckpt \
    mode=beam  snr=6
```

---

## 项目结构

```
DeepSC/
├── configs/            # Hydra 配置树
├── deepsc/             # 包含所有核心模块
│   ├── data/           # Dataset & Vocab
│   ├── engine/         # Lightning 封装
│   ├── metrics/        # BLEU / MI / SentenceSim
│   ├── models/         # Transformer + Channel + MINE
│   └── utils/          # 辅助函数
├── scripts/            # 训练/评估/迁移/推理 CLI
├── tests/              # pytest 单元测试
└── requirements.txt
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| **显存不足** | 将 `train.batch_size` 调小；或将 `precision` 设为 16（AMP）。 |
| **BERT 评估慢** | 评估阶段使用 `sentence_similarity` 时自动缓存模型；也可将 `batch_size` 调大。 |
| **MI 曲线为 0** | 确认 `train.lambda_mi` ≥ 0.01；训练日志中 `train_mi_lb` 应在 0.8~1.3 之间波动。 |
| **自定义信道** | 在 `deepsc/models/channel.py` 中继承 `BaseChannel` 并用 `@register_channel('NAME')` 装饰即可。 |

---

## 引用

若您在研究中使用了本仓库，请引用原论文：

```text
H. Xie, Z. Qin, G. Y. Li, and B.‑H. Juang,
"Deep Learning Enabled Semantic Communication Systems,"
IEEE Trans. Signal Processing, vol. 69, pp. 2663‑2675, 2021.
```

---

Happy research & coding!