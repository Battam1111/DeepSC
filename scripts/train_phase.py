# -*- coding: utf-8 -*-
"""
两阶段训练脚本
=============
严格按照论文实现DeepSC的两阶段训练流程：
1. 第一阶段：训练互信息估计网络(MINE)
2. 第二阶段：固定MINE，训练DeepSC主网络

用法示例:
    python -m scripts.train_phase

或指定配置:
    python -m scripts.train_phase train.batch_size=128 train.lr=3e-4
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pathlib import Path
import torch
import numpy as np
import os

from deepsc.utils.seed import set_global_seed
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.models.deepsc_strict import DeepSCStrict
from deepsc.models import get_channel
from deepsc.engine.trainer import TwoPhaseTrainer

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    两阶段训练主函数
    
    参数:
        cfg: Hydra配置对象
    """
    # ---------- 1. 随机种子 & 环境 ----------
    set_global_seed(cfg.seed, deterministic=True)  # 确保完全可复现
    
    # ---------- 2. 路径解析与校验 ----------
    vocab_json = to_absolute_path(cfg.data.vocab_json)
    train_pkl  = to_absolute_path(cfg.data.train_pkl)
    val_pkl    = to_absolute_path(cfg.data.val_pkl)

    # 检查所有必要文件是否存在
    missing = [p for p in [vocab_json, train_pkl, val_pkl] if not Path(p).exists()]
    if missing:
        msg = "\n".join(["  • " + str(p) for p in missing])
        raise FileNotFoundError(
            f"""❌ 发现以下数据文件缺失：
    {msg}

    解决办法：
    1) 按 README 里的 "Preprocess" 步骤下载并预处理 EuroParl；
    2) 或自行将数据放到目录：
        {Path(vocab_json).parent.parent}
    3) 或导出环境变量 EP_DATA 指向你的数据根目录，例如：
        export EP_DATA=/my/dataset
    4) 亦可在命令行用 Hydra 覆盖，例如：
        python scripts/train.py data.train_pkl=/path/train.pkl \\
                                data.val_pkl=/path/test.pkl   \\
                                data.vocab_json=/path/vocab.json
    """
        )

    # ---------- 3. 加载词表 & 注入配置 ----------
    vocab = Vocab.load(vocab_json)
    cfg.model.vocab_size = len(vocab)
    cfg.data.pad_idx   = vocab.token2idx['<PAD>']
    
    # 验证信道类型
    try:
        channel_cls = get_channel(cfg.data.channel)
    except ValueError as e:
        raise ValueError(f"无效的信道类型 '{cfg.data.channel}'。请确保已在 deepsc/models/channel.py 中注册。") from e

    # ---------- 4. 打印训练配置 ----------
    print(f"配置摘要:\n"
          f"  • 模型: DeepSCStrict (严格实现)\n"
          f"  • 词表大小: {cfg.model.vocab_size}\n"
          f"  • 信道: {cfg.data.channel}\n"
          f"  • 训练批大小: {cfg.train.batch_size}\n"
          f"  • 学习率: {cfg.train.lr}\n"
          f"  • 互信息权重: {cfg.train.lambda_mi}\n"
          f"  • 梯度裁剪值: {cfg.model.grad_clip}\n"
          f"  • 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    # ---------- 5. 构建 DataLoader ----------
    print("加载数据...")
    
    train_loader = make_dataloader(
        train_pkl,
        batch_size   = cfg.train.batch_size,
        pad_idx      = cfg.data.pad_idx,
        num_workers  = cfg.get("num_workers", 4),
        shuffle      = True,
    )
    val_loader = make_dataloader(
        val_pkl,
        batch_size   = cfg.train.batch_size,
        pad_idx      = cfg.data.pad_idx,
        shuffle      = False,
        num_workers  = cfg.get("num_workers", 4),
    )
    
    print(f"训练集大小: {len(train_loader.dataset)} 句")
    print(f"验证集大小: {len(val_loader.dataset)} 句")

    # ---------- 6. 初始化模型与设备 ----------
    print("初始化模型...")
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和信道
    model = DeepSCStrict(cfg.model)
    channel = channel_cls()
    
    # ---------- 7. 创建两阶段训练器 ----------
    print("创建两阶段训练器...")
    
    trainer = TwoPhaseTrainer(
        model=model,
        channel=channel,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg
    )
    
    # ---------- 8. 开始训练 ----------
    print(f"开始两阶段训练...")
    
    best_metric = trainer.train()
    
    print(f"训练完成! 最佳验证指标: {best_metric:.6f}")

if __name__ == '__main__':
    main()