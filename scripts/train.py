# -*- coding: utf-8 -*-
"""
Hydra + Lightning 一键训练
===========================

改进版训练脚本，支持更多的配置选项和更好的错误处理。
手动优化模式下不使用自动梯度裁剪，而是在模型内部实现。
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from deepsc.utils.seed import set_global_seed
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.models import get_channel  # 确保信道模型被注册

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    主训练函数
    
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

    # ---------- 4. 将梯度裁剪值传递给模型 ----------
    # 因为我们使用手动优化，所以在模型中应用梯度裁剪而非在 Trainer 中
    if "grad_clip" in cfg:
        cfg.model.grad_clip = cfg.grad_clip
    else:
        cfg.model.grad_clip = 1.0  # 默认梯度裁剪值

    print(f"配置摘要:\n"
          f"  • 模型: {cfg.model.get('_target_', 'DeepSC')}\n"
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

    # ---------- 6. Lightning 训练 ----------
    print("初始化模型与训练器...")
    
    lit_model = LitDeepSC(cfg)
    
    # 回调配置
    callbacks = []
    
    # 检查点回调：在验证集上监控 val_bleu，选出最高的模型
    ckpt_cb = ModelCheckpoint(
        monitor     = 'val_bleu',
        mode        = 'max',
        filename    = 'best-{epoch:02d}-{val_bleu:.3f}',
        save_top_k  = 3,              # 保存前3个最佳模型
        save_last   = True,           # 也保存最后一个检查点
        auto_insert_metric_name = False,
    )
    callbacks.append(ckpt_cb)
    
    # 学习率监控
    lr_logger = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_logger)
    
    # 可选的早停：如果验证集BLEU多个epoch没有改进，提前停止训练
    if cfg.get("early_stopping", False):
        early_stop = EarlyStopping(
            monitor    = 'val_bleu',
            mode       = 'max',
            patience   = cfg.get("early_stop_patience", 5),
            min_delta  = 0.001,        # BLEU必须至少提高0.001才算改进
        )
        callbacks.append(early_stop)
    
    # TensorBoard记录器
    logger = TensorBoardLogger(
        save_dir    = "logs",
        name        = f"deepsc_{cfg.data.channel}",
        default_hp_metric = False,
    )

    # 训练器配置 - 移除 gradient_clip_val 参数，因为我们使用手动优化
    trainer = pl.Trainer(
        max_epochs      = cfg.train.epochs,
        precision       = cfg.precision,
        accelerator     = 'auto',
        devices         = 'auto',
        callbacks       = callbacks,
        logger          = logger,
        log_every_n_steps = 50,
        val_check_interval = cfg.get("val_check_interval", 0.25), # 每1/4epoch验证一次
        # 移除 gradient_clip_val，因为我们使用手动优化
    )
    
    # 训练
    print(f"开始训练 (epochs={cfg.train.epochs})...")
    trainer.fit(lit_model, train_loader, val_loader)
    
    # 打印最佳模型路径
    print(f"\n训练完成！")
    if ckpt_cb.best_model_path:
        print(f"最佳模型: {ckpt_cb.best_model_path}")
        print(f"最佳BLEU: {ckpt_cb.best_model_score:.4f}")

if __name__ == '__main__':
    main()