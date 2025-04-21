# scripts/finetune.py - 更新后的迁移学习脚本
# -*- coding: utf-8 -*-
"""
一键迁移学习
==================================================
支持两种迁移学习场景：
1. 新信道环境：冻结语义层，只训练信道层
2. 新领域文本：冻结信道层，只训练语义层

用法示例：
▶ 新信道（AWGN→Rayleigh）：
  python scripts/finetune.py \
      ckpt_path=/path/best.ckpt \
      mode=channel new_channel=Rayleigh

▶ 新语料域（医疗文本）：
  python scripts/finetune.py \
      ckpt_path=/path/best.ckpt \
      mode=domain data.train_pkl=/new/train.pkl \
      data.val_pkl=/new/val.pkl \
      data.vocab_json=/new/vocab.json
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from deepsc.engine.lit_module import LitDeepSC
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.utils.seed import set_global_seed
from deepsc.utils.freeze import freeze, unfreeze, count_trainable_params
from deepsc.models import get_channel

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    迁移学习主函数
    
    参数:
        cfg: Hydra配置对象
    """
    # ---------- 1. 解析并校验路径 ----------
    ckpt_path     = to_absolute_path(cfg.ckpt_path)
    train_pkl     = to_absolute_path(cfg.data.train_pkl)
    val_pkl       = to_absolute_path(cfg.data.val_pkl)
    vocab_json    = to_absolute_path(cfg.data.vocab_json)

    for p in [ckpt_path, train_pkl, val_pkl, vocab_json]:
        if not Path(p).exists():
            raise FileNotFoundError(f"未找到文件：{p}，请检查配置或使用CLI覆盖参数")

    # ---------- 2. 随机种子 & 加载预训练模型 ----------
    set_global_seed(cfg.seed)
    print(f"加载源模型: {ckpt_path}")
    
    # 兼容性加载预训练模型
    try:
        # 新版加载方式
        lit = LitDeepSC.load_from_checkpoint(ckpt_path, cfg=cfg)
    except Exception as e:
        print(f"使用新方法加载失败，尝试兼容模式: {e}")
        # 加载原始检查点文件
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 手动创建模型并加载状态
        lit = LitDeepSC(cfg)
        lit.load_state_dict(checkpoint['state_dict'])
    
    # 传递梯度裁剪参数
    if hasattr(cfg, 'grad_clip'):
        lit.grad_clip = cfg.grad_clip
    elif hasattr(cfg.model, 'grad_clip'):
        lit.grad_clip = cfg.model.grad_clip
    else:
        lit.grad_clip = 1.0  # 默认值
        
    print(f"原模型可训练参数数量: {count_trainable_params(lit):,}")

    # ---------- 3. 冻结/解冻策略 ----------
    if cfg.mode == 'channel':
        # 场景①：更换信道 → 冻结语义层，仅解冻通道编/解码器
        print(f"迁移学习模式: 信道迁移 → {cfg.new_channel}")
        
        # 冻结语义层
        freeze(lit.model.encoder)
        freeze(lit.model.decoder)
        freeze(lit.model.proj)
        
        # 解冻信道层
        unfreeze(lit.model.channel_encoder)
        unfreeze(lit.model.channel_decoder)

        # 更新信道模型
        try:
            lit.channel = get_channel(cfg.new_channel)()
            print(f"> 已切换信道为 {cfg.new_channel}")
        except ValueError as e:
            raise ValueError(f"无效的信道类型 '{cfg.new_channel}'。请确保已在 deepsc/models/channel.py 中注册。") from e

    elif cfg.mode == 'domain':
        # 场景②：更换数据域 → 冻结通道层，仅解冻语义层
        print(f"迁移学习模式: 领域迁移 → 新数据源")
        
        # 冻结信道层
        freeze(lit.model.channel_encoder)
        freeze(lit.model.channel_decoder)
        
        # 选择性地解冻语义层（仅最后一层或自适应选择）
        if cfg.get('unfreeze_all_semantic', False):
            # 解冻所有语义层
            unfreeze(lit.model.encoder)
            unfreeze(lit.model.decoder)
            unfreeze(lit.model.proj)
            print("> 解冻所有语义层")
        else:
            # 仅解冻最后1-2层
            for name, module in lit.model.encoder.named_modules():
                if 'layers.2' in name:  # 假设有3层，这是最后一层
                    unfreeze(module)
            for name, module in lit.model.decoder.named_modules():
                if 'layers.2' in name:  # 最后一层解码器
                    unfreeze(module)
            unfreeze(lit.model.proj)  # 输出投影层
            print("> 仅解冻语义层最后1-2层")

        # 重新加载新域词表，并动态调整模型的embedding和proj大小
        vocab = Vocab.load(vocab_json)
        old_size = lit.model.encoder.embed.num_embeddings
        new_size = len(vocab)
        
        cfg.model.vocab_size = new_size
        cfg.data.pad_idx = vocab.token2idx['<PAD>']
        
        if new_size != old_size:
            print(f"> 词表大小变化: {old_size} → {new_size}")
            
            # 重新初始化嵌入层
            lit.model.encoder.embed = torch.nn.Embedding(
                new_size, lit.model.encoder.embed.embedding_dim
            )
            lit.model.decoder.embed = torch.nn.Embedding(
                new_size, lit.model.decoder.embed.embedding_dim
            )
            
            # 重新初始化输出投影层
            lit.model.proj = torch.nn.Linear(
                lit.model.proj.in_features, new_size
            )
            print(f"> Embedding和proj层已重置为新大小")
            
    else:
        raise ValueError("mode参数必须为'channel'或'domain'")

    # 打印可训练参数情况
    print(f"当前可训练参数数量: {count_trainable_params(lit):,}")
    trainable_ratio = count_trainable_params(lit) / sum(p.numel() for p in lit.parameters())
    print(f"可训练参数占比: {trainable_ratio:.2%}")

    # ---------- 4. 加载数据 & 训练 ----------
    print(f"加载数据集...")
    vocab = Vocab.load(vocab_json)
    train_loader = make_dataloader(
        train_pkl,
        batch_size  = cfg.train.batch_size,
        pad_idx     = vocab.token2idx['<PAD>'],
        num_workers = 4,
        shuffle     = True,
    )
    val_loader = make_dataloader(
        val_pkl,
        batch_size  = cfg.train.batch_size,
        pad_idx     = vocab.token2idx['<PAD>'],
        shuffle     = False,
        num_workers = 4,
    )
    
    print(f"训练集: {len(train_loader.dataset)} 句")
    print(f"验证集: {len(val_loader.dataset)} 句")

    # 创建输出目录
    ft_mode = 'channel_' + cfg.new_channel if cfg.mode == 'channel' else 'domain'
    ck_dir = Path(f"finetune_{ft_mode}_ckpts")
    ck_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置检查点回调
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_bleu',
        mode='max',
        filename='ft-{epoch:02d}-{val_bleu:.3f}',
        save_top_k=3,
        save_last=True,
    )
    
    # 设置早停回调
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_bleu',
        patience=cfg.ft.get('patience', 3),
        min_delta=0.001,
        mode='max',
    )

    # 配置训练器 - 不使用自动梯度裁剪
    print(f"开始迁移学习 (epochs={cfg.ft.epochs})...")
    trainer = pl.Trainer(
        max_epochs        = cfg.ft.epochs,
        precision         = cfg.precision,
        accelerator       = 'auto',
        devices           = 'auto',
        default_root_dir  = str(ck_dir),
        log_every_n_steps = 50,
        callbacks         = [checkpoint_callback, early_stop_callback],
        val_check_interval = 0.5,  # 每半个epoch验证一次
        # 不使用自动梯度裁剪，因为我们使用手动优化和手动裁剪
    )
    
    # 训练
    trainer.fit(lit, train_loader, val_loader)
    
    # 打印结果
    print("\n迁移学习完成!")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")
    print(f"最佳BLEU: {checkpoint_callback.best_model_score:.4f}")

if __name__ == '__main__':
    main()