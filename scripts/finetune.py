# -*- coding: utf-8 -*-
"""
一键迁移学习
==================================================
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
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pathlib import Path

import pytorch_lightning as pl
import torch

from deepsc.engine.lit_module import LitDeepSC
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.utils.seed import set_global_seed
from deepsc.utils.freeze import freeze, unfreeze, count_trainable_params
from deepsc.models import get_channel      # 自动触发 __init__.py

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ---------- 1. 解析并校验路径 ----------
    ckpt_path     = to_absolute_path(cfg.ckpt_path)
    train_pkl     = to_absolute_path(cfg.data.train_pkl)
    val_pkl       = to_absolute_path(cfg.data.val_pkl)
    vocab_json    = to_absolute_path(cfg.data.vocab_json)

    for p in [ckpt_path, train_pkl, val_pkl, vocab_json]:
        if not Path(p).exists():
            raise FileNotFoundError(f"未找到文件：{p}，请检查配置或使用 CLI 覆盖参数")

    # ---------- 2. 随机种子 & 加载预训练模型 ----------
    set_global_seed(cfg.seed)
    lit = LitDeepSC.load_from_checkpoint(ckpt_path, cfg=cfg)
    print(f"原模型可训练参数数量: {count_trainable_params(lit):,}")

    # ---------- 3. 冻结/解冻策略 ----------
    if cfg.mode == 'channel':
        # 场景①：更换信道 → 冻结语义层，仅解冻通道编/解码器
        freeze(lit.model.encoder)
        freeze(lit.model.decoder)
        unfreeze(lit.model.channel_encoder)
        unfreeze(lit.model.channel_decoder)

        lit.channel = get_channel(cfg.new_channel)()
        print(f"> 已切换信道为 {cfg.new_channel}")

    elif cfg.mode == 'domain':
        # 场景②：更换数据域 → 冻结通道层，仅解冻语义层最后一层
        freeze(lit.model.channel_encoder)
        freeze(lit.model.channel_decoder)
        for name, module in lit.model.encoder.named_modules():
            if 'layers.3' in name:
                unfreeze(module)
        for name, module in lit.model.decoder.named_modules():
            if 'layers.3' in name:
                unfreeze(module)

        # 重新加载新域词表，并动态调整模型的 embedding & proj 大小
        vocab = Vocab.load(vocab_json)
        cfg.model.vocab_size = len(vocab)
        cfg.data.pad_idx = vocab.token2idx['<PAD>']

        old_size = lit.model.encoder.embed.num_embeddings
        new_size = len(vocab)
        if new_size != old_size:
            lit.model.encoder.embed = torch.nn.Embedding(
                new_size, lit.model.encoder.embed.embedding_dim
            )
            lit.model.decoder.embed = torch.nn.Embedding(
                new_size, lit.model.decoder.embed.embedding_dim
            )
            lit.model.proj = torch.nn.Linear(
                lit.model.proj.in_features, new_size
            )
            print(f"> Embedding & proj 重置大小: {old_size} → {new_size}")
    else:
        raise ValueError("mode 必须为 'channel' 或 'domain'")

    print(f"当前可训练参数数量: {count_trainable_params(lit):,}")

    # ---------- 4. 加载数据 & 训练 ----------
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

    ck_dir = Path("finetune_ckpts")
    ck_dir.mkdir(exist_ok=True)

    trainer = pl.Trainer(
        max_epochs        = cfg.ft.epochs,
        precision         = cfg.precision,
        accelerator       = 'auto',
        devices           = 'auto',
        default_root_dir  = str(ck_dir),
        log_every_n_steps = 50,
    )
    trainer.fit(lit, train_loader, val_loader)

if __name__ == '__main__':
    main()
