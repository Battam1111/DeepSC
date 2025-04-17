# -*- coding: utf-8 -*-
"""
Hydra + Lightning 一键训练
"""
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from deepsc.utils.seed import set_global_seed
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ---------- 1. 随机种子 & 环境 ----------
    set_global_seed(cfg.seed)

    # ---------- 2. 路径解析与校验 ----------
    vocab_json = to_absolute_path(cfg.data.vocab_json)
    train_pkl  = to_absolute_path(cfg.data.train_pkl)
    val_pkl    = to_absolute_path(cfg.data.val_pkl)

    missing = [p for p in [vocab_json, train_pkl, val_pkl] if not Path(p).exists()]
    if missing:
        msg = "\n".join(["  • " + str(p) for p in missing])
        raise FileNotFoundError(
            f"""❌ 发现以下数据文件缺失：
    {msg}

    解决办法：
    1) 按 README 里的 “Preprocess” 步骤下载并预处理 EuroParl；
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

    # ---------- 4. 构建 DataLoader ----------
    train_loader = make_dataloader(
        train_pkl,
        batch_size   = cfg.train.batch_size,
        pad_idx      = cfg.data.pad_idx,
        num_workers  = 4,
        shuffle      = True,
    )
    val_loader = make_dataloader(
        val_pkl,
        batch_size   = cfg.train.batch_size,
        pad_idx      = cfg.data.pad_idx,
        shuffle      = False,
        num_workers  = 4,
    )

    # ---------- 5. Lightning 训练 ----------
    lit_model = LitDeepSC(cfg)
    # 在验证集上监控 val_bleu，选出最高的模型
    ckpt_cb   = ModelCheckpoint(
        monitor  = 'val_bleu',
        mode     = 'max',
        filename = 'best-{epoch:02d}-{val_bleu:.3f}',
    )
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs      = cfg.train.epochs,
        precision       = cfg.precision,
        accelerator     = 'auto',
        devices         = 'auto',
        callbacks       = [ckpt_cb, lr_logger],
        log_every_n_steps = 50,
    )
    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == '__main__':
    main()
