# -*- coding: utf-8 -*-
"""
LightningModule：DeepSC 端到端训练封装
======================================
改动要点
--------
1. **修正学习率调度**：支持
      • inverse_sqrt (Transformer 原式)
      • linear_decay  (线性 warm‑up)
   解决 lr≈0 导致 loss 不下降的问题；
2. 训练日志新增 `lr`，便于观察；
3. 其余逻辑（手动优化、交替更新 MINE、指标 rename）保持不变。
"""

import itertools, math, torch
import pytorch_lightning as pl
from torch.optim import Adam

from deepsc.models.transformer import DeepSC
from deepsc.models.mine import MINE
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score

# -------------------------------- 超参 -------------------------------- #
MI_UPDATE_FREQ = 10             # 每 N step 更新一次 MINE
SNR_LOW, SNR_HIGH = 10, 20      # 训练时随机采样的 SNR(dB) 区间
# ---------------------------------------------------------------------- #

class LitDeepSC(pl.LightningModule):
    automatic_optimization: bool = False

    # --------------------------- 初始化 --------------------------- #
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # —— 子网络 —— #
        self.model   = DeepSC(cfg.model)
        self.channel = get_channel(cfg.data.channel)()
        self.mine    = MINE(cfg.model.latent_dim)

        # —— 损失 & 常量 —— #
        self.ce_loss   = torch.nn.CrossEntropyLoss(ignore_index=cfg.data.pad_idx)
        self.lambda_mi = cfg.train.lambda_mi
        self._step_cnt = itertools.count()

    # --------------------------- 前向 --------------------------- #
    def forward(self, src: torch.Tensor, n_var: float):
        """仅推理用；训练逻辑见 training_step"""
        return self.model(src, n_var, self.channel)

    # --------------------------- 训练 --------------------------- #
    def training_step(self, batch, batch_idx):
        opt_main, opt_mine = self.optimizers()

        src   = batch.to(self.device)
        n_var = self._sample_noise()

        # ① 主网络前向
        logits, tx, rx = self.model(src, n_var, self.channel,
                                    return_tx_rx=True)
        loss_ce = self._compute_ce(logits, src)

        tx_f, rx_f = [z.reshape(-1, z.size(-1)) for z in (tx, rx)]

        # ② —— 更新 MINE —— #
        if next(self._step_cnt) % MI_UPDATE_FREQ == 0:
            opt_mine.zero_grad()
            mi_lb_mine = self.mine(tx_f.detach(), rx_f.detach())
            self.manual_backward(-mi_lb_mine)   # 最大化 lb
            opt_mine.step()

        # ③ —— 更新主网络 —— #
        mi_lb = self.mine(tx_f, rx_f)           # 重新前向，使梯度回流主网
        loss   = loss_ce - self.lambda_mi * mi_lb

        opt_main.zero_grad()
        self.manual_backward(loss)
        opt_main.step()

        # ④ —— 记录 —— #
        self.log_dict({
            "train_loss": loss,
            "train_ce":   loss_ce,
            "train_mi_lb": mi_lb.detach(),
            "lr": opt_main.param_groups[0]["lr"],
        }, on_step=True, prog_bar=True)

    # --------------------------- 验证 --------------------------- #
    def validation_step(self, batch, _):
        src   = batch.to(self.device)
        n_var = self._sample_noise()
        logits = self(src, n_var)

        val_ce   = self._compute_ce(logits, src)
        val_bleu = bleu_score(logits.argmax(-1), src)

        self.log_dict({
            "val_ce":  val_ce,
            "val_bleu": val_bleu,
        }, sync_dist=True, prog_bar=True)

    # ------------------------ Optimizer & LR -------------------- #
    def configure_optimizers(self):
        cfg_t = self.hparams.train
        d_model = self.hparams.model.d_model
        peak_lr = cfg_t.lr
        warmup  = cfg_t.warmup
        lr_type = cfg_t.get("lr_type", "inverse_sqrt")

        # —— 主网络 —— #
        opt_main = Adam(self.model.parameters(),
                lr = d_model ** -0.5,  # ★移除 peak_lr
                betas=(0.9, 0.98), eps=1e-8)

        def lr_lambda(step: int):
            step += 1
            if lr_type == "inverse_sqrt":
                return min(step ** -0.5,
                           step * warmup ** -1.5)
            elif lr_type == "linear_decay":
                return min(step / warmup, 1.0)
            else:
                raise ValueError(f"未知 lr_type: {lr_type}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda)

        # —— MINE —— #
        opt_mine = Adam(self.mine.parameters(), lr=1e-3)

        return [opt_main, opt_mine], [scheduler]

    # --------------------------- 工具 --------------------------- #
    def _compute_ce(self, logits: torch.Tensor, trg: torch.Tensor):
        B, Lm1, V = logits.size()
        return self.ce_loss(logits.reshape(-1, V),
                            trg[:, 1:].reshape(-1))

    @staticmethod
    def _sample_noise() -> float:
        snr_db = torch.empty(1).uniform_(SNR_LOW, SNR_HIGH).item()
        snr_lin = 10 ** (snr_db / 10)
        # σ = √(1/(2·snr))
        return math.sqrt(1 / (2 * snr_lin))
