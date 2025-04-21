# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC/deepsc/engine/lit_module.py
# (已修改，为手动优化添加显式的 AMP 支持)
# --------------------------------------------------------------------------------
import itertools, math, torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast # 导入 AMP 相关工具

from deepsc.models.transformer import DeepSC
from deepsc.models.mine import MINE
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score
from contextlib import nullcontext # 用于创建无操作的上下文管理器

class LitDeepSC(pl.LightningModule):
    """
    DeepSC端到端训练封装 (改进版，支持手动优化下的 AMP)

    使用PyTorch Lightning框架封装DeepSC模型，实现端到端训练。
    包含交替优化MINE网络和主网络的策略，以及灵活的学习率调度。
    在手动优化中实现梯度裁剪，解决自动裁剪不兼容问题。
    显式支持自动混合精度 (AMP) 训练以提高效率。

    参数:
        cfg: 配置对象，包含模型参数和训练超参数
    """
    automatic_optimization: bool = False # 保持手动优化

    # --------------------------- 初始化 --------------------------- #
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 虽然使用手动优化，但 save_hyperparameters 仍有其用途 (如保存配置到检查点)
        # 过滤掉不可序列化的部分，例如 OmegaConf 对象本身
        hparams_to_save = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list, tuple))}
        # 如果模型配置也是字典，可以递归处理或简单地保存关键参数
        if isinstance(cfg.get('model'), dict):
             hparams_to_save['model_params'] = {mk: mv for mk, mv in cfg.model.items() if isinstance(mv, (int, float, str, bool))}
        # ... 可以为 data, train 等添加类似逻辑 ...
        self.save_hyperparameters(hparams_to_save)
        # 也可以直接将 cfg 存起来，供后续访问
        self.hparams.cfg = cfg # 存储完整配置，即使它可能不可序列化

        # ---- AMP 相关 ----
        # 检查是否启用 AMP (基于 precision 设置)
        self.use_amp = str(self.cfg.get('precision', 32)) in ['16', 16, '16-mixed']
        # 仅在启用 AMP 时创建 GradScaler
        self.scaler = GradScaler() if self.use_amp else None
        print(f"  LitDeepSC 初始化: AMP {'启用' if self.use_amp else '禁用'}")


        # ---- 子网络 ---- #
        self.model = DeepSC(cfg.model)
        self.channel = get_channel(cfg.data.channel)()
        self.mine = MINE(cfg.model.latent_dim, hidden=256, activation='relu')

        # ---- 损失 & 常量 ---- #
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.data.pad_idx)

        # 扩展配置项：MINE 相关参数
        self.mine_update_freq = cfg.train.get('mine_update_freq', 10)
        self.lambda_mi = cfg.train.lambda_mi
        self.mine_extra_steps = cfg.train.get('mine_extra_steps', 1)
        self.mine_warmup = cfg.train.get('mine_warmup', 1000)

        # 梯度裁剪值
        self.grad_clip = cfg.model.get('grad_clip', 1.0)

        # 动态 SNR 范围
        self.snr_low = cfg.train.get('snr_low', 10)
        self.snr_high = cfg.train.get('snr_high', 20)

        # 步数计数器
        self._step_cnt = 0

    # --------------------------- 前向 --------------------------- #
    def forward(self, src: torch.Tensor, n_var: float):
        """
        模型前向传播（推理用）

        参数:
            src: [batch, seq_len] 输入序列
            n_var: 噪声方差的平方 (sigma^2)

        返回:
            模型预测的logits
        """
        # 推理时也建议使用 autocast (如果训练时用了 AMP)
        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()
        with amp_context:
             return self.model(src, n_var, self.channel)

    # --------------------------- 训练 --------------------------- #
    def training_step(self, batch, batch_idx):
        """
        单步训练逻辑 (包含显式 AMP 处理)

        包括主网络更新和MINE网络更新，使用手动优化策略。
        实现手动梯度裁剪，替代自动梯度裁剪功能。
        使用 torch.cuda.amp.autocast 和 GradScaler 处理混合精度。

        参数:
            batch: 当前批次数据
            batch_idx: 批次索引

        返回:
            None (使用manual_backward)
        """
        opt_main, opt_mine = self.optimizers()

        src = batch.to(self.device)
        n_var_squared = self._sample_noise_var() # 获取噪声方差 sigma^2

        # 选择 AMP 上下文
        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()

        # --- 更新 MINE 网络 ---
        # MINE 的更新通常不需要在 AMP 上下文中进行，因为它计算标量损失，梯度可能较小
        # 但为了代码一致性，也可以包含在内，或者单独处理
        # 这里选择在 AMP 外更新 MINE，除非 MINE 本身很大或计算复杂
        if self._step_cnt % self.mine_update_freq == 0:
            with torch.no_grad(): # 生成 MINE 输入时不需要梯度
                 with amp_context: # 但生成 tx, rx 可能在 AMP 下
                      _, tx_detached, rx_detached = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
                      tx_f_detached = tx_detached.reshape(-1, tx_detached.size(-1))
                      rx_f_detached = rx_detached.reshape(-1, rx_detached.size(-1))

            # 过滤空行 (PAD)
            non_empty_mask_detached = (tx_f_detached.abs().sum(dim=1) > 1e-6)
            if non_empty_mask_detached.sum() > 0:
                tx_f_detached = tx_f_detached[non_empty_mask_detached]
                rx_f_detached = rx_f_detached[non_empty_mask_detached]

                # MINE 网络更新 (通常不在 autocast 下，除非 MINE 计算量大)
                for _ in range(self.mine_extra_steps):
                     opt_mine.zero_grad()
                     mi_lb_mine = self.mine(tx_f_detached, rx_f_detached)
                     loss_mine = -mi_lb_mine # 最大化下界等于最小化负下界

                     # MINE 的反向传播不需要 GradScaler，因为我们没有用 autocast 包裹 MINE 的 forward
                     self.manual_backward(loss_mine)

                     # 手动梯度裁剪 (MINE网络)
                     # 在 optimizer.step() 之前裁剪 MINE 的梯度
                     self.clip_gradients(opt_mine, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
                     # clip_grad_norm_(self.mine.parameters(), self.grad_clip) # 旧方法

                     opt_mine.step()
            # else: MINE 输入为空，跳过更新

        # --- 更新主网络 ---
        opt_main.zero_grad()
        with amp_context:
            # ① 主网络前向传播
            logits, tx, rx = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
            loss_ce = self._compute_ce(logits, src)

            # ② 计算 MI Lower Bound
            tx_f = tx.reshape(-1, tx.size(-1))
            rx_f = rx.reshape(-1, rx.size(-1))
            non_empty_mask = (tx_f.abs().sum(dim=1) > 1e-6)
            mi_lb = torch.tensor(0.0, device=self.device) # 默认值
            if non_empty_mask.sum() > 0:
                tx_f = tx_f[non_empty_mask]
                rx_f = rx_f[non_empty_mask]
                # MINE 前向传播 (读取 MI，不训练 MINE)
                # 注意：这里 self.mine 的前向应该也在 autocast 内，因为它参与主网络的损失计算
                mi_lb = self.mine(tx_f, rx_f)

        # ③ 计算总损失 (在 autocast 外计算，以 float32 进行)
        if self._step_cnt >= self.mine_warmup:
            mi_loss_weight = self.lambda_mi
        else:
            mi_loss_weight = 0.0
        loss = loss_ce - mi_loss_weight * mi_lb

        # ④ 反向传播 (使用 GradScaler)
        if self.scaler:
            self.manual_backward(self.scaler.scale(loss)) # 使用 scaler 缩放损失
        else:
            self.manual_backward(loss) # 无 AMP 时正常反向传播

        # ⑤ 梯度裁剪 (在 unscale_ 和 step 之间)
        if self.scaler:
            self.scaler.unscale_(opt_main) # 先 unscale 梯度
        # 手动梯度裁剪 (主网络) - 使用 Lightning 的 clip_gradients 更方便
        self.clip_gradients(opt_main, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
        # clip_grad_norm_(self.model.parameters(), self.grad_clip) # 旧方法

        # ⑥ 更新主网络权重 (使用 GradScaler)
        if self.scaler:
            self.scaler.step(opt_main) # scaler.step 会检查梯度是否 inf/nan
            self.scaler.update() # 更新 scaler 的缩放因子
        else:
            opt_main.step() # 无 AMP 时正常更新

        # 更新学习率调度器 (通常在优化器步骤之后)
        sch = self.lr_schedulers()
        if sch: sch.step()


        # ⑦ 记录训练指标
        # 在 autocast 外记录，确保是 float32
        self.log_dict({
            "train_loss": loss.item(), # 使用 .item() 获取标量值
            "train_ce": loss_ce.item(),
            "train_mi_lb": mi_lb.item(),
            "mi_weight": float(mi_loss_weight), # 确保是 python float
            "lr": opt_main.param_groups[0]["lr"],
            "scale": self.scaler.get_scale() if self.scaler else 1.0 # 记录 GradScaler 的缩放因子
        }, on_step=True, prog_bar=True)

        # 更新步数计数器
        self._step_cnt += 1

    # --------------------------- 验证 --------------------------- #
    def validation_step(self, batch, batch_idx):
        """
        验证步骤 (添加 AMP 支持)
        """
        src = batch.to(self.device)
        n_var_squared = self._sample_noise_var()

        # 使用 AMP 上下文进行前向传播
        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()
        with amp_context:
            logits = self.model(src, n_var_squared, self.channel) # 调用模型 forward
            val_ce = self._compute_ce(logits, src)

        # BLEU 计算不需要在 AMP 下
        val_bleu = bleu_score(logits.argmax(-1), src[:, 1:])

        self.log_dict({
            "val_ce": val_ce.item(), # 获取标量
            "val_bleu": val_bleu, # bleu_score 返回 float
        }, sync_dist=True, prog_bar=True, on_epoch=True) # 确保在 epoch 结束时记录

    # ------------------------ Optimizer & LR -------------------- #
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        cfg_t = self.cfg.train
        d_model = self.cfg.model.d_model
        peak_lr = cfg_t.lr
        warmup = cfg_t.warmup
        lr_type = cfg_t.get("lr_type", "inverse_sqrt")

        # ---- 主网络 ---- #
        # 不再对 LR 进行 d_model 缩放，因为 inverse_sqrt 调度器内部处理了类似效果
        opt_main = Adam(self.model.parameters(),
                        lr=peak_lr, # 直接使用配置的学习率作为峰值
                        betas=(0.9, 0.98), eps=1e-9) # 论文中用的 eps=1e-9

        def lr_lambda(step: int):
            # Lightning 的 step 从 0 开始计数
            step += 1 # 调整为从 1 开始计数，以匹配公式
            arg1 = step ** -0.5
            arg2 = step * (warmup ** -1.5)
            # inverse_sqrt 调度: lr = peak_lr * min(step^-0.5, step * warmup^-1.5) * (d_model^-0.5)
            # LambdaLR 的返回值是乘法因子，所以我们需要包含 d_model^-0.5
            scale = d_model ** -0.5
            if lr_type == "inverse_sqrt":
                 # 注意：这里返回的是学习率的乘法因子
                 return scale * min(arg1, arg2) if warmup > 0 else scale * arg1
            elif lr_type == "linear_decay":
                # 线性预热 + 线性衰减到 0 (需要总步数) - 不推荐与 Adam + Transformer 结合
                print("警告: linear_decay 学习率调度器通常不与 Adam + Transformer 一起使用。推荐 inverse_sqrt。")
                if step < warmup:
                    return float(step) / float(max(1, warmup))
                # 计算总步数 (近似值)
                total_steps = self.trainer.estimated_stepping_batches if self.trainer else warmup * 10 # 粗略估计
                return max(0.0, 1.0 - float(step - warmup) / float(max(1, total_steps - warmup)))
            else:
                raise ValueError(f"未知 lr_type: {lr_type}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda)

        # ---- MINE ---- #
        # MINE 通常使用固定的较小学习率
        mine_lr = self.cfg.train.get("mine_lr", 1e-4) # 允许配置 MINE 学习率
        opt_mine = Adam(self.mine.parameters(), lr=mine_lr)

        return [opt_main, opt_mine], [{"scheduler": scheduler, "interval": "step"}] # 确保调度器按步更新

    # --------------------------- 工具 --------------------------- #
    def _compute_ce(self, logits: torch.Tensor, trg: torch.Tensor):
        """
        计算交叉熵损失
        """
        # 确保 logits 是 float32 进行损失计算，防止 AMP 下的问题
        logits_float32 = logits.float()
        B, L_pred, V = logits_float32.size()
        # 目标是 src[:, 1:]
        trg_flat = trg[:, 1:].reshape(-1)
        logits_flat = logits_float32.reshape(-1, V)
        return self.ce_loss(logits_flat, trg_flat)

    def _sample_noise_var(self) -> float:
        """
        随机采样信噪比(dB)并转换为噪声方差 sigma^2
        """
        snr_db = torch.empty(1).uniform_(self.snr_low, self.snr_high).item()
        snr_lin = 10 ** (snr_db / 10)
        # 噪声方差 σ^2 = 1 / (2 * SNR_lin)，符合论文设置
        # 增加最小值钳制，避免 snr_lin 接近 0 时方差爆炸
        noise_variance = 1.0 / (2.0 * max(snr_lin, 1e-10))
        return noise_variance