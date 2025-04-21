# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC-Fork/deepsc/engine/lit_module.py
# (最终确认版: 包含 weight_decay, 调整 epsilon, MINE LR, 移除主网络梯度裁剪, 添加初始化逻辑)
# --------------------------------------------------------------------------------
import itertools, math, torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast # 导入 AMP 相关工具
# 导入 Xavier 初始化
from torch.nn.init import xavier_uniform_
# 导入 nn 模块
import torch.nn as nn

# 动态导入模型，避免严格依赖
try:
    from deepsc.models.transformer import DeepSC # 标准版模型
except ImportError:
    DeepSC = None # 定义占位符
try:
    from deepsc.models.deepsc_strict import DeepSCStrict
except ImportError:
    DeepSCStrict = None # 定义占位符
try:
    from deepsc.models.mine import MINE # 标准版 MINE
except ImportError:
    MINE = None # 定义占位符
try:
    from deepsc.models.mine_strict import MINEStrict
except ImportError:
    MINEStrict = None # 定义占位符

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
        # 存储超参数
        hparams_to_save = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list, tuple))}
        if isinstance(cfg.get('model'), dict):
             hparams_to_save['model_params'] = {mk: mv for mk, mv in cfg.model.items() if isinstance(mv, (int, float, str, bool))}
        self.save_hyperparameters(hparams_to_save)
        self.hparams.cfg = cfg # 存储完整配置

        # ---- AMP 相关 ----
        self.use_amp = str(self.cfg.get('precision', 32)) in ['16', 16, '16-mixed']
        self.scaler = GradScaler() if self.use_amp else None
        print(f"  LitDeepSC 初始化: AMP {'启用' if self.use_amp else '禁用'}")

        # ---- 子网络 ---- #
        # 根据 strict_model 决定加载哪个模型和 MINE
        is_strict = cfg.get("strict_model", False)
        if is_strict:
             if DeepSCStrict is None or MINEStrict is None:
                 raise ImportError("严格模式需要 deepsc.models.deepsc_strict 和 deepsc.models.mine_strict")
             print("  加载严格版本的 DeepSCStrict 和 MINEStrict")
             self.model = DeepSCStrict(cfg.model)
             self.mine = MINEStrict(cfg.model.latent_dim) # 严格 MINE 初始化已在类中定义
             # 对严格模型应用 Xavier 初始化
             print("  应用 Xavier Uniform 初始化到 DeepSCStrict...")
             self.model.apply(self._init_weights_xavier)
        else:
             if DeepSC is None or MINE is None:
                 raise ImportError("标准模式需要 deepsc.models.transformer 和 deepsc.models.mine")
             print("  加载标准版本的 DeepSC 和 MINE")
             self.model = DeepSC(cfg.model)
             self.mine = MINE(cfg.model.latent_dim, hidden=256, activation='relu') # 标准 MINE 初始化已在类中
             # 对标准模型应用 Xavier 初始化
             print("  应用 Xavier Uniform 初始化到 DeepSC...")
             self.model.apply(self._init_weights_xavier)


        self.channel = get_channel(cfg.data.channel)()

        # ---- 损失 & 常量 ---- #
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.data.pad_idx)

        # MINE 相关参数
        self.mine_update_freq = cfg.train.get('mine_update_freq', 1) # 尝试每步更新 MINE
        self.lambda_mi = cfg.train.lambda_mi # 从配置读取，应设为 0.001 或 0.0009
        self.mine_extra_steps = cfg.train.get('mine_extra_steps', 1) # 每次更新 MINE 的步数
        self.mine_warmup = cfg.train.get('mine_warmup', 0) # 尝试禁用 MINE 预热

        # 梯度裁剪值 (仅用于 MINE)
        self.grad_clip = cfg.model.get('grad_clip', 1.0)

        # SNR范围 (应从配置读取，例如 5, 10)
        self.snr_low = cfg.train.get('snr_low', 5)  # 默认改为 5
        self.snr_high = cfg.train.get('snr_high', 10) # 默认改为 10

        self._step_cnt = 0

    # Xavier 初始化函数
    def _init_weights_xavier(self, module):
        """使用 Xavier Uniform 初始化线性层权重"""
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                 nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
             nn.init.constant_(module.weight, 1.0)
             nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
             # 使用正态分布初始化 Embedding
             # Pylance 可能在此处误报，但代码逻辑正确
             nn.init.normal_(module.weight, mean=0, std=module.embedding_dim ** -0.5)


    # --------------------------- 前向 --------------------------- #
    def forward(self, src: torch.Tensor, n_var: float):
        """
        模型前向传播（主要用于推理或需要直接调用模型时）
        注意：训练时通常直接在 training_step 中调用 self.model(...)
        """
        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()
        with amp_context:
            # 假设 model 的 forward 签名统一为 (src, n_var, channel, return_tx_rx)
            output = self.model(src, n_var, self.channel, return_tx_rx=False)
            # 如果 model 返回了 tuple (例如包含 tx, rx)，只取第一个元素 (logits)
            if isinstance(output, tuple):
                return output[0]
            else:
                return output

    # --------------------------- 训练 --------------------------- #
    def training_step(self, batch, batch_idx):
        """
        单步训练逻辑 (包含显式 AMP 处理)
        """
        opt_main, opt_mine = self.optimizers()

        src = batch.to(self.device)
        n_var_squared = self._sample_noise_var() # 获取噪声方差 sigma^2

        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()

        # --- 更新 MINE 网络 ---
        if self._step_cnt % self.mine_update_freq == 0:
            # 将 model 置于 eval 模式生成 MINE 输入，避免 BN/Dropout 影响
            self.model.eval()
            with torch.no_grad():
                with amp_context:
                    _, tx_detached, rx_detached = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
                    tx_f_detached = tx_detached.reshape(-1, tx_detached.size(-1))
                    rx_f_detached = rx_detached.reshape(-1, rx_detached.size(-1))
            # 恢复 model 到 train 模式
            self.model.train()

            non_empty_mask_detached = (tx_f_detached.abs().sum(dim=1) > 1e-6)
            if non_empty_mask_detached.sum() > 0:
                tx_f_detached = tx_f_detached[non_empty_mask_detached]
                rx_f_detached = rx_f_detached[non_empty_mask_detached]

                # MINE 网络更新
                self.mine.train() # 确保 MINE 在训练模式
                for _ in range(self.mine_extra_steps):
                    opt_mine.zero_grad()
                    mi_lb_mine = self.mine(tx_f_detached, rx_f_detached)
                    loss_mine = -mi_lb_mine

                    self.manual_backward(loss_mine) # 对 MINE 的损失反向传播

                    # --- 修改：仅对 MINE 应用梯度裁剪 ---
                    # 使用 clip_grad_norm_，因为它不依赖 Lightning 的 scaler
                    # 在 optimizer.step() 之前裁剪
                    clip_grad_norm_(self.mine.parameters(), self.grad_clip)

                    opt_mine.step() # 更新 MINE 参数
                self.mine.eval() # 更新完后切换回评估模式，因为它会参与主网络损失

        # --- 更新主网络 ---
        self.model.train() # 确保主网络在训练模式
        opt_main.zero_grad()
        with amp_context:
            # ① 主网络前向传播
            # 假设 model forward 返回 (logits, tx, rx)
            logits, tx, rx = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
            loss_ce = self._compute_ce(logits, src) # 计算 CE Loss

            # ② 计算 MI Lower Bound
            tx_f = tx.reshape(-1, tx.size(-1))
            rx_f = rx.reshape(-1, rx.size(-1))
            non_empty_mask = (tx_f.abs().sum(dim=1) > 1e-6)
            mi_lb = torch.tensor(0.0, device=self.device)
            if non_empty_mask.sum() > 0:
                tx_f = tx_f[non_empty_mask]
                rx_f = rx_f[non_empty_mask]
                # 使用 eval 模式的 MINE 计算 MI，它参与主网络损失
                self.mine.eval()
                mi_lb = self.mine(tx_f, rx_f)

        # ③ 计算总损失 (在 autocast 外计算，以 float32 进行)
        if self._step_cnt >= self.mine_warmup: # 检查 MINE warmup
            mi_loss_weight = self.lambda_mi # 使用从配置中读取的调整后的 MI 权重
        else:
            mi_loss_weight = 0.0
        loss = loss_ce - mi_loss_weight * mi_lb # 总损失

        # ④ 反向传播 (使用 GradScaler)
        if self.scaler:
            self.manual_backward(self.scaler.scale(loss))
        else:
            self.manual_backward(loss)

        # ⑤ 梯度裁剪 (主网络) --- 修改：移除主网络的梯度裁剪 ---
        if self.scaler:
            self.scaler.unscale_(opt_main) # unscale 梯度以便检查或裁剪（如果需要）
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) # <--- 移除或注释掉这行

        # ⑥ 更新主网络权重 (使用 GradScaler)
        if self.scaler:
            self.scaler.step(opt_main)
            self.scaler.update()
        else:
            opt_main.step()

        # 更新学习率调度器
        sch = self.lr_schedulers()
        # 检查 sch 是否为 None 或者是一个列表/元组
        if sch:
            # 如果是列表或元组，假设第一个是主网络的调度器
            if isinstance(sch, (list, tuple)):
                # 确保列表不为空
                if sch:
                    # Lightning 会自动处理调度器的 step，这里不需要手动调用
                    # sch[0].step() # 只更新主网络的调度器
                    pass
            else: # 如果只有一个调度器
                # sch.step()
                pass

        # ⑦ 记录训练指标
        self.log_dict({
            "train_loss": loss.item(),
            "train_ce": loss_ce.item(),
            "train_mi_lb": mi_lb.item(),
            "mi_weight": float(mi_loss_weight),
            "lr": opt_main.param_groups[0]["lr"],
            "scale": self.scaler.get_scale() if self.scaler else 1.0
        }, on_step=True, prog_bar=True, logger=True) # 确保 logger=True

        self._step_cnt += 1


    # --------------------------- 验证 --------------------------- #
    def validation_step(self, batch, batch_idx):
        """
        验证步骤 (添加 AMP 支持)
        """
        src = batch.to(self.device)
        # 在验证时也使用训练期间的 SNR 范围采样，以模拟真实条件
        n_var_squared = self._sample_noise_var()

        amp_context = autocast() if self.use_amp and self.scaler is not None else nullcontext()
        with torch.no_grad(): # 验证时不需要计算梯度
             with amp_context:
                 # 确保调用 model 的 forward
                 logits, _, _ = self.model(src, n_var_squared, self.channel, return_tx_rx=True) # 获取 logits
                 val_ce = self._compute_ce(logits, src) # 计算 CE Loss

        # BLEU 计算不需要在 AMP 下
        # 预测索引
        pred_indices = logits.argmax(-1)
        # 目标索引 (移除 <START>)
        target_indices = src[:, 1:]

        # --- 对齐预测和目标的长度 (重要) ---
        pred_len = pred_indices.size(1)
        target_len = target_indices.size(1)
        if pred_len > target_len:
            pred_aligned = pred_indices[:, :target_len]
        elif pred_len < target_len:
            padding = torch.full((pred_indices.size(0), target_len - pred_len), self.cfg.data.pad_idx, device=self.device, dtype=torch.long)
            pred_aligned = torch.cat([pred_indices, padding], dim=1)
        else:
            pred_aligned = pred_indices
        # --- 对齐结束 ---

        val_bleu = bleu_score(pred_aligned.cpu(), target_indices.cpu()) # 使用对齐后的预测

        self.log_dict({
            "val_ce": val_ce.item(),
            "val_bleu": val_bleu,
        }, sync_dist=True, prog_bar=True, on_epoch=True, logger=True) # 确保 logger=True


    # ------------------------ Optimizer & LR -------------------- #
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        cfg_t = self.cfg.train
        d_model = self.cfg.model.d_model
        peak_lr = cfg_t.lr # 使用配置中的 lr 作为峰值或固定值
        warmup = cfg_t.warmup
        lr_type = cfg_t.get("lr_type", "fixed") # <--- 默认改为 "fixed" 以匹配原始代码
        weight_decay = cfg_t.get("weight_decay", 5e-4) # <--- 从配置读取 weight_decay, 默认 5e-4
        adam_eps = cfg_t.get("adam_eps", 1e-8) # <--- 从配置读取 epsilon, 默认 1e-8

        # ---- 主网络 ---- #
        print(f"  配置主网络优化器: Adam, LR Type: {lr_type}, Peak LR: {peak_lr}, Weight Decay: {weight_decay}, Epsilon: {adam_eps}")
        # 过滤掉不需要 weight decay 的参数 (例如 LayerNorm, bias)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        opt_main = Adam(optimizer_grouped_parameters,
                        lr=peak_lr,
                        betas=(0.9, 0.98),
                        eps=adam_eps) # Epsilon 应用于所有参数组

        scheduler_config = None
        if lr_type == "inverse_sqrt":
            print(f"  使用 inverse_sqrt 学习率调度器, Warmup: {warmup} steps")
            def lr_lambda(step: int):
                # LambdaLR 的 step 是从 0 开始的
                if step < warmup:
                    return float(step + 1) / float(warmup) # 线性预热因子
                else:
                    # 衰减因子，保持 d_model 影响
                    return (d_model ** -0.5) * (step ** -0.5) / (d_model ** -0.5 * warmup ** -0.5) # 修正: 使 warmup 结束时因子为 1

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda)
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1} # 按步更新
        elif lr_type == "fixed":
             print("  使用固定学习率，无调度器。")
             scheduler_config = None # 无调度器
        else:
            raise ValueError(f"未知 lr_type: {lr_type}")


        # ---- MINE ---- #
        # MINE 使用固定的较小学习率
        mine_lr = self.cfg.train.get("mine_lr", 1e-3) # <--- 从配置读取 MINE LR, 默认 1e-3
        mine_adam_eps = self.cfg.train.get("mine_adam_eps", 1e-8) # MINE 的 epsilon
        print(f"  配置 MINE 优化器: Adam, LR: {mine_lr}, Epsilon: {mine_adam_eps}")
        opt_mine = Adam(self.mine.parameters(), lr=mine_lr, eps=mine_adam_eps) # MINE 通常不需要 weight decay

        optimizers = [opt_main, opt_mine]
        schedulers = [scheduler_config] if scheduler_config else []

        return optimizers, schedulers

    # --------------------------- 工具 --------------------------- #
    def _compute_ce(self, logits: torch.Tensor, trg: torch.Tensor):
        """
        计算交叉熵损失
        """
        logits_float32 = logits.float() # 计算损失时使用 float32
        # 目标是 src[:, 1:]
        trg_real = trg[:, 1:] # 获取真实目标

        # 确保 logits 和 target 的序列长度匹配
        B, L_pred, V = logits_float32.size()
        L_trg = trg_real.size(1)
        if L_pred != L_trg:
            # 如果长度不匹配，通常是模型输出或目标处理逻辑错误
            # 这里采取截断策略，但这可能掩盖问题
            # print(f"警告: _compute_ce 发现 logits 长度 ({L_pred}) 与目标长度 ({L_trg}) 不匹配。将截断到最短长度 {min(L_pred, L_trg)}。")
            min_len = min(L_pred, L_trg)
            logits_float32 = logits_float32[:, :min_len, :]
            trg_real = trg_real[:, :min_len]

        trg_flat = trg_real.reshape(-1)
        logits_flat = logits_float32.reshape(-1, V)
        return self.ce_loss(logits_flat, trg_flat)

    def _sample_noise_var(self) -> float:
        """
        随机采样信噪比(dB)并转换为噪声方差 sigma^2
        """
        # 使用配置中定义的 SNR 范围
        snr_db = torch.empty(1).uniform_(self.snr_low, self.snr_high).item()
        snr_lin = 10 ** (snr_db / 10)
        # 论文定义 SNR = 1 / (2 * sigma^2)，所以 sigma^2 = 1 / (2 * SNR)
        noise_variance = 1.0 / (2.0 * max(snr_lin, 1e-10))
        return noise_variance