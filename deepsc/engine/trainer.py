# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC-Fork/deepsc/engine/trainer.py
# (已修改: 添加 weight_decay, 调整 epsilon, 调整 MINE LR, 修改梯度裁剪逻辑, 确保初始化)
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path
from torch.amp import GradScaler, autocast # 新的导入 (需要 PyTorch 1.10+)
from contextlib import nullcontext
from omegaconf import DictConfig, OmegaConf # 导入 OmegaConf
# 导入 Xavier 初始化
from torch.nn.init import xavier_uniform_

from ..models.mine_strict import MINEStrict
from ..models.deepsc_strict import DeepSCStrict # 严格版本模型
from ..utils.power_norm import power_normalize
from ..metrics.bleu import bleu_score # 导入 BLEU

class TwoPhaseTrainer:
    """
    两阶段训练器 (添加了 AMP 支持, 更新了 AMP API)
    严格按照论文实现两阶段训练流程，并支持自动混合精度训练。

    参数:
        model: DeepSC模型 (应为 DeepSCStrict)
        channel: 信道模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        config: 训练配置 (omegaconf.DictConfig)
    """
    def __init__(self, model, channel, train_loader, val_loader, device, config: DictConfig):
        self.model = model.to(device)
        # --- 新增：应用 Xavier 初始化 ---
        print("  应用 Xavier Uniform 初始化到 DeepSCStrict 模型...")
        self.model.apply(self._init_weights_xavier)

        self.channel = channel.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 创建严格版 MINE 网络 (内部自带初始化)
        self.mine = MINEStrict(latent_dim=config.model.latent_dim).to(device)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.pad_idx)

        # 优化器 (读取配置)
        mine_lr = config.train.get('mine_lr', 1e-3) # <--- 默认 MINE LR 改为 1e-3
        mine_adam_eps = config.train.get('mine_adam_eps', 1e-8)
        print(f"  配置 MINE 优化器 (阶段1): Adam, LR: {mine_lr}, Epsilon: {mine_adam_eps}")
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=mine_lr, eps=mine_adam_eps)

        main_lr = config.train.lr
        main_weight_decay = config.train.get("weight_decay", 5e-4) # <--- 添加 weight decay
        main_adam_eps = config.train.get("adam_eps", 1e-8)
        print(f"  配置主模型优化器 (阶段2): Adam, LR: {main_lr}, Weight Decay: {main_weight_decay}, Epsilon: {main_adam_eps}")
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=main_lr, betas=(0.9, 0.98), eps=main_adam_eps, weight_decay=main_weight_decay) # <--- 添加 weight_decay

        # 学习率调度器 (只为主网络创建，如果需要)
        self.scheduler = self._create_scheduler(self.model_optimizer)

        # ---- AMP 相关 ----
        self.use_amp = str(config.get('precision', 32)) in ['16', 16, '16-mixed'] and device.type == 'cuda'
        self.scaler_mine = GradScaler(device=self.device, enabled=self.use_amp)
        self.scaler_model = GradScaler(device=self.device, enabled=self.use_amp)
        print(f"  TwoPhaseTrainer 初始化: AMP {'启用' if self.use_amp else '禁用'}")

        # 最佳模型记录
        self.best_val_metric = 0.0 # 使用 BLEU 作为指标
        self.save_dir = Path(config.get("checkpoint_dir", "checkpoints_phase"))
        self.save_dir.mkdir(exist_ok=True, parents=True) # 确保 parents=True
        print(f"  检查点将保存到: {self.save_dir.resolve()}")

        # 训练记录
        self.train_losses = []
        self.val_metrics = []

        # 梯度裁剪值
        self.grad_clip = config.model.get('grad_clip', 1.0)

        # SNR 范围
        self.snr_low = config.train.get('snr_low', 5) # 默认改为 5
        self.snr_high = config.train.get('snr_high', 10) # 默认改为 10


    # 新增：Xavier 初始化函数 (与 LitDeepSC 中一致)
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
             nn.init.normal_(module.weight, mean=0, std=module.embedding_dim ** -0.5)

    def _create_scheduler(self, optimizer):
        """创建学习率调度器 (如果配置了)"""
        lr_type = self.config.train.get('lr_type', 'fixed') # 默认改为 fixed
        warmup = self.config.train.get('warmup', 8000)
        d_model = self.config.model.d_model

        if lr_type == 'inverse_sqrt':
             print(f"  使用 inverse_sqrt 学习率调度器 (阶段2), Warmup: {warmup} steps")
             def lr_lambda(step):
                 step += 1
                 arg1 = step ** -0.5
                 arg2 = step * (warmup ** -1.5)
                 scale = d_model ** -0.5
                 factor = scale * min(arg1, arg2) if warmup > 0 else scale * arg1
                 # 修正 LambdaLR 因子计算
                 base_factor = scale * (warmup**-0.5) if warmup > 0 else scale
                 # 使得初始学习率由 Adam 中的 lr 控制，因子从 warmup 开始增长
                 # return factor / base_factor # 这个计算可能需要重新审视
                 # 保持和 LitModule 一致
                 return scale * min(arg1, arg2) if warmup > 0 else scale * arg1

             return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif lr_type == 'fixed':
            print("  使用固定学习率 (阶段2)，无调度器。")
            return None
        else:
             raise ValueError(f"未知的学习率类型: {lr_type}")

    def _sample_noise_var(self):
        """采样噪声方差 sigma^2"""
        # 使用配置中定义的 SNR 范围
        snr_db = torch.empty(1).uniform_(self.snr_low, self.snr_high).item()
        snr_lin = 10 ** (snr_db / 10.0)
        noise_variance = 1.0 / (2.0 * max(snr_lin, 1e-10))
        return noise_variance

    def _compute_ce_loss(self, logits, targets):
        """计算交叉熵损失"""
        logits_float32 = logits.float()
        B, L_pred, V = logits_float32.size()
        # 确保目标长度与 logits 匹配
        trg_real = targets # 假设传入的就是 targets (shape [B, L_pred])
        if L_pred != trg_real.size(1):
            # print(f"警告: _compute_ce_loss 中 logits 长度 ({L_pred}) 与目标长度 ({trg_real.size(1)}) 不匹配。")
            min_len = min(L_pred, trg_real.size(1))
            logits_float32 = logits_float32[:, :min_len, :]
            trg_real = trg_real[:, :min_len]
        return self.criterion(logits_float32.reshape(-1, V), trg_real.reshape(-1))

    def train_mine(self, epochs=10):
        """
        第一阶段：训练互信息估计网络 (使用更新的 AMP API)
        """
        print("======= 阶段1：训练互信息估计网络 =======")
        for param in self.model.parameters(): param.requires_grad = False # 冻结主模型
        self.model.eval()
        self.mine.train() # MINE 设为训练模式

        amp_context = autocast(device_type=self.device.type, enabled=self.use_amp)

        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            pbar = tqdm(self.train_loader, desc=f"阶段1 Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                batch = batch.to(self.device)
                n_var_squared = self._sample_noise_var()

                with torch.no_grad(): # 生成 MINE 输入时不需要梯度
                    with amp_context: # 生成 tx, rx 在 autocast 下
                        _, tx_detached, rx_detached = self.model(batch, n_var_squared, self.channel, return_tx_rx=True)
                        tx_flat = tx_detached.reshape(-1, tx_detached.size(-1))
                        rx_flat = rx_detached.reshape(-1, rx_detached.size(-1))

                non_empty_mask = (tx_flat.abs().sum(dim=1) > 1e-6)
                if non_empty_mask.sum() == 0: continue
                tx_flat = tx_flat[non_empty_mask]
                rx_flat = rx_flat[non_empty_mask]

                self.mine_optimizer.zero_grad(set_to_none=True)

                # MINE forward 通常可以在 autocast 外
                mi_lb = self.mine(tx_flat, rx_flat)
                loss_mine = -mi_lb

                # scaler_mine.scale() 会自动处理 enable=False 的情况
                self.scaler_mine.scale(loss_mine).backward() # 反向传播
                self.scaler_mine.unscale_(self.mine_optimizer) # unscale
                # --- 修改：仅对 MINE 应用梯度裁剪 ---
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), self.grad_clip) # 应用裁剪
                self.scaler_mine.step(self.mine_optimizer) # 更新
                self.scaler_mine.update() # 更新 scaler

                total_loss += loss_mine.item()
                batch_count += 1
                pbar.set_postfix({"MI Loss": f"{-mi_lb.item():.4f}"})

            avg_loss = total_loss / max(1, batch_count)
            print(f"Epoch {epoch+1}/{epochs}, Avg MINE Loss (Neg MI LB): {avg_loss:.6f}")

        self.mine.eval() # 训练完 MINE 后设为评估模式
        for param in self.model.parameters(): param.requires_grad = True # 解冻主模型
        print("MINE训练完成!")


    def train_model(self, epochs):
        """
        第二阶段：训练主网络 (使用更新的 AMP API)
        """
        print("======= 阶段2：训练主网络 =======")
        self.model.train() # 主模型设为训练模式
        for param in self.mine.parameters(): param.requires_grad = False # 冻结 MINE
        self.mine.eval() # MINE 保持评估模式

        amp_context = autocast(device_type=self.device.type, enabled=self.use_amp)
        global_step = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_ce_loss = 0.0
            total_mi_lb = 0.0
            batch_count = 0
            pbar = tqdm(self.train_loader, desc=f"阶段2 Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                batch = batch.to(self.device)
                n_var_squared = self._sample_noise_var()

                self.model_optimizer.zero_grad(set_to_none=True)

                with amp_context:
                    # 主网络前向传播
                    logits, tx, rx = self.model(batch, n_var_squared, self.channel, return_tx_rx=True)
                    # 目标是 src[:, 1:]
                    targets = batch[:, 1:]
                    # 调整 logits 形状以匹配 target
                    logits_aligned = logits[:, :targets.size(1), :] # 确保长度一致
                    ce_loss = self._compute_ce_loss(logits_aligned, targets) # 计算 CE Loss

                    # 计算 MI (使用 MINE)
                    tx_flat = tx.reshape(-1, tx.size(-1))
                    rx_flat = rx.reshape(-1, rx.size(-1))
                    non_empty_mask = (tx_flat.abs().sum(dim=1) > 1e-6)
                    mi_lb = torch.tensor(0.0, device=self.device)
                    if non_empty_mask.sum() > 0:
                        tx_flat = tx_flat[non_empty_mask]
                        rx_flat = rx_flat[non_empty_mask]
                        mi_lb = self.mine(tx_flat, rx_flat) # 使用冻结的 MINE

                # 总损失在 autocast 外计算
                lambda_mi = self.config.train.get('lambda_mi', 0.001) # 读取调整后的 MI 权重
                loss = ce_loss - lambda_mi * mi_lb # 总损失

                # 反向传播和更新
                self.scaler_model.scale(loss).backward()
                self.scaler_model.unscale_(self.model_optimizer)
                # --- 修改：移除主网络的梯度裁剪 ---
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) # <--- 移除或注释掉
                self.scaler_model.step(self.model_optimizer)
                self.scaler_model.update()

                # 更新学习率调度器（如果存在）
                if self.scheduler:
                    self.scheduler.step()
                global_step += 1

                total_ce_loss += ce_loss.item()
                total_mi_lb += mi_lb.item()
                batch_count += 1
                lr_to_log = self.scheduler.get_last_lr()[0] if self.scheduler else self.model_optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "CE Loss": f"{ce_loss.item():.4f}",
                    "MI": f"{mi_lb.item():.4f}",
                    "LR": f"{lr_to_log:.2e}"
                })

            avg_ce_loss = total_ce_loss / max(1, batch_count)
            avg_mi_lb = total_mi_lb / max(1, batch_count)

            val_metric = self.validate() # 计算验证集 BLEU

            self.train_losses.append((avg_ce_loss, avg_mi_lb))
            self.val_metrics.append(val_metric)

            # --- 修改：保存检查点逻辑 ---
            # 保存当前 epoch 的模型
            self.save_checkpoint(f"epoch_{epoch+1}_bleu_{val_metric:.4f}.pt")

            # 如果当前模型是最佳模型，则额外保存一个名为 best_model.pt 的文件
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint("best_model.pt") # 覆盖之前的最佳模型
                print(f"  ** 新的最佳验证 BLEU: {val_metric:.6f} (已保存为 best_model.pt) **")

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s, Train CE: {avg_ce_loss:.6f}, "
                  f"Train MI: {avg_mi_lb:.6f}, Val BLEU: {val_metric:.6f}, Best BLEU: {self.best_val_metric:.6f}")

        print(f"训练完成! 最佳验证 BLEU: {self.best_val_metric:.6f}")


    def validate(self):
        """
        在验证集上评估模型 (使用 BLEU 分数, 更新 AMP API)
        """
        self.model.eval()
        total_bleu = 0.0
        sample_count = 0 # 使用样本数而不是批次数来平均 BLEU
        pad_idx = self.config.data.pad_idx

        amp_context = autocast(device_type=self.device.type, enabled=self.use_amp)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中", leave=False):
                batch = batch.to(self.device)
                n_var_squared = self._sample_noise_var() # 验证时也使用训练的 SNR 范围

                with amp_context:
                    logits, _, _ = self.model(batch, n_var_squared, self.channel, return_tx_rx=True)

                pred_indices = logits.argmax(dim=-1)
                target_indices = batch[:, 1:]
                current_batch_size = batch.size(0)

                # --- 对齐预测和目标的长度 (重要) ---
                pred_len = pred_indices.size(1)
                target_len = target_indices.size(1)
                if pred_len > target_len:
                    pred_aligned = pred_indices[:, :target_len]
                elif pred_len < target_len:
                    padding = torch.full((current_batch_size, target_len - pred_len), pad_idx, device=self.device, dtype=torch.long)
                    pred_aligned = torch.cat([pred_indices, padding], dim=1)
                else:
                    pred_aligned = pred_indices
                # --- 对齐结束 ---

                # bleu_score 返回的是批次中所有句子 BLEU 分数的平均值
                # 为了得到整个验证集的平均 BLEU，我们需要累加所有样本的 BLEU 再除以总样本数
                # 或者，更简单地，累加批次平均 BLEU，然后除以批次数
                # 这里我们选择后者，因为它更常用
                batch_bleu = bleu_score(pred_aligned.cpu(), target_indices.cpu())
                total_bleu += batch_bleu * current_batch_size # 乘以批大小进行加权
                sample_count += current_batch_size

        self.model.train() # 恢复训练模式
        return total_bleu / max(1, sample_count) # 返回加权平均 BLEU

    def save_checkpoint(self, filename: str):
        """保存检查点（包含模型和MINE状态）"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / filename
        checkpoint = {
            'model': self.model.state_dict(),
            'mine': self.mine.state_dict(),
            'config': OmegaConf.to_container(self.config, resolve=True), # 保存可序列化的配置
            'best_val_metric': self.best_val_metric,
            # 可选：保存优化器和调度器状态
            # 'model_optimizer': self.model_optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        try:
            torch.save(checkpoint, save_path)
            print(f"已保存检查点: {save_path}")
        except Exception as e:
            print(f"错误：保存检查点到 {save_path} 失败: {e}")

    # load_checkpoint 函数保持不变

    def train(self):
        """执行完整的两阶段训练"""
        # 第一阶段：训练MINE
        mine_epochs = self.config.train.get('mine_epochs', 10) # 从配置读取
        self.train_mine(epochs=mine_epochs)

        # 第二阶段：训练主网络
        model_epochs = self.config.train.epochs # 从配置读取
        self.train_model(epochs=model_epochs)

        return self.best_val_metric