# -*- coding: utf-8 -*-
"""
两阶段训练器
==========
严格按照论文实现两阶段训练流程：
1. 第一阶段：仅训练互信息估计网络(MINE)
2. 第二阶段：训练主网络，使用已训练好的MINE进行互信息估计

此模块提供了独立于PyTorch Lightning的原生PyTorch实现，
以确保完全控制训练过程。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import os
from pathlib import Path

from ..models.mine_strict import MINEStrict
from ..models.deepsc_strict import DeepSCStrict
from ..utils.power_norm import power_normalize

class TwoPhaseTrainer:
    """
    两阶段训练器
    
    严格按照论文实现两阶段训练流程
    
    参数:
        model: DeepSC模型
        channel: 信道模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        config: 训练配置
    """
    def __init__(self, model, channel, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.channel = channel.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 创建MINE网络
        self.mine = MINEStrict(latent_dim=config.model.latent_dim).to(device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.pad_idx)
        
        # 优化器
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=1e-3)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr)
        
        # 学习率调度器
        self.scheduler = self._create_scheduler(self.model_optimizer)
        
        # 最佳模型记录
        self.best_val_metric = 0.0
        self.save_dir = Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        # 训练记录
        self.train_losses = []
        self.val_metrics = []
        
    def _create_scheduler(self, optimizer):
        """创建学习率调度器"""
        warmup = self.config.train.get('warmup', 4000)
        lr_type = self.config.train.get('lr_type', 'inverse_sqrt')
        
        if lr_type == 'inverse_sqrt':
            # Transformer原始论文使用的逆平方根调度
            def lr_lambda(step):
                step += 1  # 避免 step=0 时除零
                return min(step ** -0.5, step * warmup ** -1.5)
        else:
            # 线性预热，然后线性衰减
            total_steps = len(self.train_loader) * self.config.train.epochs
            def lr_lambda(step):
                if step < warmup:
                    return float(step) / float(max(1, warmup))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _sample_noise(self):
        """采样噪声方差，按照SNR范围均匀采样"""
        snr_low = self.config.train.get('snr_low', 0)
        snr_high = self.config.train.get('snr_high', 15)
        
        # 均匀采样SNR (dB)
        snr_db = torch.empty(1).uniform_(snr_low, snr_high).item()
        
        # 转换为线性SNR
        snr_lin = 10 ** (snr_db / 10)
        
        # 计算噪声方差
        n_var = 1.0 / (2.0 * snr_lin)
        
        return np.sqrt(n_var)  # 返回标准差
    
    def _compute_ce_loss(self, logits, targets):
        """计算交叉熵损失"""
        B, Lm1, V = logits.size()
        return self.criterion(logits.reshape(-1, V), targets.reshape(-1))
    
    def train_mine(self, epochs=10):
        """
        第一阶段：训练互信息估计网络
        
        参数:
            epochs: 训练轮数
        """
        print("======= 阶段1：训练互信息估计网络 =======")
        
        # 冻结主网络参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        self.mine.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(self.train_loader, desc=f"阶段1 Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                batch = batch.to(self.device)
                n_var = self._sample_noise()
                
                # 使用主网络前向传播生成信道输入/输出
                with torch.no_grad():
                    _, tx, rx = self.model(batch, n_var, self.channel, return_tx_rx=True)
                
                # 展平信道符号
                tx_flat = tx.reshape(-1, tx.size(-1))
                rx_flat = rx.reshape(-1, rx.size(-1))
                
                # 过滤掉空行 (PAD)
                non_empty_mask = (tx_flat.abs().sum(dim=1) > 1e-6)
                if non_empty_mask.sum() == 0:
                    continue  # 跳过空批次
                    
                tx_flat = tx_flat[non_empty_mask]
                rx_flat = rx_flat[non_empty_mask]
                
                # 训练MINE
                self.mine_optimizer.zero_grad()
                # 计算互信息下界并取负值，因为我们要最大化互信息
                mi_loss = -self.mine(tx_flat, rx_flat)
                mi_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), self.config.model.get('grad_clip', 1.0))
                self.mine_optimizer.step()
                
                total_loss += mi_loss.item()
                batch_count += 1
                
                pbar.set_postfix({"MI Loss": mi_loss.item()})
            
            avg_loss = total_loss / max(1, batch_count)
            print(f"Epoch {epoch+1}/{epochs}, Avg MINE Loss: {avg_loss:.6f}")
        
        # 训练后设置MINE为评估模式
        self.mine.eval()
        
        # 解冻主网络参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("MINE训练完成!")
    
    def train_model(self, epochs):
        """
        第二阶段：训练主网络
        
        参数:
            epochs: 训练轮数
        """
        print("======= 阶段2：训练主网络 =======")
        
        self.model.train()
        
        # 冻结MINE网络参数
        for param in self.mine.parameters():
            param.requires_grad = False
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_ce_loss = 0.0
            total_mi_lb = 0.0
            batch_count = 0
            
            pbar = tqdm(self.train_loader, desc=f"阶段2 Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                batch = batch.to(self.device)
                n_var = self._sample_noise()
                
                # 前向传播
                logits, tx, rx = self.model(batch, n_var, self.channel, return_tx_rx=True)
                
                # 计算交叉熵损失
                targets = batch[:, 1:]  # 目标是输入右移一位
                ce_loss = self._compute_ce_loss(logits, targets)
                
                # 计算互信息下界
                tx_flat = tx.reshape(-1, tx.size(-1))
                rx_flat = rx.reshape(-1, rx.size(-1))
                
                # 过滤掉空行 (PAD)
                non_empty_mask = (tx_flat.abs().sum(dim=1) > 1e-6)
                if non_empty_mask.sum() > 0:
                    tx_flat = tx_flat[non_empty_mask]
                    rx_flat = rx_flat[non_empty_mask]
                    mi_lb = self.mine(tx_flat, rx_flat)
                else:
                    mi_lb = torch.tensor(0.0, device=self.device)
                
                # 总损失
                lambda_mi = self.config.train.get('lambda_mi', 0.01)
                loss = ce_loss - lambda_mi * mi_lb
                
                # 反向传播
                self.model_optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.model.get('grad_clip', 1.0))
                self.model_optimizer.step()
                
                # 更新学习率
                self.scheduler.step()
                
                # 记录损失
                total_ce_loss += ce_loss.item()
                total_mi_lb += mi_lb.item()
                batch_count += 1
                
                pbar.set_postfix({
                    "CE Loss": ce_loss.item(),
                    "MI": mi_lb.item(),
                    "LR": self.scheduler.get_last_lr()[0]
                })
            
            # 计算平均损失
            avg_ce_loss = total_ce_loss / max(1, batch_count)
            avg_mi_lb = total_mi_lb / max(1, batch_count)
            
            # 验证
            val_metric = self.validate()
            
            # 记录训练信息
            self.train_losses.append((avg_ce_loss, avg_mi_lb))
            self.val_metrics.append(val_metric)
            
            # 保存最佳模型
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint(f"best_model_epoch{epoch+1}.pt")
            
            # 计算耗时
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s, CE Loss: {avg_ce_loss:.6f}, "
                  f"MI: {avg_mi_lb:.6f}, Val Metric: {val_metric:.6f}, Best: {self.best_val_metric:.6f}")
        
        print(f"训练完成! 最佳验证指标: {self.best_val_metric:.6f}")
    
    def validate(self):
        """
        在验证集上评估模型
        
        返回:
            验证指标 (BLEU分数)
        """
        self.model.eval()
        
        from ..metrics.bleu import bleu_score
        
        total_bleu = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中", leave=False):
                batch = batch.to(self.device)
                n_var = self._sample_noise()
                
                # 前向传播
                logits = self.model(batch, n_var, self.channel)
                
                # 硬判决
                pred_indices = logits.argmax(dim=-1)
                
                # 计算BLEU分数
                batch_bleu = bleu_score(pred_indices, batch[:, 1:])
                total_bleu += batch_bleu
                batch_count += 1
        
        self.model.train()
        
        return total_bleu / max(1, batch_count)
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'model': self.model.state_dict(),
            'mine': self.mine.state_dict(),
            'config': self.config,
            'best_val_metric': self.best_val_metric
        }
        torch.save(checkpoint, self.save_dir / filename)
        print(f"已保存检查点: {self.save_dir / filename}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.mine.load_state_dict(checkpoint['mine'])
        self.best_val_metric = checkpoint['best_val_metric']
        print(f"已加载检查点: {self.save_dir / filename}")
    
    def train(self):
        """执行完整的两阶段训练"""
        # 第一阶段：训练MINE
        mine_epochs = self.config.train.get('mine_epochs', 10)
        self.train_mine(epochs=mine_epochs)
        
        # 第二阶段：训练主网络
        model_epochs = self.config.train.epochs
        self.train_model(epochs=model_epochs)
        
        return self.best_val_metric