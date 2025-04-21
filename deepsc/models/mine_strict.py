# -*- coding: utf-8 -*-
"""
严格按照论文实现的互信息神经估计器
===================================

本模块实现了完全遵循论文的互信息神经估计器(MINE)。
与增强版MINE相比，其网络结构更简单，仅包含两层全连接网络。

主要特性:
1. 2层全连接网络，256个隐藏单元
2. 使用数值稳定性技巧估计互信息下界
3. 随机排列采样方法生成边缘分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MINEStrict(nn.Module):
    """
    严格按照论文实现的互信息神经估计器
    
    参数:
        latent_dim: 输入特征维度（默认为16）
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 网络结构：两个全连接层，中间256隐藏单元，与论文完全一致
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z1, z2):
        """
        前向传播计算互信息下界
        
        参数:
            z1: [batch_size, latent_dim] 第一个随机变量
            z2: [batch_size, latent_dim] 第二个随机变量
            
        返回:
            互信息下界估计值
        """
        batch_size = z1.size(0)
        
        # 联合分布样本
        joint = torch.cat([z1, z2], dim=1)
        
        # 边缘分布样本（随机打乱z2）
        idx = torch.randperm(batch_size, device=z1.device)
        marginal = torch.cat([z1, z2[idx]], dim=1)
        
        # 计算T统计量
        t_joint = self.net(joint)
        t_marginal = self.net(marginal)
        
        # 使用数值稳定性技巧
        max_val = torch.max(t_marginal).detach()
        exp_t_marginal = torch.exp(t_marginal - max_val)
        log_mean_exp = torch.log(torch.mean(exp_t_marginal)) + max_val
        
        # 计算互信息下界: E_joint[T] - log(E_marginal[e^T])
        mi_lb = torch.mean(t_joint) - log_mean_exp
        
        return mi_lb