# deepsc/models/mine.py - 完善的互信息估计器
import torch
import torch.nn as nn
import torch.nn.functional as F

class MINE(nn.Module):
    """
    互信息神经估计器
    
    基于论文中的描述，用于估计两组随机变量间的互信息下界。
    使用 Donsker-Varadhan 表示法：
        I(X;Y) >= E_{p(x,y)}[T(x,y)] - log(E_{p(x)p(y)}[e^{T(x,y)}])
    
    参数:
        latent_dim (int): 输入特征维度
        hidden (int): 隐藏层维度
        activation (str): 激活函数类型，可选 'relu', 'tanh', 'leaky_relu'
    """
    def __init__(self, latent_dim: int = 16, hidden: int = 256, activation: str = 'relu'):
        super().__init__()
        # 选择激活函数
        act_fn = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True)
        }[activation]
        
        # 网络结构：更深，更宽
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            act_fn,
            nn.Linear(hidden, hidden),
            act_fn,
            nn.Linear(hidden, hidden // 2),
            act_fn,
            nn.Linear(hidden // 2, 1)
        )
        
        # 初始化权重，促进训练稳定性
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重，使用均值为0、标准差为0.02的正态分布"""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算互信息下界
        
        参数:
            z1: 形状为 [batch_size, latent_dim] 的随机变量X样本
            z2: 形状为 [batch_size, latent_dim] 的随机变量Y样本
            
        返回:
            互信息下界估计值 (标量)
        """
        batch_size = z1.size(0)
        
        # 构建联合分布样本
        joint = torch.cat([z1, z2], dim=-1)
        
        # 构建边缘分布样本 (使用随机排列索引打乱z2)
        idx = torch.randperm(batch_size, device=z1.device)
        marginal = torch.cat([z1, z2[idx]], dim=-1)
        
        # 计算统计量 T
        t_joint = self.net(joint)
        t_marginal = self.net(marginal)
        
        # 使用 LogSumExp 技巧提高数值稳定性
        max_t = torch.max(t_marginal).detach()
        exp_t_marginal = torch.exp(t_marginal - max_t)
        log_exp_mean = torch.log(torch.mean(exp_t_marginal)) + max_t
        
        # 计算互信息下界: E_joint[T] - log(E_marginal[e^T])
        mi_lb = torch.mean(t_joint) - log_exp_mean
        
        return mi_lb