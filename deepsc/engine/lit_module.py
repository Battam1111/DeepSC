# deepsc/engine/lit_module.py - 改进的训练策略
import itertools, math, torch
import pytorch_lightning as pl
from torch.optim import Adam

from deepsc.models.transformer import DeepSC
from deepsc.models.mine import MINE
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score

class LitDeepSC(pl.LightningModule):
    """
    DeepSC端到端训练封装
    
    使用PyTorch Lightning框架封装DeepSC模型，实现端到端训练。
    包含交替优化MINE网络和主网络的策略，以及灵活的学习率调度。
    
    参数:
        cfg: 配置对象，包含模型参数和训练超参数
    """
    automatic_optimization: bool = False

    # --------------------------- 初始化 --------------------------- #
    def __init__(self, cfg):
        super().__init__()
        # 直接存储配置而不使用 save_hyperparameters
        self.cfg = cfg
        # 仍然调用 save_hyperparameters，但不传参数，确保初始化内部状态
        self.save_hyperparameters()
        # 手动将 cfg 存储到 hparams 中
        self.hparams.update({"cfg": cfg})

        # ---- 子网络 ---- #
        self.model   = DeepSC(cfg.model)
        self.channel = get_channel(cfg.data.channel)()
        self.mine    = MINE(cfg.model.latent_dim, hidden=256, activation='relu')

        # ---- 损失 & 常量 ---- #
        self.ce_loss   = torch.nn.CrossEntropyLoss(ignore_index=cfg.data.pad_idx)
        
        # 扩展配置项：MINE 相关参数
        self.mine_update_freq = cfg.train.get('mine_update_freq', 10)  # 默认值为10，但可配置
        self.lambda_mi = cfg.train.lambda_mi
        self.mine_extra_steps = cfg.train.get('mine_extra_steps', 1)   # 每次更新时，MINE可额外训练几步
        self.mine_warmup = cfg.train.get('mine_warmup', 1000)          # MINE 预热期
        
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
            n_var: 噪声方差
            
        返回:
            模型预测的logits
        """
        return self.model(src, n_var, self.channel)

    # --------------------------- 训练 --------------------------- #
    def training_step(self, batch, batch_idx):
        """
        单步训练逻辑
        
        包括主网络更新和MINE网络更新，使用手动优化策略。
        
        参数:
            batch: 当前批次数据
            batch_idx: 批次索引
            
        返回:
            None (使用manual_backward)
        """
        opt_main, opt_mine = self.optimizers()
        
        src = batch.to(self.device)
        n_var = self._sample_noise()
        
        # ① 主网络前向传播
        logits, tx, rx = self.model(src, n_var, self.channel, return_tx_rx=True)
        loss_ce = self._compute_ce(logits, src)
        
        tx_f, rx_f = [z.reshape(-1, z.size(-1)) for z in (tx, rx)]
        
        # ② 更新 MINE 网络（更灵活的策略）
        if self._step_cnt >= self.mine_warmup:  # MINE 预热期后才更新主网络的 MI 损失
            mi_loss_weight = self.lambda_mi
        else:
            mi_loss_weight = 0.0
            
        # 定期更新 MINE
        if self._step_cnt % self.mine_update_freq == 0:
            # 可进行多步更新，增强 MINE 的估计能力
            for _ in range(self.mine_extra_steps):
                opt_mine.zero_grad()
                mi_lb_mine = self.mine(tx_f.detach(), rx_f.detach())
                self.manual_backward(-mi_lb_mine)   # 最大化下界
                opt_mine.step()
        
        # ③ 更新主网络
        mi_lb = self.mine(tx_f, rx_f)
        loss = loss_ce - mi_loss_weight * mi_lb
        
        opt_main.zero_grad()
        self.manual_backward(loss)
        opt_main.step()
        
        # ④ 记录训练指标
        self.log_dict({
            "train_loss": loss,
            "train_ce": loss_ce,
            "train_mi_lb": mi_lb.detach(),
            "mi_weight": mi_loss_weight,
            "lr": opt_main.param_groups[0]["lr"],
        }, on_step=True, prog_bar=True)
        
        # 更新步数计数器
        self._step_cnt += 1

    # --------------------------- 验证 --------------------------- #
    def validation_step(self, batch, _):
        """
        验证步骤
        
        计算验证集上的交叉熵损失和BLEU分数。
        
        参数:
            batch: 验证批次数据
            _: 批次索引（未使用）
            
        返回:
            None (使用self.log记录指标)
        """
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
        """
        配置优化器和学习率调度器
        
        支持两种学习率调度策略：
          1. inverse_sqrt: Transformer原论文中的策略
          2. linear_decay: 线性预热后衰减
        
        返回:
            优化器和学习率调度器的列表
        """
        cfg_t = self.cfg.train
        d_model = self.cfg.model.d_model
        peak_lr = cfg_t.lr
        warmup  = cfg_t.warmup
        lr_type = cfg_t.get("lr_type", "inverse_sqrt")

        # ---- 主网络 ---- #
        opt_main = Adam(self.model.parameters(),
                lr = peak_lr * d_model ** -0.5,  # 恢复 peak_lr 因子
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

        # ---- MINE ---- #
        opt_mine = Adam(self.mine.parameters(), lr=1e-3)

        return [opt_main, opt_mine], [scheduler]

    # --------------------------- 工具 --------------------------- #
    def _compute_ce(self, logits: torch.Tensor, trg: torch.Tensor):
        """
        计算交叉熵损失
        
        参数:
            logits: [batch, seq_len-1, vocab_size] 预测logits
            trg: [batch, seq_len] 目标序列
            
        返回:
            交叉熵损失值
        """
        B, Lm1, V = logits.size()
        return self.ce_loss(logits.reshape(-1, V),
                            trg[:, 1:].reshape(-1))

    def _sample_noise(self) -> float:
        """
        随机采样信噪比并转换为噪声方差
        
        返回:
            噪声标准差 σ
        """
        snr_db = torch.empty(1).uniform_(self.snr_low, self.snr_high).item()
        snr_lin = 10 ** (snr_db / 10)
        # σ = √(1/(2·snr))，符合论文设置
        return math.sqrt(1 / (2 * snr_lin))