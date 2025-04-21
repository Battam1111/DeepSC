# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC-Fork/deepsc/engine/lit_module.py
# --------------------------------------------------------------------------------
# 导入必要的库
import itertools, math, torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
# 导入新版 AMP 工具 (需要 PyTorch 1.10+)
from torch.amp import GradScaler, autocast
# 导入 Xavier 初始化
from torch.nn.init import xavier_uniform_
# 导入 nn 模块
import torch.nn as nn

# 动态导入模型，增加代码灵活性，即使某些模型文件不存在也不会在导入时报错
try: from deepsc.models.transformer import DeepSC # 标准版模型
except ImportError: DeepSC = None
try: from deepsc.models.deepsc_strict import DeepSCStrict # 严格版模型
except ImportError: DeepSCStrict = None
try: from deepsc.models.mine import MINE # 标准版 MINE
except ImportError: MINE = None
try: from deepsc.models.mine_strict import MINEStrict # 严格版 MINE
except ImportError: MINEStrict = None

# 导入其他必要的自定义模块
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score
from contextlib import nullcontext # 用于在不使用 AMP 时创建无操作的上下文管理器

class LitDeepSC(pl.LightningModule):
    """
    DeepSC 端到端训练的 Pytorch Lightning 封装模块 (修复版)

    功能:
    - 支持标准版 (DeepSC) 和严格版 (DeepSCStrict) 模型架构。
    - 支持 MINE (标准版/严格版) 网络的训练与集成。
    - 实现基于 PyTorch Lightning 的手动优化 (`automatic_optimization = False`)。
    - 解决了手动优化、多优化器、条件更新与自动混合精度 (AMP) 的复杂交互问题。
    - 包含必要的权重初始化、梯度裁剪（仅 MINE）、学习率调度、损失计算和评估逻辑。
    - 对齐了关键超参数（Weight Decay, Adam Epsilon, MINE LR, MI Weight, SNR 范围）以接近原始论文设置。

    参数:
        cfg: Hydra 配置对象 (omegaconf.DictConfig)，包含所有模型和训练参数。
    """
    automatic_optimization: bool = False # 明确指定使用手动优化

    # --------------------------- 初始化 (Constructor) --------------------------- #
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # 保存完整配置

        # 将配置保存为超参数，便于 Lightning 记录和加载 (过滤不可序列化项)
        hparams_to_save = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list, tuple))}
        if isinstance(cfg.get('model'), dict):
             hparams_to_save['model_params'] = {mk: mv for mk, mv in cfg.model.items() if isinstance(mv, (int, float, str, bool))}
        self.save_hyperparameters(hparams_to_save)
        # 也可以直接保存整个 cfg 对象，方便访问，但可能无法完全被 Lightning 保存
        self.hparams.cfg = cfg

        # ---- AMP (自动混合精度) 相关设置 ----
        # 检查是否启用 AMP，并且 CUDA 是否可用
        self.use_amp = str(self.cfg.get('precision', 32)) in ['16', 16, '16-mixed'] and torch.cuda.is_available()
        # 根据设备类型确定 AMP 上下文使用的设备
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  LitDeepSC 初始化: AMP {'启用' if self.use_amp else '禁用'} (设备类型: {self.device_type})")

        # ---- 子网络初始化 ---- #
        # 根据配置中的 'strict_model' 标志决定加载哪个模型和 MINE 网络
        is_strict = cfg.get("strict_model", False)
        if is_strict:
             # 检查严格模式所需的类是否已成功导入
             if DeepSCStrict is None or MINEStrict is None:
                 raise ImportError("错误：严格模式需要 'deepsc.models.deepsc_strict' 和 'deepsc.models.mine_strict' 模块。")
             print("  加载严格版本的 DeepSCStrict 和 MINEStrict")
             self.model = DeepSCStrict(cfg.model) # 使用配置初始化严格模型
             self.mine = MINEStrict(cfg.model.latent_dim) # 使用配置初始化严格 MINE
             # 对严格模型应用 Xavier 初始化
             print("  应用 Xavier Uniform 初始化到 DeepSCStrict...")
             self.model.apply(self._init_weights_xavier)
        else:
             # 检查标准模式所需的类是否已成功导入
             if DeepSC is None or MINE is None:
                 raise ImportError("错误：标准模式需要 'deepsc.models.transformer' 和 'deepsc.models.mine' 模块。")
             print("  加载标准版本的 DeepSC 和 MINE")
             self.model = DeepSC(cfg.model) # 使用配置初始化标准模型
             self.mine = MINE(cfg.model.latent_dim, hidden=256, activation='relu') # 使用配置初始化标准 MINE
             # 对标准模型应用 Xavier 初始化
             print("  应用 Xavier Uniform 初始化到 DeepSC...")
             self.model.apply(self._init_weights_xavier)

        # 检查 MINE 网络是否成功初始化
        if self.mine is None:
             print("警告：MINE 网络未能成功初始化。互信息相关功能将不可用。")

        # 获取并初始化信道模型
        self.channel = get_channel(cfg.data.channel)()

        # ---- 损失函数和训练常量 ---- #
        # 交叉熵损失，忽略填充符
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.data.pad_idx)
        # MINE 更新频率 (每多少步更新一次 MINE)
        self.mine_update_freq = cfg.train.get('mine_update_freq', 1)
        # 互信息损失权重 (从配置读取，应设为 0.001 或 0.0009 以匹配原始论文)
        self.lambda_mi = cfg.train.lambda_mi
        # 每次更新 MINE 时执行的额外优化步骤数
        self.mine_extra_steps = cfg.train.get('mine_extra_steps', 1)
        # MINE 预热步数 (多少步之后才开始将 MI 损失加入总损失)
        self.mine_warmup = cfg.train.get('mine_warmup', 0) # 默认禁用预热
        # 梯度裁剪值 (仅用于 MINE 网络)
        self.grad_clip = cfg.model.get('grad_clip', 1.0)
        # 训练时采样 SNR 的范围 (dB)
        self.snr_low = cfg.train.get('snr_low', 5)  # 默认 5 dB
        self.snr_high = cfg.train.get('snr_high', 10) # 默认 10 dB
        # 全局训练步数计数器
        self._step_cnt = 0

    # Xavier 初始化辅助函数
    def _init_weights_xavier(self, module):
        """
        递归地将 Xavier Uniform 初始化应用于模型中的线性层和 LayerNorm 层，
        并对 Embedding 层使用正态分布初始化。
        """
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight) # 对权重使用 Xavier 初始化
            if module.bias is not None:
                 nn.init.constant_(module.bias, 0) # 将偏置初始化为 0
        elif isinstance(module, nn.LayerNorm):
             nn.init.constant_(module.weight, 1.0) # LayerNorm 权重初始化为 1
             nn.init.constant_(module.bias, 0) # LayerNorm 偏置初始化为 0
        elif isinstance(module, nn.Embedding):
             # 对 Embedding 层使用正态分布初始化，标准差与维度相关
             if hasattr(module, 'embedding_dim'):
                 nn.init.normal_(module.weight, mean=0, std=module.embedding_dim ** -0.5)
             else:
                 # 理论上 nn.Embedding 总有 embedding_dim，此警告很少触发
                 print(f"警告: 尝试初始化 Embedding 层 {module} 时未找到 embedding_dim 属性。跳过初始化。")

    # --------------------------- 模型前向传播 (Forward) --------------------------- #
    def forward(self, src: torch.Tensor, n_var: float):
        """
        模型前向传播 (主要用于推理或外部直接调用模型时)。
        训练时通常在 training_step 内部直接调用 self.model(...)。

        参数:
            src: 输入序列张量 [batch, seq_len]
            n_var: 噪声方差 (sigma^2)

        返回:
            模型输出的 logits [batch, seq_len-1, vocab_size] (注意长度减 1)
        """
        # 根据是否启用 AMP 选择合适的上下文管理器
        amp_context = autocast(device_type=self.device_type, enabled=self.use_amp)
        with amp_context:
            # 调用底层模型 (DeepSC 或 DeepSCStrict) 的 forward 方法
            # 假设其签名统一为 (src, n_var, channel, return_tx_rx)
            # 设置 return_tx_rx=False 因为推理时通常只需要 logits
            output = self.model(src, n_var, self.channel, return_tx_rx=False)
            # 如果模型返回了元组 (例如包含了 tx, rx)，只取第一个元素 (logits)
            if isinstance(output, tuple):
                return output[0]
            else:
                return output

    # --------------------------- 训练步骤 (Training Step) --------------------------- #
    def training_step(self, batch, batch_idx):
        """
        单个训练批次的处理逻辑。
        包含 MINE 网络和主网络的更新，并正确处理手动优化和 AMP。

        参数:
            batch: 当前批次的数据张量 [batch, seq_len]
            batch_idx: 当前批次的索引

        返回:
            None (因为使用了手动优化)
        """
        # 安全地获取优化器
        optimizers = self.optimizers()
        # 处理优化器，确保兼容单个优化器或优化器列表的情况
        if not isinstance(optimizers, list):
            # 如果只返回单个优化器，则将其包装为列表
            optimizers = [optimizers]
        
        # 获取主优化器和MINE优化器（如果存在）
        opt_main = optimizers[0]
        opt_mine = optimizers[1] if len(optimizers) > 1 else None
        
        # 将数据移到计算设备
        src = batch.to(self.device)
        # 采样当前批次的噪声方差
        n_var_squared = self._sample_noise_var()
        
        # 获取混合精度上下文
        amp_context = autocast(device_type=self.device_type, enabled=self.use_amp)
        
        # 检查 MINE 网络是否存在且可训练
        is_mine_trainable = (self.mine is not None) and any(p.requires_grad for p in self.mine.parameters())
        mine_did_update = False # 标记 MINE 在此步骤是否实际被更新了

        # --- 更新 MINE 网络 (仅当 MINE 可训练且达到更新频率时) ---
        if is_mine_trainable and opt_mine is not None and (self._step_cnt % self.mine_update_freq == 0):
            # 1. 准备 MINE 输入数据
            self.model.eval() # 设为评估模式
            with torch.no_grad():
                with amp_context:
                    _, tx_detached, rx_detached = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
                    # 将特征展平为 [batch*seq_len, feature_dim]
                    tx_f_detached = tx_detached.reshape(-1, tx_detached.size(-1))
                    rx_f_detached = rx_detached.reshape(-1, rx_detached.size(-1))
            self.model.train() # 恢复训练模式

            # 过滤无效特征
            non_empty_mask_detached = (tx_f_detached.abs().sum(dim=1) > 1e-6)
            if non_empty_mask_detached.sum() > 0:
                tx_f_detached = tx_f_detached[non_empty_mask_detached]
                rx_f_detached = rx_f_detached[non_empty_mask_detached]

                # 2. 更新 MINE 网络
                self.mine.train()
                for _ in range(self.mine_extra_steps):
                    opt_mine.zero_grad()
                    
                    # MINE 前向传播
                    with amp_context:
                        mi_lb_mine = self.mine(tx_f_detached, rx_f_detached)
                        loss_mine = -mi_lb_mine
                    
                    # 反向传播（通过Lightning API）
                    self.manual_backward(loss_mine)
                    
                    # 梯度裁剪
                    clip_grad_norm_(self.mine.parameters(), self.grad_clip)
                    
                    # 优化器步骤
                    opt_mine.step()
                    mine_did_update = True
                
                self.mine.eval() # 切换回评估模式

        # --- 更新主网络 (DeepSC Model) ---
        self.model.train()
        opt_main.zero_grad()
        
        with amp_context:
            # 1. 主网络前向传播
            logits, tx, rx = self.model(src, n_var_squared, self.channel, return_tx_rx=True)
            
            # 2. 计算交叉熵损失
            loss_ce = self._compute_ce(logits, src)
            
            # 3. 计算互信息下界
            mi_lb = torch.tensor(0.0, device=self.device)
            if self.mine is not None:
                tx_f = tx.reshape(-1, tx.size(-1))
                rx_f = rx.reshape(-1, rx.size(-1))
                non_empty_mask = (tx_f.abs().sum(dim=1) > 1e-6)
                if non_empty_mask.sum() > 0:
                    tx_f = tx_f[non_empty_mask]
                    rx_f = rx_f[non_empty_mask]
                    self.mine.eval()
                    mi_lb = self.mine(tx_f, rx_f)
        
        # 4. 计算总损失
        mi_loss_weight = self.lambda_mi if self._step_cnt >= self.mine_warmup else 0.0
        loss = loss_ce - mi_loss_weight * mi_lb.to(loss_ce.device)
        
        # 5. 通过Lightning的API进行反向传播
        self.manual_backward(loss)
        
        # 6. 优化器步骤
        opt_main.step()
        
        # 记录训练指标
        self.log_dict({
            "train_loss": loss.item(),          # 总损失
            "train_ce": loss_ce.item(),        # 交叉熵损失部分
            "train_mi_lb": mi_lb.item(),       # 互信息下界估计
            "mi_weight": float(mi_loss_weight),# 当前使用的 MI 权重
            "lr": opt_main.param_groups[0]["lr"],# 主网络当前学习率
        }, on_step=True, prog_bar=True, logger=True)

        # 更新全局步数计数器
        self._step_cnt += 1

    # --------------------------- 验证步骤 (Validation Step) --------------------------- #
    @torch.no_grad() # 明确使用 no_grad 装饰器，替代 with torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        单个验证批次的处理逻辑。
        计算验证集上的交叉熵损失和 BLEU 分数。
        """
        src = batch.to(self.device)
        # 验证时也使用与训练时相同的 SNR 范围进行采样，以评估模型在相似条件下的表现
        n_var_squared = self._sample_noise_var()
        # 获取 AMP 上下文，即使验证时通常不进行梯度计算，使用 autocast 也可以匹配训练时的数值精度
        amp_context = autocast(device_type=self.device_type, enabled=self.use_amp)

        with amp_context:
             # 模型前向传播获取 logits
             # 验证时也需要 channel 和 return_tx_rx=True (如果后面要算 MI) 或 False
             # 为了与 training_step 对称，获取 tx, rx 但不使用它们计算梯度
             # 如果完全不需要 tx, rx，可以设为 False
             logits, _, _ = self.model(src, n_var_squared, self.channel, return_tx_rx=True) # 获取 logits
             # 计算验证集上的交叉熵损失
             val_ce = self._compute_ce(logits, src)

        # ----- BLEU 分数计算 (在 CPU 上进行) -----
        # 获取预测的 token 索引 (取概率最大的)
        pred_indices = logits.argmax(-1)
        # 获取目标的 token 索引 (去掉 <START> token)
        target_indices = src[:, 1:]

        # 对齐预测和目标的序列长度，以便比较
        pred_len = pred_indices.size(1)
        target_len = target_indices.size(1)
        if pred_len > target_len:
            pred_aligned = pred_indices[:, :target_len]
        elif pred_len < target_len:
            # 如果预测短了，用 pad_idx 填充
            padding = torch.full((pred_indices.size(0), target_len - pred_len), self.cfg.data.pad_idx, device=self.device, dtype=torch.long)
            pred_aligned = torch.cat([pred_indices, padding], dim=1)
        else:
            # 长度一致，无需处理
            pred_aligned = pred_indices

        # 计算 BLEU 分数 (需要在 CPU 上进行)
        val_bleu = bleu_score(pred_aligned.cpu(), target_indices.cpu())

        # 记录验证指标 (确保 sync_dist=True 以在分布式训练中正确聚合)
        self.log_dict({
            "val_ce": val_ce.item(), # 记录验证集交叉熵
            "val_bleu": val_bleu,   # 记录验证集 BLEU 分数
        }, sync_dist=True, prog_bar=True, on_epoch=True, logger=True) # on_epoch=True 表示在 epoch 结束时聚合

    # ------------------------ 优化器与学习率调度器配置 -------------------- #
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。
        返回优化器列表和关联的调度器配置列表。
        """
        cfg_t = self.cfg.train # 训练配置
        d_model = self.cfg.model.d_model # 模型维度
        peak_lr = cfg_t.lr # 基础学习率 (或峰值学习率)
        warmup = cfg_t.warmup # 预热步数
        lr_type = cfg_t.get("lr_type", "fixed") # 学习率调度类型，默认为 fixed
        weight_decay = cfg_t.get("weight_decay", 5e-4) # 权重衰减，默认 5e-4
        adam_eps = cfg_t.get("adam_eps", 1e-8) # Adam Epsilon，默认 1e-8

        # ---- 配置主网络优化器 (opt_main) ---- #
        print(f"  配置主网络优化器: Adam, LR Type: {lr_type}, Peak LR: {peak_lr}, Weight Decay: {weight_decay}, Epsilon: {adam_eps}")
        # 定义不需要应用权重衰减的参数名称模式
        no_decay = ['bias', 'LayerNorm.weight', 'norm1.weight', 'norm2.weight', 'norm3.weight']
        # 将模型参数分组，区分是否应用权重衰减
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        # 过滤掉参数列表为空的组 (例如，如果所有参数都被冻结了)
        optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group['params']]

        # 检查是否有可训练参数
        if not optimizer_grouped_parameters and any(p.requires_grad for p in self.model.parameters()):
             # 如果有可训练参数但分组后为空，可能是 no_decay 规则问题，回退到不分组
             print("警告：未能成功分组参数进行权重衰减，将对所有主模型参数应用默认权重衰减。")
             opt_main = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=peak_lr, betas=(0.9, 0.98), eps=adam_eps, weight_decay=weight_decay)
        elif not optimizer_grouped_parameters:
             print("警告：主模型中没有找到需要优化的参数！将创建一个空的优化器。")
             # 创建一个象征性的优化器，避免后续代码出错
             opt_main = Adam([torch.nn.Parameter(torch.empty(0))], lr=peak_lr)
        else:
             # 使用分组参数创建优化器
             opt_main = Adam(optimizer_grouped_parameters, lr=peak_lr, betas=(0.9, 0.98), eps=adam_eps)

        # ---- 配置学习率调度器 (如果需要) ---- #
        scheduler_config = None
        if lr_type == "inverse_sqrt":
            # 使用 inverse square root 调度器，带 warmup
            print(f"  使用 inverse_sqrt 学习率调度器, Warmup: {warmup} steps")
            def lr_lambda(step: int):
                 # 根据 Transformer 论文的 Noam / Vaswani 调度器公式
                 step += 1 # 步数从 1 开始
                 arg1 = step ** -0.5
                 arg2 = step * (warmup ** -1.5) if warmup > 0 else 0
                 # 修正：确保因子与 d_model 相关
                 return (d_model ** -0.5) * min(arg1, arg2) if warmup > 0 else (d_model ** -0.5) * arg1

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda)
            # 配置调度器按步更新
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif lr_type == "fixed":
             # 使用固定的学习率，不需要调度器
             print("  使用固定学习率，无调度器。")
             scheduler_config = None
        else:
            # 不支持的调度器类型
            raise ValueError(f"未知的 lr_type: {lr_type}")

        # ---- 配置 MINE 网络优化器 (opt_mine) ---- #
        opt_mine = None # 默认为 None
        if self.mine is not None: # 仅当 MINE 网络实例存在时才配置
             mine_lr = self.cfg.train.get("mine_lr", 1e-3) # MINE 学习率，默认 1e-3
             mine_adam_eps = self.cfg.train.get("mine_adam_eps", 1e-8) # MINE Epsilon
             # 检查 MINE 网络是否有可训练的参数
             if any(p.requires_grad for p in self.mine.parameters()):
                 print(f"  配置 MINE 优化器: Adam, LR: {mine_lr}, Epsilon: {mine_adam_eps}")
                 # 创建 MINE 优化器 (通常不需要权重衰减)
                 opt_mine = Adam(self.mine.parameters(), lr=mine_lr, eps=mine_adam_eps)
             else:
                 print("  MINE 网络已配置，但其参数已被冻结，不为其创建优化器。")
        else:
             print("  MINE 网络未初始化或不存在，跳过其优化器配置。")

        # ---- 组装返回值 ---- #
        optimizers_list = [opt_main]
        if opt_mine: # 只有当 MINE 优化器被创建时才加入列表
            optimizers_list.append(opt_mine)

        # 最终返回值处理
        if scheduler_config:
            return optimizers_list, [scheduler_config]
        else:
            return optimizers_list

    # --------------------------- 工具方法 (Utilities) --------------------------- #
    def _compute_ce(self, logits: torch.Tensor, trg: torch.Tensor):
        """
        计算交叉熵损失，处理长度可能不匹配的情况。

        参数:
            logits: 模型输出的 Logits [batch, L_pred, vocab_size]
            trg: 目标序列 (包含 <START>) [batch, L_src]

        返回:
            标量交叉熵损失值
        """
        logits_float32 = logits.float() # 确保使用 float32 计算损失
        # 目标是去掉 <START> 的部分 trg[:, 1:]
        trg_real = trg[:, 1:]

        # 获取 logits 和 target 的序列长度
        B, L_pred, V = logits_float32.size()
        L_trg = trg_real.size(1)

        # 如果预测长度和目标长度不匹配（可能发生在 forward 逻辑或数据处理错误时）
        if L_pred != L_trg:
            # 采取截断策略，保证计算可以进行，但应留意此警告
            min_len = min(L_pred, L_trg)
            logits_float32 = logits_float32[:, :min_len, :]
            trg_real = trg_real[:, :min_len]

        # 将 logits 和 target 展平以便计算损失
        trg_flat = trg_real.reshape(-1)             # [batch * seq_len]
        logits_flat = logits_float32.reshape(-1, V) # [batch * seq_len, vocab_size]

        # 使用预定义的 CrossEntropyLoss 计算损失 (会自动处理 ignore_index)
        return self.ce_loss(logits_flat, trg_flat)

    def _sample_noise_var(self) -> float:
        """
        根据配置的 SNR 范围 (dB) 随机采样信噪比，并计算对应的噪声方差 sigma^2。
        假定信号功率归一化为 1。

        返回:
            噪声方差 (float)
        """
        # 在配置的 [snr_low, snr_high] 范围内均匀采样一个 SNR 值 (dB)
        snr_db = torch.empty(1).uniform_(self.snr_low, self.snr_high).item()
        # 将 dB 值转换为线性值 SNR_linear = 10^(SNR_dB / 10)
        snr_lin = 10 ** (snr_db / 10.0)
        # 根据论文中的定义 SNR = 1 / (2 * sigma^2) 来计算噪声方差 sigma^2
        # sigma^2 = 1 / (2 * SNR_linear)
        # 添加 clamp_min(1e-10) 防止 SNR 接近 0 时除以零
        noise_variance = 1.0 / (2.0 * max(snr_lin, 1e-10))
        return noise_variance