# scripts/finetune.py - 更新后的迁移学习脚本 (修复 proj/output_proj 命名不一致问题)
# -*- coding: utf-8 -*-
"""
一键迁移学习 (使用专用配置文件 configs/finetune.yaml)
===================================================================
支持两种迁移学习场景：
1. 新信道环境：冻结语义层，只训练信道层
2. 新领域文本：冻结信道层，只训练语义层

用法示例：
▶ 新信道（AWGN→Rayleigh）：
  python -m scripts.finetune \
      ckpt_path=/path/best.ckpt \
      finetune.mode=channel finetune.new_channel=RAYLEIGH \
      strict_model=True # 如果预训练模型是严格版

▶ 新语料域（医疗文本）：
  python -m scripts.finetune \
      ckpt_path=/path/best.ckpt \
      finetune.mode=domain data.train_pkl=/new/train.pkl \
      data.val_pkl=/new/val.pkl \
      data.vocab_json=/new/vocab.json \
      strict_model=True # 如果预训练模型是严格版
"""
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict # 导入 open_dict
from hydra.utils import to_absolute_path
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm
import sys
import traceback # 用于打印错误
import torch.nn as nn # 导入 nn

# 确保相关模块可以被导入
try:
    from deepsc.engine.lit_module import LitDeepSC
    from deepsc.data.europarl import make_dataloader
    from deepsc.data.vocab import Vocab
    from deepsc.utils.seed import set_global_seed
    from deepsc.utils.freeze import freeze, unfreeze, count_trainable_params
    from deepsc.models import get_channel
except ImportError:
    # 如果在 scripts 目录下直接运行可能需要调整路径
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from deepsc.engine.lit_module import LitDeepSC
    from deepsc.data.europarl import make_dataloader
    from deepsc.data.vocab import Vocab
    from deepsc.utils.seed import set_global_seed
    from deepsc.utils.freeze import freeze, unfreeze, count_trainable_params
    from deepsc.models import get_channel

# ===========================================================
#  使用 config_name="finetune" 加载 finetune.yaml
# ===========================================================
@hydra.main(config_path="../configs", config_name="finetune", version_base=None)
def main(cfg: DictConfig):
    """
    迁移学习主函数

    参数:
        cfg: Hydra配置对象 (已自动合并 base.yaml 和 finetune/base.yaml)
    """
    # ---------- 1. 解析并校验路径 ----------
    if "ckpt_path" not in cfg:
         raise ValueError("错误：缺少 'ckpt_path' 配置项。请在命令行或配置文件中提供预训练模型的路径。")
    ckpt_path = to_absolute_path(cfg.ckpt_path)

    try:
        train_pkl = to_absolute_path(cfg.data.train_pkl)
        val_pkl = to_absolute_path(cfg.data.val_pkl)
        vocab_json = to_absolute_path(cfg.data.vocab_json)
    except Exception as e:
        raise ValueError(f"错误：解析数据路径时出错（data.train_pkl, data.val_pkl, data.vocab_json）。请确保它们在配置中正确定义。原始错误：{e}")

    print("--- 检查文件路径 ---")
    required_paths = {'ckpt_path': ckpt_path}
    finetune_mode = cfg.finetune.mode # 从 finetune 组获取模式
    if finetune_mode == 'domain':
        required_paths.update({
            'data.train_pkl': train_pkl,
            'data.val_pkl': val_pkl,
            'data.vocab_json': vocab_json
        })
    elif finetune_mode == 'channel':
         required_paths['data.vocab_json'] = vocab_json
    else:
         raise ValueError(f"错误：无效的 finetune.mode 配置值 '{finetune_mode}'。必须是 'channel' 或 'domain'。")

    for name, p_str in required_paths.items():
        p = Path(p_str)
        if not p.exists():
            raise FileNotFoundError(f"错误：未找到文件 '{p}' (配置项: {name})，请检查配置或使用CLI覆盖参数")
        print(f"  [√] 文件存在: {p} (来自 {name})")


    # ---------- 2. 随机种子 & 加载预训练模型 ----------
    set_global_seed(cfg.seed)
    print(f"\n--- 加载与准备模型 ---")
    print(f"加载源模型: {ckpt_path}")

    print(f"加载词表: {vocab_json}")
    vocab = Vocab.load(vocab_json)
    actual_vocab_size = len(vocab)
    pad_idx = vocab.token2idx['<PAD>']
    print(f"  实际词表大小: {actual_vocab_size}")
    print(f"  Pad Index: {pad_idx}")

    try:
        with open_dict(cfg): # 允许修改配置
            cfg.model.vocab_size = actual_vocab_size
            cfg.model.pad_idx = pad_idx
            cfg.data.vocab_size = actual_vocab_size
            cfg.data.pad_idx = pad_idx
        print("  模型和数据配置已用实际词表信息更新。")
    except Exception as e_cfg:
        print(f"  警告：更新配置时出错：{e_cfg}")


    # --- 加载模型 ---
    is_strict = cfg.get("strict_model", False) # 从顶层读取 strict_model 标志
    print(f"预训练模型类型: {'严格版本' if is_strict else '标准版本'}")

    lit = None
    if is_strict:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model' not in checkpoint:
            raise ValueError(f"错误：严格版本检查点 '{ckpt_path}' 格式无效，缺少 'model' 键。")
        with open_dict(cfg): # 确保加载 Lit 时 cfg.strict_model=True
            cfg.strict_model = True
        lit = LitDeepSC(cfg)
        lit.model.load_state_dict(checkpoint['model'], strict=False)
        print(f"  [√] 成功加载严格版本模型 state_dict。")
        if 'mine' in checkpoint:
             try:
                 lit.mine.load_state_dict(checkpoint['mine'], strict=False)
                 print(f"  [√] 成功加载严格版本 MINE state_dict。")
             except Exception as e_mine:
                 print(f"  [!] 警告: 加载严格版本 MINE state_dict 失败: {e_mine}。")
        else:
             print(f"  [!] 警告: 严格版本检查点中未找到 'mine' state_dict。")
    else:
        try:
            lit = LitDeepSC.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location='cpu', strict=False)
            print(f"  [√] 成功使用 load_from_checkpoint 加载标准版模型。")
        except Exception as e:
            print(f"  [!] 使用 load_from_checkpoint 加载失败: {e}")
            print("  尝试手动创建并加载 state_dict...")
            try:
                lit = LitDeepSC(cfg)
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'state_dict' not in checkpoint:
                    raise ValueError("检查点缺少 'state_dict'")
                missing, unexpected = lit.load_state_dict(checkpoint['state_dict'], strict=False)
                if missing: print(f"  [!] 手动加载时发现缺失键: {missing}")
                if unexpected: print(f"  [!] 手动加载时发现意外键: {unexpected}")
                print("  [√] 成功手动加载标准版 state_dict (strict=False)。")
            except Exception as e2:
                print(f"  [X] 手动加载标准版 state_dict 也失败: {e2}")
                raise RuntimeError(f"无法加载标准版模型检查点: {ckpt_path}") from e2

    # 传递梯度裁剪参数
    lit.grad_clip = cfg.model.get('grad_clip', 1.0)
    print(f"  模型加载完成。梯度裁剪值设置为: {lit.grad_clip}")

    total_params = sum(p.numel() for p in lit.parameters())
    print(f"原模型总参数数量: {total_params:,}")
    print(f"原模型可训练参数数量: {count_trainable_params(lit):,}")

    # ---------- 3. 冻结/解冻策略 ----------
    print(f"\n--- 配置迁移学习模式 ---")
    finetune_mode = cfg.finetune.mode # 从 finetune 组获取
    new_channel_name = cfg.finetune.get('new_channel')

    # ===========================================================
    #  关键修改：动态确定投影层的名称
    # ===========================================================
    # 检查模型实例是 DeepSC 还是 DeepSCStrict 来确定投影层名称
    if hasattr(lit.model, 'proj'):
        proj_layer_name = 'proj'
        proj_layer = lit.model.proj
    elif hasattr(lit.model, 'output_proj'):
        proj_layer_name = 'output_proj'
        proj_layer = lit.model.output_proj
    else:
        raise AttributeError(f"错误：无法在加载的模型 ({type(lit.model).__name__}) 中找到 'proj' 或 'output_proj' 属性。")
    print(f"  检测到模型的投影层名称为: '{proj_layer_name}'")
    # ===========================================================

    if finetune_mode == 'channel':
        if not new_channel_name:
            raise ValueError("错误：在 'channel' 模式下，必须通过命令行或配置文件提供 'finetune.new_channel' 参数。")
        print(f"迁移学习模式: 信道迁移 (Channel Mode) → {new_channel_name}")

        freeze(lit.model.encoder)
        freeze(lit.model.decoder)
        # --- 使用动态获取的投影层 ---
        freeze(proj_layer)
        print(f"  [Frozen] Semantic Encoder, Decoder, {proj_layer_name.capitalize()} Layer")

        unfreeze(lit.model.channel_encoder)
        unfreeze(lit.model.channel_decoder)
        print("  [Unfrozen] Channel Encoder, Channel Decoder")

        try:
            lit.channel = get_channel(new_channel_name)()
            print(f"  [Updated] 信道模型已切换为 {new_channel_name}")
        except ValueError as e:
            raise ValueError(f"错误：无效的信道类型 '{new_channel_name}'。请确保已在 deepsc/models/channel.py 中注册。") from e

        if hasattr(lit, 'mine') and lit.mine is not None:
            freeze(lit.mine)
            print("  [Frozen] MINE Network")

    elif finetune_mode == 'domain':
        print(f"迁移学习模式: 领域迁移 (Domain Mode)")
        print(f"  使用新数据源: train='{train_pkl}', val='{val_pkl}', vocab='{vocab_json}'")

        freeze(lit.model.channel_encoder)
        freeze(lit.model.channel_decoder)
        print("  [Frozen] Channel Encoder, Channel Decoder")
        if hasattr(lit, 'mine') and lit.mine is not None:
            freeze(lit.mine)
            print("  [Frozen] MINE Network")

        unfreeze(lit.model.encoder)
        unfreeze(lit.model.decoder)
        # --- 使用动态获取的投影层 ---
        unfreeze(proj_layer)
        print(f"  [Unfrozen] Semantic Encoder, Decoder, {proj_layer_name.capitalize()} Layer")

        # 检查词表大小是否变化，如果变化则重新初始化相关层
        old_vocab_size = lit.model.encoder.embed.num_embeddings
        if actual_vocab_size != old_vocab_size:
            print(f"  检测到词表大小变化: {old_vocab_size} -> {actual_vocab_size}")
            print("  重新初始化 Embedding 和 Projection 层...")
            embed_dim = lit.model.encoder.embed.embedding_dim
            # --- 使用动态获取的投影层信息 ---
            proj_in_features = proj_layer.in_features

            new_encoder_embed = torch.nn.Embedding(actual_vocab_size, embed_dim, padding_idx=pad_idx)
            new_decoder_embed = torch.nn.Embedding(actual_vocab_size, embed_dim, padding_idx=pad_idx)
            lit._init_weights_xavier(new_encoder_embed)
            lit._init_weights_xavier(new_decoder_embed)
            lit.model.encoder.embed = new_encoder_embed
            lit.model.decoder.embed = new_decoder_embed

            new_proj = torch.nn.Linear(proj_in_features, actual_vocab_size)
            lit._init_weights_xavier(new_proj)
            # --- 使用 setattr 动态设置属性 ---
            setattr(lit.model, proj_layer_name, new_proj) # 使用 setattr 赋值

            print(f"  [Updated] Embedding 和 {proj_layer_name} 层已更新为新词表大小 {actual_vocab_size}。")
        else:
            print("  词表大小未变化，无需重置 Embedding/Projection 层。")

    else:
         raise ValueError(f"错误：无效的 finetune.mode 配置值 '{finetune_mode}'。必须是 'channel' 或 'domain'。")

    # 重新计算并打印可训练参数情况
    trainable_params = count_trainable_params(lit)
    total_params = sum(p.numel() for p in lit.parameters())
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    print(f"\n当前模型总参数: {total_params:,}")
    print(f"当前可训练参数数量: {trainable_params:,}")
    print(f"可训练参数占比: {trainable_ratio:.2%}")

    # ---------- 4. 加载数据 & 训练 ----------
    print(f"\n--- 加载数据集进行微调 ---")
    train_batch_size = cfg.train.get('batch_size', 128)
    num_workers = cfg.get("num_workers", 4)
    print(f"  使用训练批大小: {train_batch_size}")
    print(f"  使用数据加载 Workers: {num_workers}")

    train_loader = make_dataloader(
        str(train_pkl),
        batch_size = train_batch_size,
        pad_idx    = pad_idx,
        num_workers= num_workers,
        shuffle    = True,
    )
    val_loader = make_dataloader(
        str(val_pkl),
        batch_size = train_batch_size,
        pad_idx    = pad_idx,
        shuffle    = False,
        num_workers= num_workers,
    )
    try:
        print(f"训练集大小: {len(train_loader.dataset)} 句")
        print(f"验证集大小: {len(val_loader.dataset)} 句")
    except TypeError:
         print("训练集/验证集大小: 未知 (IterableDataset)")

    # ---------- 5. 配置训练器和回调 ----------
    print(f"\n--- 配置 Pytorch Lightning Trainer ---")
    output_dir_name = f"finetune_channel_{new_channel_name}" if finetune_mode == 'channel' else 'finetune_domain'
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"训练日志和检查点将保存在 Hydra 输出目录: {output_dir.resolve()}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_bleu',
        mode='max',
        dirpath=output_dir / 'checkpoints',
        filename='ft-{epoch:02d}-{val_bleu:.3f}',
        save_top_k=cfg.finetune.save_top_k,
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_bleu',
        patience=cfg.finetune.patience,
        min_delta=cfg.finetune.min_delta,
        mode='max',
        verbose=True
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(output_dir),
        name="",
        version="",
        default_hp_metric=False
    )
    trainer = pl.Trainer(
        max_epochs       = cfg.finetune.epochs,
        precision        = cfg.precision,
        accelerator      = 'auto',
        devices          = 'auto',
        default_root_dir = str(output_dir),
        logger           = logger,
        callbacks        = [checkpoint_callback, lr_monitor, early_stop_callback],
        log_every_n_steps= cfg.train.get('log_every_n_steps', 50),
        val_check_interval= cfg.train.get('val_check_interval', 0.5),
    )

    # ---------- 6. 开始微调训练 ----------
    print(f"\n--- 开始迁移学习 (Epochs={cfg.finetune.epochs}) ---")
    try:
        trainer.fit(lit, train_loader, val_loader)
    except Exception as train_err:
        print(f"\n错误：训练过程中发生错误：{train_err}")
        traceback.print_exc()
        print("请检查模型、数据或训练配置。")
        sys.exit(1)

    # ---------- 7. 结束 ----------
    print("\n迁移学习完成!")
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path and Path(checkpoint_callback.best_model_path).exists():
        print(f"最佳模型检查点保存在: {checkpoint_callback.best_model_path}")
        if hasattr(checkpoint_callback, 'best_model_score') and checkpoint_callback.best_model_score:
            print(f"对应的最佳验证 BLEU 分数: {checkpoint_callback.best_model_score:.4f}")
        else:
            print("未能获取最佳模型的 BLEU 分数。")
    else:
        print("警告：未能找到最佳模型检查点路径或文件不存在。")
        if hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path:
            print(f"最后一个检查点路径: {checkpoint_callback.last_model_path}")
    print(f"所有日志和检查点位于: {output_dir.resolve()}")

if __name__ == '__main__':
    main()