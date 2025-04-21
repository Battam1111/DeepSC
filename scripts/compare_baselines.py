# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC-Fork/scripts/compare_baselines.py
# (已修改: 修复 DeepSCStrict 加载错误, 增强检查点兼容性, 完善错误处理和日志)
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
比较脚本：评估DeepSC与传统方法的性能
==================================================
同时评估不同SNR下的BLEU分数、句子相似度和互信息

特性：
1. 自动适应不同词表大小
2. 支持多种信道类型
3. 全面的传统基线比较 (Huffman/Fixed-Length + Turbo/RS)
4. 结果可视化和保存 (CSV, PNG 图表)
5. 修正了 RS 码的评估流程 (使用硬判决字节)
6. 更健壮的错误处理和结果统计
7. 增强的检查点兼容性处理
8. 改进的配置访问逻辑
9. 修复了严格模式模型初始化问题
"""
import sys
import torch
import transformers
import math, tqdm, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors # 导入 OmegaConf
from hydra.utils import to_absolute_path
import json
import os
from typing import Tuple, List, Dict, Any, Optional, Union
from contextlib import nullcontext # 用于创建无操作的上下文管理器
import traceback # 用于打印更详细的错误信息
import warnings
import inspect # 用于检查函数签名

# --- 数据处理与模型相关 ---
from deepsc.data.europarl import make_dataloader
from deepsc.data.vocab import Vocab
from deepsc.engine.lit_module import LitDeepSC
from deepsc.models.transformer import DeepSC # 标准版模型
from deepsc.models.deepsc_strict import DeepSCStrict # 导入严格模型
from deepsc.models.mine import MINE # 标准版 MINE
from deepsc.models.mine_strict import MINEStrict # 严格版 MINE
from deepsc.models import get_channel
from deepsc.metrics.bleu import bleu_score
from deepsc.metrics.sentence_sim import sentence_similarity

# --- 导入基线方法 ---
try:
    from deepsc.baselines.traditional import HuffmanEncoder, FixedLengthEncoder
    from deepsc.baselines.channel_coding import TurboCoder, RSCoder
    _GLOBAL_BASELINES_AVAILABLE = True
    print("信息: 传统基线方法库加载成功。")
except ImportError as e:
    print(f"警告: 传统基线方法库 (deepsc.baselines) 未找到或导入失败: {e}")
    print("      将仅评估 DeepSC 模型，跳过与传统方法的比较。")
    _GLOBAL_BASELINES_AVAILABLE = False
except Exception as e_other:
    print(f"警告: 加载传统基线方法库时发生意外错误: {e_other}")
    _GLOBAL_BASELINES_AVAILABLE = False


# --- 工具函数：检查点加载和分析 ---
def inspect_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """
    分析检查点文件，确定其格式和内容结构

    参数:
        ckpt_path: 检查点文件路径

    返回:
        包含检查点分析结果的字典
    """
    # 屏蔽 PyTorch 的 FutureWarning 消息
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            return {
                'error': f"加载检查点失败: {str(e)}",
                'raw_checkpoint': None, # 修改: 保持一致性，使用 raw_checkpoint
                'valid': False
            }

    # 分析检查点内容和结构
    if not isinstance(checkpoint, dict):
        return {
            'error': f"检查点不是字典类型，而是 {type(checkpoint)}",
            'raw_checkpoint': checkpoint,
            'valid': False
        }

    # 识别检查点类型
    keys = list(checkpoint.keys())

    # 判断是否是 PyTorch Lightning 检查点
    is_pl_checkpoint = 'state_dict' in keys and 'hyper_parameters' in keys
    # 判断是否是严格版本检查点
    is_strict_checkpoint = 'model' in keys # TwoPhaseTrainer 保存的检查点包含 'model' 键
    # 判断配置是否存在以及其类型
    has_config = 'config' in keys
    config_type = type(checkpoint.get('config', None)).__name__ if 'config' in keys else None
    has_hparams = 'hyper_parameters' in keys

    # 获取词表大小信息
    vocab_size = None
    if is_pl_checkpoint and isinstance(checkpoint.get('state_dict'), dict):
        sd = checkpoint['state_dict']
        # 尝试从不同可能的嵌入层名称获取 vocab_size
        embed_keys = ['model.embedding.weight', 'model.encoder.embed.weight', 'embedding.weight', 'encoder.embed.weight']
        for key in embed_keys:
            if key in sd:
                vocab_size = sd[key].shape[0]
                break
    elif is_strict_checkpoint and isinstance(checkpoint.get('model'), dict):
        sd = checkpoint['model']
        # 严格版本中嵌入层路径固定
        if 'encoder.embedding.weight' in sd: # 修正: DeepSCStrict 中是 encoder.embedding
             vocab_size = sd['encoder.embedding.weight'].shape[0]
        elif 'embedding.weight' in sd: # 兼容旧版或其他可能
             vocab_size = sd['embedding.weight'].shape[0]


    # 尝试从 hyper_parameters 获取 vocab_size (作为备选)
    if vocab_size is None and is_pl_checkpoint and has_hparams and isinstance(checkpoint.get('hyper_parameters'), dict):
         hparams = checkpoint['hyper_parameters']
         # 尝试从 hparams.cfg.model 或 hparams.model_params
         if 'cfg' in hparams and hasattr(hparams['cfg'], 'model') and hasattr(hparams['cfg'].model, 'vocab_size'):
              vocab_size = hparams['cfg'].model.vocab_size
         elif 'model_params' in hparams and 'vocab_size' in hparams['model_params']:
              vocab_size = hparams['model_params']['vocab_size']

    # 尝试从 config 字典获取 vocab_size (用于严格版本检查点)
    if vocab_size is None and has_config and isinstance(checkpoint.get('config'), dict):
         config_dict = checkpoint['config']
         if 'model' in config_dict and 'vocab_size' in config_dict['model']:
              vocab_size = config_dict['model']['vocab_size']

    return {
        'valid': True,
        'raw_checkpoint': checkpoint, # 返回原始检查点数据
        'keys': keys,
        'is_pl_checkpoint': is_pl_checkpoint,
        'is_strict_checkpoint': is_strict_checkpoint,
        'has_config': has_config,
        'config_type': config_type,
        'has_hparams': has_hparams,
        'vocab_size': vocab_size, # 从检查点推断的词表大小
    }

def get_model_config(checkpoint_info: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    从检查点信息和传入的Hydra配置中合并获取最终的模型配置字典。
    优先使用检查点中的配置，若无则使用传入的配置。

    参数:
        checkpoint_info: inspect_checkpoint 返回的检查点分析结果。
        cfg: 外部传入的 Hydra 配置对象 (作为备选)。

    返回:
        一个标准的 Python 字典，包含用于初始化模型的配置。
    """
    model_config_dict = {}

    if not checkpoint_info['valid']:
        print("  警告: 检查点信息无效，将完全使用外部传入的配置。")
        # 确保返回的是标准字典
        return OmegaConf.to_container(cfg.model, resolve=True) if hasattr(cfg, 'model') else {}

    checkpoint = checkpoint_info['raw_checkpoint']
    config_source = "未知来源" # 用于日志记录

    # 优先级 1: 检查点内的 'config' 字段 (通常来自 TwoPhaseTrainer)
    if checkpoint_info['has_config']:
        config_obj = checkpoint['config']
        if isinstance(config_obj, dict) and 'model' in config_obj:
            model_config_dict = config_obj['model']
            config_source = "检查点内 'config' 字典"
        elif hasattr(config_obj, 'model'): # 兼容 OmegaConf 对象
             try:
                  # 尝试解析 OmegaConf 对象
                  model_config_dict = OmegaConf.to_container(config_obj.model, resolve=True)
                  config_source = f"检查点内 'config' 对象 ({checkpoint_info['config_type']})"
             except Exception as e_resolve:
                  print(f"  警告：解析检查点内 'config' 对象时出错: {e_resolve}。将尝试其他来源。")
                  model_config_dict = {} # 解析失败，重置

    # 优先级 2: 检查点内的 'hyper_parameters' (来自 PyTorch Lightning)
    if not model_config_dict and checkpoint_info['has_hparams']:
        hparams = checkpoint.get('hyper_parameters', {})
        if 'cfg' in hparams and hasattr(hparams['cfg'], 'model'):
            try:
                model_config_dict = OmegaConf.to_container(hparams['cfg'].model, resolve=True)
                config_source = "检查点内 'hyper_parameters.cfg.model'"
            except Exception as e_resolve_hparam:
                 print(f"  警告：解析 'hyper_parameters.cfg.model' 时出错: {e_resolve_hparam}。")
                 model_config_dict = {}
        elif 'model_params' in hparams and isinstance(hparams['model_params'], dict):
             model_config_dict = hparams['model_params']
             config_source = "检查点内 'hyper_parameters.model_params'"

    # 优先级 3: 使用外部传入的配置
    if not model_config_dict:
        print("  信息: 未在检查点中找到有效的模型配置，将使用外部传入的配置。")
        if hasattr(cfg, 'model'):
             model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
             config_source = "外部传入的 Hydra 配置 (cfg.model)"
        else:
             print("  警告：外部传入的配置中也缺少 'model' 部分。")
             model_config_dict = {} # 最终的备选为空字典

    print(f"  使用的模型配置来源: {config_source}")

    # 最后确保返回的是标准 Python 字典
    if not isinstance(model_config_dict, dict):
         # 如果经过上述步骤仍然不是字典（可能性很小），尝试最后转换
         try:
              model_config_dict = OmegaConf.to_container(model_config_dict, resolve=True)
         except Exception:
              print(f"  错误：无法将最终的模型配置转换为字典 (类型: {type(model_config_dict)})。返回空配置。")
              return {}

    return model_config_dict


def filter_kwargs_for_init(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤关键字参数字典，只保留类的__init__方法接受的参数。
    (注意：此函数不适用于__init__只接受单个config字典的类，如DeepSCStrict)

    参数:
        cls: 要检查的类
        kwargs: 原始关键字参数字典

    返回:
        过滤后的关键字参数字典
    """
    # 检查类的__init__方法签名
    try:
        init_signature = inspect.signature(cls.__init__)
        valid_params = list(init_signature.parameters.keys())
        # 移除self参数
        if 'self' in valid_params:
            valid_params.remove('self')

        # 检查是否存在 **kwargs
        has_kwargs_param = any(param.kind == inspect.Parameter.VAR_KEYWORD
                             for param in init_signature.parameters.values())

        if has_kwargs_param:
             # 如果 __init__ 接受 **kwargs，则不过滤，直接返回所有参数
             # print(f"  信息: {cls.__name__}.__init__ 接受 **kwargs，不过滤参数。")
             return kwargs

        # 过滤出有效参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        # 打印被过滤掉的参数（用于调试）
        # ignored_keys = set(kwargs.keys()) - set(filtered_kwargs.keys())
        # if ignored_keys:
        #      print(f"  调试: 为 {cls.__name__} 过滤掉的参数: {ignored_keys}")

        return filtered_kwargs
    except (ValueError, TypeError) as e:
        # 如果签名检查失败，保守地返回原始kwargs
        print(f"  警告: 无法检查 {cls.__name__} 的初始化参数: {e}。返回原始参数。")
        return kwargs

def load_model_from_checkpoint(checkpoint_info: Dict[str, Any], cfg: DictConfig, is_strict_version: bool,
                               vocab_size: int) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    根据检查点信息和模型类型加载模型、MINE网络和信道模型。
    自动处理严格版本和标准版本检查点的加载。

    参数:
        checkpoint_info: inspect_checkpoint 返回的检查点分析结果。
        cfg: Hydra 配置对象。
        is_strict_version: 是否期望加载严格版本模型。
        vocab_size: 实际的词表大小（从外部加载的 vocab.json 确定）。

    返回:
        (model, mine, channel) 元组。如果加载失败，对应项可能为 None。
        注意：channel 模型总是根据当前 cfg.data.channel 重新创建。
    """
    if not checkpoint_info['valid']:
        raise ValueError(f"无效的检查点信息: {checkpoint_info.get('error', '未知错误')}")

    checkpoint = checkpoint_info['raw_checkpoint']
    model: Optional[torch.nn.Module] = None
    mine: Optional[torch.nn.Module] = None
    channel: Optional[torch.nn.Module] = None

    # 1. 获取模型配置 (优先从检查点获取，然后合并外部配置)
    try:
        # 尝试从检查点和传入的cfg合并获取最完整的模型配置
        model_config_base = get_model_config(checkpoint_info, cfg)

        # 确保关键参数存在并正确 (使用外部传入的值覆盖或补充)
        model_config_base['vocab_size'] = vocab_size # 使用外部加载的词表大小
        model_config_base['pad_idx'] = cfg.data.pad_idx # 使用外部加载的 pad_idx

        # 从外部cfg补充可能缺失的参数 (例如旧检查点可能没有 dropout 等)
        for key, value in OmegaConf.to_container(cfg.model, resolve=True).items():
             if key not in model_config_base:
                  print(f"  信息: 从外部配置补充参数 model.{key} = {value}")
                  model_config_base[key] = value

        print(f"  最终用于模型初始化的配置键: {list(model_config_base.keys())}")
    except Exception as e:
        print(f"  错误: 准备模型配置时出错: {e}，将使用默认配置。")
        traceback.print_exc()
        model_config_base = {
            'vocab_size': vocab_size,
            'pad_idx': cfg.data.pad_idx,
            'latent_dim': cfg.model.get('latent_dim', 16),
            'd_model': cfg.model.get('d_model', 512),
            'n_layers': cfg.model.get('n_layers', 3),
            'n_heads': cfg.model.get('n_heads', 8),
            'd_ff': cfg.model.get('d_ff', 2048),
            'dropout': cfg.model.get('dropout', 0.1),
            'max_len': cfg.model.get('max_len', 100)
        }

    # 2. 加载模型
    if is_strict_version:
        # --- 加载严格版本模型 ---
        print("  使用严格版本模型加载策略 (尝试加载 DeepSCStrict)")
        if checkpoint_info['is_strict_checkpoint'] and 'model' in checkpoint:
            print("    检测到严格版本格式检查点 (包含 'model' 键)")
            try:
                # ****************************************************************
                # ********** 核心修复：直接传递配置字典 **********
                # ****************************************************************
                print(f"    使用配置初始化 DeepSCStrict: {list(model_config_base.keys())}")
                deepsc_model = DeepSCStrict(model_config_base) # 直接传递字典
                # ****************************************************************

                # 加载模型权重
                print("    加载模型 state_dict...")
                missing_keys, unexpected_keys = deepsc_model.load_state_dict(checkpoint['model'], strict=False)
                if missing_keys: print(f"      警告: 加载模型时发现缺失键: {missing_keys}")
                if unexpected_keys: print(f"      警告: 加载模型时发现意外键: {unexpected_keys}")
                model = deepsc_model
                print("    [√] 成功加载严格版本模型状态")

                # 尝试加载MINE网络 (严格版本)
                if 'mine' in checkpoint:
                    try:
                        latent_dim = model_config_base.get('latent_dim', 16)
                        print(f"    加载严格版本 MINE (latent_dim={latent_dim})...")
                        mine_strict = MINEStrict(latent_dim=latent_dim)
                        missing_mine, unexpected_mine = mine_strict.load_state_dict(checkpoint['mine'], strict=False)
                        if missing_mine: print(f"      警告: 加载MINE时发现缺失键: {missing_mine}")
                        if unexpected_mine: print(f"      警告: 加载MINE时发现意外键: {unexpected_mine}")
                        mine = mine_strict
                        print("    [√] 成功加载严格版本 MINE 状态")
                    except Exception as e_mine:
                        print(f"    [!] 警告: 加载严格版本 MINE 失败: {e_mine}")
                        mine = None # 加载失败则不使用 MINE
                else:
                     print("    [!] 警告: 严格版本检查点中未找到 'mine' state_dict。")

            except Exception as e:
                print(f"    [X] 加载严格版本模型失败: {e}")
                traceback.print_exc()
                # 如果加载失败，可以选择抛出异常或返回 None
                # raise RuntimeError(f"无法从严格检查点加载 DeepSCStrict 模型: {e}") from e
                print("      将尝试使用外部配置创建新的严格模型（无预训练权重）。")
                try:
                     model = DeepSCStrict(model_config_base)
                     mine = MINEStrict(latent_dim=model_config_base.get('latent_dim', 16))
                except Exception as e_create:
                     print(f"      错误：创建新的严格模型也失败：{e_create}")
                     model, mine = None, None

        else:
             # 检查点不是严格版本格式，但要求加载严格版本（例如从标准版迁移）
             print("  警告: 检查点不是严格版本格式，但要求加载严格版本模型。")
             print("        将尝试使用外部配置创建新的严格模型（无预训练权重）。")
             # （未来可以添加从标准版检查点转换参数的逻辑，但目前只创建新模型）
             try:
                  model = DeepSCStrict(model_config_base)
                  mine = MINEStrict(latent_dim=model_config_base.get('latent_dim', 16))
                  print("    [√] 已创建新的未训练的严格版本模型和MINE。")
             except Exception as e_create:
                  print(f"    [X] 错误：创建新的严格模型失败：{e_create}")
                  model, mine = None, None

    else:
        # --- 加载标准版本模型 ---
        print("  使用标准版本模型加载策略 (尝试加载 LitDeepSC 或 DeepSC)")
        try:
            # 优先级 1: 尝试使用 Lightning 的 load_from_checkpoint
            print("    尝试使用 LitDeepSC.load_from_checkpoint...")
            # 确保加载时 strict_model 标志为 False
            with OmegaConf.open_dict(cfg):
                 cfg.strict_model = False
            lit_model = LitDeepSC.load_from_checkpoint(
                checkpoint_path=str(checkpoint_info['ckpt_path']),
                map_location='cpu',
                cfg=cfg,  # 传递完整的配置
                strict=False # 允许部分参数不匹配
            )
            model = lit_model.model
            mine = lit_model.mine
            print("    [√] 成功使用 load_from_checkpoint 加载标准版模型和 MINE。")

        except Exception as e1:
            print(f"    [!] 使用 load_from_checkpoint 加载失败: {e1}")
            print("    尝试手动创建 LitDeepSC 并加载 state_dict...")

            try:
                # 优先级 2: 手动创建 LitDeepSC 并加载 state_dict
                with OmegaConf.open_dict(cfg):
                    cfg.strict_model = False
                lit_model = LitDeepSC(cfg)
                if 'state_dict' in checkpoint:
                    missing_keys, unexpected_keys = lit_model.load_state_dict(checkpoint['state_dict'], strict=False)
                    if missing_keys: print(f"      警告: 加载 state_dict 时发现缺失键: {missing_keys}")
                    if unexpected_keys: print(f"      警告: 加载 state_dict 时发现意外键: {unexpected_keys}")
                    model = lit_model.model
                    mine = lit_model.mine
                    print("    [√] 成功手动加载标准版 state_dict 到 LitDeepSC。")
                else:
                    raise ValueError("检查点中缺少 'state_dict' 键。")

            except Exception as e2:
                print(f"    [!] 手动加载 state_dict 到 LitDeepSC 失败: {e2}")
                print("    尝试直接创建 DeepSC 模型并加载匹配的参数...")

                try:
                     # 优先级 3: 直接创建 DeepSC 模型并尝试加载部分参数
                     model = DeepSC(model_config_base) # 使用原始 DeepSC 类
                     if 'state_dict' in checkpoint:
                          # 从 PL state_dict 中提取模型参数
                          model_state_dict = {}
                          for k, v in checkpoint['state_dict'].items():
                               if k.startswith('model.'):
                                    model_state_dict[k[len('model.'):]] = v # 去掉 'model.' 前缀
                               elif not k.startswith('mine.'): # 保留非 MINE 也非 model 的参数（可能性小）
                                    model_state_dict[k] = v

                          if model_state_dict:
                               missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                               if missing_keys: print(f"      警告: 加载模型参数时发现缺失键: {missing_keys}")
                               if unexpected_keys: print(f"      警告: 加载模型参数时发现意外键: {unexpected_keys}")
                               print("    [√] 成功直接创建 DeepSC 模型并加载了部分参数。")
                          else:
                               print("      警告: 未能在 state_dict 中找到 'model.' 前缀的参数。模型权重未加载。")

                          # 尝试加载 MINE (标准版)
                          mine_state_dict = {}
                          has_mine_params = False
                          for k, v in checkpoint['state_dict'].items():
                               if k.startswith('mine.'):
                                    mine_state_dict[k[len('mine.'):]] = v # 去掉 'mine.' 前缀
                                    has_mine_params = True
                          if has_mine_params:
                               try:
                                    latent_dim = model_config_base.get('latent_dim', 16)
                                    mine = MINE(latent_dim, hidden=256) # 使用标准版 MINE
                                    mine.load_state_dict(mine_state_dict, strict=False)
                                    print("    [√] 成功创建并加载了标准版 MINE 网络参数。")
                               except Exception as e_mine_load:
                                    print(f"    [!] 警告: 加载标准版 MINE 网络参数失败: {e_mine_load}")
                                    mine = None
                          else:
                               print("      [!] 警告: 未能在 state_dict 中找到 'mine.' 前缀的参数。MINE 网络未加载。")
                               mine = None
                     else:
                           print("      [!] 警告: 检查点中缺少 'state_dict'，模型和 MINE 权重未加载。")
                           mine = None # 确保 mine 在这种情况下也是 None

                except Exception as e3:
                    print(f"    [X] 所有加载标准版模型的尝试均失败: {e3}")
                    traceback.print_exc()
                    # 如果所有尝试都失败，返回 None
                    model, mine = None, None


    # 3. 创建信道模型 (总是根据当前配置创建)
    try:
        channel = get_channel(cfg.data.channel)()
        print(f"  [√] 成功创建信道模型: {cfg.data.channel}")
    except Exception as e_chan:
        print(f"  [X] 错误: 无法创建信道模型 '{cfg.data.channel}': {e_chan}")
        channel = None # 创建失败则返回 None

    return model, mine, channel

# --- 辅助函数：模拟传统信道 ---
# (保持不变)
def simulate_traditional_channel_llr(bits: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    """
    模拟传统通信系统中的 BPSK 调制、AWGN 信道和解调（软判决 LLR）。
    适用于需要 LLR 输入的解码器（如 Turbo）。

    参数:
        bits (np.ndarray): 输入比特序列，形状 [batch_size, bit_length]，元素为 0 或 1。
        snr_db (float): 信噪比 (dB)。

    返回:
        Tuple[np.ndarray, float]:
        - LLR (np.ndarray): 解调后的对数似然比 LLR，形状与输入比特相同。
        - noise_var (float): 计算得到的噪声方差 σ^2。
    """
    if bits.size == 0: # 处理空输入
        return np.array([]), 0.0

    # 1. 比特映射到 BPSK 符号 (+1 / -1)
    symbols = 2 * bits.astype(np.float32) - 1

    # 2. 计算噪声方差 σ^2 = 1 / (2 * SNR_lin)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * max(snr_lin, 1e-10)) # 噪声方差 σ^2

    # 3. 生成加性高斯白噪声 N ~ N(0, σ^2)
    noise = np.random.normal(0, np.sqrt(noise_var), symbols.shape) # 标准差是 sqrt(方差)

    # 4. 接收信号 y = x + n
    received = symbols + noise

    # 5. 计算软判决 LLR = 2 * y / σ^2
    # 增加数值稳定性检查，避免除以非常小的 noise_var
    llr = 2.0 * received / max(noise_var, 1e-20)

    return llr, noise_var

# --- 辅助函数：确保是 NumPy 数组 ---
# (保持不变)
def ensure_numpy_array(array):
    """
    确保输入数组是标准的 numpy.ndarray 类型，而不是 galois.FieldArray。

    参数:
        array: 输入数组，可能是 galois.FieldArray 或标准 numpy.ndarray

    返回:
        numpy.ndarray: 转换后的标准 NumPy 数组
    """
    # 检查是否是 FieldArray 类型
    if hasattr(array, 'view') and hasattr(array, '__class__') and 'FieldArray' in array.__class__.__name__:
        # 转换为标准 numpy 数组
        return array.view(np.ndarray)
    return array

# --- 辅助函数：比特到字节的打包 ---
# (保持不变)
def pack_bits_to_bytes(bits_array: np.ndarray) -> np.ndarray:
    """
    将 NumPy 比特数组 (0/1) 打包成字节数组 (uint8)。
    在末尾填充 0 比特以确保总比特数是 8 的倍数。
    支持 galois.FieldArray 类型的输入。

    参数:
        bits_array (np.ndarray): 输入比特数组，形状 [batch_size, bit_length]。

    返回:
        np.ndarray: 打包后的字节数组，形状 [batch_size, byte_length]，dtype=uint8。
    """
    # 确保输入是标准 numpy 数组
    bits_array = ensure_numpy_array(bits_array)

    if bits_array.ndim != 2:
        raise ValueError(f"输入比特数组必须是二维 [batch_size, bit_length]，但得到 {bits_array.ndim}维")

    batch_size, bit_length = bits_array.shape
    if bit_length == 0:
        return np.zeros((batch_size, 0), dtype=np.uint8)

    # 修正: num_padding_bits 的计算应该针对每个样本，因为长度可能不同
    # 但 packbits 要求输入能被8整除，所以需要在循环内处理
    byte_batches = []
    max_byte_len = 0

    for i in range(batch_size):
        current_bits = bits_array[i]
        current_bit_length = len(current_bits)
        if current_bit_length == 0:
            byte_batches.append(np.array([], dtype=np.uint8))
            continue

        # 确保当前批次也是标准的 numpy 数组
        current_bits = ensure_numpy_array(current_bits)

        # 计算当前样本需要的填充位数
        num_padding_bits = (8 - (current_bit_length % 8)) % 8
        padded_bits = np.pad(current_bits, (0, num_padding_bits), constant_values=0).astype(np.uint8)

        try:
            # np.packbits 对每 8 个比特进行打包
            packed_bytes = np.packbits(padded_bits)
            byte_batches.append(packed_bytes)
            max_byte_len = max(max_byte_len, len(packed_bytes))
        except ValueError as e:
            print(f"错误: 打包比特到字节时出错 (Batch={i}, len={len(padded_bits)}): {e}")
            traceback.print_exc()  # 添加堆栈跟踪以便更好地调试
            byte_batches.append(np.array([], dtype=np.uint8)) # 添加空数组作为错误处理

    # 填充到批次内最大字节长度
    padded_bytes_batches = []
    for b in byte_batches:
        pad_width = max(0, max_byte_len - len(b))
        padded = np.pad(b, (0, pad_width), mode='constant', constant_values=0)
        padded_bytes_batches.append(padded)

    return np.array(padded_bytes_batches, dtype=np.uint8)


# --- 辅助函数：字节到比特的解包 ---
# (保持不变)
def unpack_bytes_to_bits(bytes_array: np.ndarray, original_bit_lengths: List[int]) -> np.ndarray:
    """
    将 NumPy 字节数组 (uint8) 解包成比特数组 (0/1)，并根据原始比特长度截断。
    支持 galois.FieldArray 类型的输入。

    参数:
        bytes_array (np.ndarray): 输入字节数组，形状 [batch_size, byte_length]。
        original_bit_lengths (List[int]): 每个样本原始的比特数列表。

    返回:
        np.ndarray: 解包后的比特数组，形状 [batch_size, max_original_bit_length]。
    """
    # 确保输入是标准 numpy 数组
    bytes_array = ensure_numpy_array(bytes_array)

    if bytes_array.ndim != 2:
        raise ValueError(f"输入字节数组必须是二维 [batch_size, byte_length]，但得到 {bytes_array.ndim}维")
    batch_size = bytes_array.shape[0]
    if len(original_bit_lengths) != batch_size:
        raise ValueError(f"原始比特长度列表大小 ({len(original_bit_lengths)}) 与批大小 ({batch_size}) 不匹配")

    unpacked_bits_batches = []
    max_len_out = 0

    for i in range(batch_size):
        current_bytes = bytes_array[i]
        original_len = original_bit_lengths[i]

        if len(current_bytes) == 0 and original_len > 0:
             # 如果字节数组为空，但预期长度非零，则输出全零比特
             unpacked_bits = np.zeros(original_len, dtype=int)
        elif len(current_bytes) == 0 and original_len == 0:
             unpacked_bits = np.array([], dtype=int)
        elif len(current_bytes) > 0:
             try:
                  # 确保当前批次也是标准的 numpy 数组
                  current_bytes = ensure_numpy_array(current_bytes)

                  # np.unpackbits 将每个字节解包为 8 个比特
                  unpacked_bits_full = np.unpackbits(current_bytes)
                  # 根据原始长度截断
                  unpacked_bits = unpacked_bits_full[:original_len]
             except Exception as e:
                  print(f"错误: 解包字节到比特时出错 (Batch={i}): {e}")
                  traceback.print_exc()  # 添加堆栈跟踪以便更好地调试
                  # 出错时返回全零比特
                  unpacked_bits = np.zeros(original_len, dtype=int) if original_len > 0 else np.array([], dtype=int)
        else: # len(current_bytes) > 0 and original_len == 0 (不常见，但处理一下)
             unpacked_bits = np.array([], dtype=int)


        unpacked_bits_batches.append(unpacked_bits.astype(int)) # 确保是 int
        max_len_out = max(max_len_out, len(unpacked_bits))

    # 填充到批次内最大原始比特长度
    padded_bits_batches = []
    for bits in unpacked_bits_batches:
        pad_width = max(0, max_len_out - len(bits))
        padded = np.pad(bits, (0, pad_width), mode='constant', constant_values=0)
        padded_bits_batches.append(padded)

    return np.array(padded_bits_batches, dtype=int)


# --------------------------- 主程序 (Hydra 配置) --------------------------- #
@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    主函数：加载模型，运行评估，比较DeepSC与传统方法在不同SNR下的性能。

    参数:
        cfg (DictConfig): Hydra 加载的配置对象。
    """
    # ---------- 0. 初始化局部基线可用性标志 ----------
    local_baselines_available = _GLOBAL_BASELINES_AVAILABLE

    # ---------- 1. 路径解析与检查 ----------
    print("--- 1. 解析和检查路径 ---")
    ckpt_path = Path(to_absolute_path(cfg.ckpt_path))
    val_pkl = Path(to_absolute_path(cfg.data.val_pkl))
    vocab_json = Path(to_absolute_path(cfg.data.vocab_json))
    print(f"  检查点路径: {ckpt_path}")
    print(f"  验证数据: {val_pkl}")
    print(f"  词表文件: {vocab_json}")
    for p in (ckpt_path, val_pkl, vocab_json):
        if not p.exists():
            raise FileNotFoundError(f"错误: 未找到文件 '{p}'。请检查配置。")
    print("  所有必需文件均存在。")

    # ---------- 2. 加载词表并更新配置 ----------
    print("\n--- 2. 加载词表和更新配置 ---")
    print(f"  加载词表: {vocab_json}")
    vocab = Vocab.load(vocab_json)
    actual_vocab_size = len(vocab)
    pad_idx = vocab.token2idx.get('<PAD>', 0)
    start_idx = vocab.token2idx.get('<START>', 1) # 获取 START 索引
    end_idx = vocab.token2idx.get('<END>', 2) # 获取 END 索引
    print(f"  实际词表大小: {actual_vocab_size}")
    print(f"  PAD Token 索引: {pad_idx}")
    print(f"  START Token 索引: {start_idx}")
    print(f"  END Token 索引: {end_idx}")

    # --- 动态更新配置 (使用 open_dict 确保可以修改) ---
    try:
        with OmegaConf.open_dict(cfg):
             # 更新 model 配置
             if 'model' not in cfg: cfg.model = OmegaConf.create()
             cfg.model.vocab_size = actual_vocab_size
             cfg.model.pad_idx = pad_idx

             # 更新 data 配置
             if 'data' not in cfg: cfg.data = OmegaConf.create()
             cfg.data.vocab_size = actual_vocab_size
             cfg.data.pad_idx = pad_idx
        print("  模型和数据配置已用实际词表信息更新。")
    except Exception as e_cfg:
        print(f"  警告：更新配置时出错：{e_cfg}")
        # 即使更新失败，也尝试继续，因为加载逻辑会再次检查

    # ---------- 3. 设置设备 & 加载 DeepSC 模型 ----------
    print("\n--- 3. 设置设备和加载 DeepSC 模型 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    print(f"  加载模型检查点: {ckpt_path}")

    # 检查是否是严格版本模型 (从配置读取，更可靠)
    is_strict_version = cfg.get("strict_model", False)
    print(f"  模型类型: {'严格版本 (DeepSCStrict)' if is_strict_version else '标准版本 (LitDeepSC/DeepSC)'}")

    # ----- 检查点分析和自适应加载 -----
    print("  分析检查点格式...")
    checkpoint_info = inspect_checkpoint(ckpt_path)
    checkpoint_info['ckpt_path'] = ckpt_path # 将路径也存入信息，方便加载函数使用

    if 'error' in checkpoint_info:
        raise ValueError(f"无效的检查点文件: {checkpoint_info['error']}")

    print(f"  检查点键: {checkpoint_info['keys']}")
    print(f"  检查点类型: {'PyTorch Lightning' if checkpoint_info['is_pl_checkpoint'] else '标准字典'}")
    print(f"  严格版本格式推断: {'是' if checkpoint_info['is_strict_checkpoint'] else '否'}")
    print(f"  检查点推断词表大小: {checkpoint_info['vocab_size'] if checkpoint_info['vocab_size'] else '未推断出'}")

    # --- 根据检查点信息和 strict_model 标志加载模型 ---
    try:
        deepsc_model, mine_network, channel_model = load_model_from_checkpoint(
            checkpoint_info, cfg, is_strict_version, actual_vocab_size
        )
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        traceback.print_exc()
        raise RuntimeError(f"无法加载模型: {e}") from e

    # 最终检查模型和信道是否成功加载
    if deepsc_model is None:
         raise RuntimeError("DeepSC 模型未能成功加载或初始化。")
    if channel_model is None:
         raise RuntimeError("信道模型未能成功创建。")


    # 验证词表大小 (再次检查，因为加载逻辑可能调整)
    loaded_vocab_size = None
    try:
        # 尝试多种方式获取嵌入层
        if hasattr(deepsc_model, 'encoder') and hasattr(deepsc_model.encoder, 'embedding'): # DeepSCStrict
             loaded_vocab_size = deepsc_model.encoder.embedding.num_embeddings
        elif hasattr(deepsc_model, 'encoder') and hasattr(deepsc_model.encoder, 'embed'): # DeepSC (standard)
             loaded_vocab_size = deepsc_model.encoder.embed.num_embeddings
        elif hasattr(deepsc_model, 'embedding'): # 可能的简化模型
            loaded_vocab_size = deepsc_model.embedding.num_embeddings
        else:
             print("  警告: 无法自动检测模型的词表大小。")

        if loaded_vocab_size is not None and loaded_vocab_size != actual_vocab_size:
            print(f"  警告: 最终加载的模型词表大小 ({loaded_vocab_size}) 与当前词表 ({actual_vocab_size}) 不符。")
            print("        评估可能不准确。建议使用与训练时匹配的词表。")
            # (可选) 强制更新配置以匹配模型？这可能导致后续 DataLoader 出错，暂时不这样做。
            # cfg.model.vocab_size = loaded_vocab_size
            # cfg.data.vocab_size = loaded_vocab_size
        elif loaded_vocab_size is not None:
             print(f"  [√] 模型词表大小 ({loaded_vocab_size}) 与当前词表匹配。")

    except AttributeError:
        print("  警告: 尝试检查模型词表大小时发生属性错误。")


    # 转移到设备并设置为评估模式
    deepsc_model = deepsc_model.to(device).eval()
    if mine_network: mine_network = mine_network.to(device).eval()
    channel_model = channel_model.to(device).eval() # 信道模型也需要移到设备

    # 冻结所有参数
    for param in deepsc_model.parameters(): param.requires_grad = False
    if mine_network:
        for param in mine_network.parameters(): param.requires_grad = False
    print(f"  模型已移至 {device} 并设置为评估模式。")
    print(f"  使用的信道模型 (DeepSC): {channel_model.__class__.__name__}")

    # ---------- 4. 加载评估数据 ----------
    print("\n--- 4. 加载评估数据集 ---")
    print(f"  加载验证集数据: {val_pkl}")
    eval_batch_size = cfg.get('eval_batch_size', 32) # 允许在配置中设置评估批大小
    print(f"  评估批大小: {eval_batch_size}")
    num_available_cpus = os.cpu_count()
    num_workers = min(cfg.get("num_workers", num_available_cpus if num_available_cpus is not None else 1), 4) # 使用配置或CPU核心数，最多4个
    print(f"  数据加载器 Workers: {num_workers}")
    test_loader = make_dataloader(
        str(val_pkl),
        batch_size = eval_batch_size,
        pad_idx = pad_idx,
        shuffle = False,
        num_workers = num_workers,
    )
    try:
        dataset_len = len(test_loader.dataset)
        print(f"  测试集样本数量: {dataset_len} 句")
    except TypeError:
        print("  测试集样本数量: 未知 (IterableDataset)")

    # ---------- 5. 初始化传统基线方法 ----------
    print("\n--- 5. 初始化传统基线方法 ---")
    baselines: Dict[str, Any] = {}
    baseline_methods_to_compare: List[str] = []
    if local_baselines_available:
        print("  尝试初始化传统源编码和信道编码器...")
        try:
            # 源编码器
            baselines['huffman'] = HuffmanEncoder(actual_vocab_size) # 使用实际词表大小
            bits_per_token_fixed = int(np.ceil(np.log2(actual_vocab_size)))
            baselines['fixed'] = FixedLengthEncoder(actual_vocab_size, bits_per_token=bits_per_token_fixed)
            print(f"    HuffmanEncoder (vocab={actual_vocab_size})")
            print(f"    FixedLengthEncoder (vocab={actual_vocab_size}, bits_per_token={bits_per_token_fixed})")

            # 信道编码器
            baselines['turbo'] = TurboCoder(rate=1/3, iterations=5) # 假设码率1/3
            print(f"    TurboCoder (rate=1/3, iterations=5)")
            # 检查 RS(255, 223)
            baselines['rs'] = RSCoder(n=255, k=223) # 常用 RS(255, 223)
            print(f"    RSCoder (n=255, k=223)")

            # 定义要比较的组合
            baseline_methods_to_compare = [
                'huffman_turbo', 'fixed_turbo', 'huffman_rs', 'fixed_rs'
            ]
            print(f"  成功初始化基线方法: {', '.join(baselines.keys())}")
            print(f"  将比较以下组合: {', '.join(baseline_methods_to_compare)}")
        except Exception as e:
            print(f"  错误: 初始化传统基线方法时发生异常: {e}")
            traceback.print_exc()
            print("        将跳过与传统方法的比较。")
            local_baselines_available = False # 出错则禁用基线比较
    else:
        print("  跳过初始化传统基线方法（库未导入或导入失败）。")

    # ---------- 6. 设置评估参数和结果存储 ----------
    print("\n--- 6. 设置评估参数和结果存储 ---")
    snrs_db = cfg.get("eval_snrs", [0, 3, 6, 9, 12, 15, 18]) # 从配置获取SNR列表
    print(f"  将评估以下 SNR (dB) 点: {snrs_db}")
    # 初始化结果字典
    results: Dict[str, List] = {'snr': snrs_db}
    results['deepsc_bleu'] = []
    results['deepsc_sim'] = []
    results['deepsc_mi'] = []
    if local_baselines_available:
        for method in baseline_methods_to_compare:
            results[f'{method}_bleu'] = []
            results[f'{method}_sim'] = []
    print("  结果存储结构已初始化。")

    # 设置 AMP 上下文
    amp_enabled = (str(cfg.get('precision', '32')) in ['16', '16-mixed', 16]) and device.type == 'cuda'
    amp_ctx = torch.cuda.amp.autocast() if amp_enabled else nullcontext()
    print(f"  自动混合精度 (AMP) 评估: {'启用' if amp_enabled else '禁用'}")

    # ---------- 7. 循环评估不同 SNR ----------
    print("\n--- 7. 开始在不同 SNR 下评估性能 ---")
    baseline_eval_max_batches = cfg.get("baseline_eval_max_batches", 5) # 限制基线评估的批次数
    print(f"  传统基线将只评估前 {baseline_eval_max_batches} 个批次以节省时间。")

    for snr_db in snrs_db:
        print(f"\n  评估 SNR = {snr_db} dB")
        # DeepSC 使用的噪声方差 sigma^2
        snr_lin_deepsc = 10.0 ** (snr_db / 10.0)
        deepsc_noise_var_squared = 1.0 / (2.0 * max(snr_lin_deepsc, 1e-10))

        # 初始化当前 SNR 的临时结果列表/字典
        current_snr_deepsc_bleu: List[float] = []
        current_snr_deepsc_sim: List[float] = []
        current_snr_deepsc_mi: List[float] = []
        current_snr_baseline_results: Dict[str, Dict[str, Any]] = {}
        if local_baselines_available:
            current_snr_baseline_results = {method: {'bleu': [], 'sim': [], 'count': 0, 'total_samples': 0} for method in baseline_methods_to_compare}

        # 创建进度条
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc=f"  SNR {snr_db} dB", leave=False, unit="batch")

        for batch_idx, batch in pbar:
            # --- 7.1 评估 DeepSC ---
            with torch.no_grad(), amp_ctx:
                batch_tensor = batch.to(device)
                logits, tx, rx = None, None, None # 初始化
                try:
                    # 统一接口调用，期望返回 logits, tx, rx
                    # LitDeepSC.model.forward 或 DeepSCStrict.forward 都应支持 return_tx_rx=True
                    outputs = deepsc_model(batch_tensor, deepsc_noise_var_squared, channel_model, return_tx_rx=True)
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                         logits, tx, rx = outputs[:3]
                    elif isinstance(outputs, torch.Tensor): # 如果只返回 logits
                         logits = outputs
                         # 尝试再次调用只获取 tx, rx (效率较低，但确保能拿到)
                         try:
                              _, tx_only, rx_only = deepsc_model(batch_tensor, deepsc_noise_var_squared, channel_model, return_tx_rx=True)
                              tx, rx = tx_only, rx_only
                         except Exception:
                              print(f"\n    警告: 无法获取 tx, rx 用于 MI 计算 (SNR={snr_db}, Batch={batch_idx})")
                              tx, rx = None, None
                    else:
                         raise TypeError(f"模型返回类型未知: {type(outputs)}")

                    # 获取预测结果
                    pred_indices = logits.argmax(dim=-1)

                except Exception as e_deepsc:
                    print(f"\n错误: DeepSC 前向传播失败 (SNR={snr_db}, Batch={batch_idx}): {e_deepsc}")
                    traceback.print_exc()
                    continue # 跳过此批次的 DeepSC 评估

                # 计算 BLEU Score
                target_indices = batch_tensor[:, 1:] # 参考序列，去除 <START>
                # 对齐预测和目标长度
                pred_len = pred_indices.size(1)
                target_len = target_indices.size(1)
                if pred_len > target_len:
                    pred_indices_aligned = pred_indices[:, :target_len]
                elif pred_len < target_len:
                    padding = torch.full((pred_indices.size(0), target_len - pred_len), pad_idx, device=device, dtype=torch.long)
                    pred_indices_aligned = torch.cat([pred_indices, padding], dim=1)
                else:
                    pred_indices_aligned = pred_indices
                # 计算批次BLEU (累加每个样本的分数，最后求平均)
                # bleu_score 函数内部处理了批次平均，这里直接追加批次平均值
                batch_bleu_avg = bleu_score(pred_indices_aligned.cpu(), target_indices.cpu())
                current_snr_deepsc_bleu.append(batch_bleu_avg)

                # 计算 Sentence Similarity
                str_pred_list, str_ref_list = [], []
                for p_idx, r_idx in zip(pred_indices_aligned.cpu(), target_indices.cpu()):
                    str_pred_list.append(' '.join(vocab.decode(p_idx.tolist(), stop_at_end=True)))
                    str_ref_list.append(' '.join(vocab.decode(r_idx.tolist(), stop_at_end=True)))
                try:
                    # 计算批内所有样本的相似度，返回一个列表
                    batch_sim_list = sentence_similarity(str_pred_list, str_ref_list, device=device, max_length=cfg.model.get('max_len', 32))
                    current_snr_deepsc_sim.extend(batch_sim_list) # 将每个样本的相似度加入列表
                except Exception as sim_err:
                    print(f"\n警告: 计算 DeepSC 句子相似度时出错 (SNR={snr_db}, Batch={batch_idx}): {sim_err}")
                    # 出错则不添加此批次的相似度分数

                # 计算 Mutual Information (使用模型对应的 MINE)
                batch_mi = 0.0
                if mine_network is not None and tx is not None and rx is not None:
                    try:
                        tx_flat = tx.reshape(-1, tx.size(-1))
                        rx_flat = rx.reshape(-1, rx.size(-1))
                        # 过滤掉全零的填充部分
                        non_empty_mask = (tx_flat.abs().sum(dim=1) > 1e-6)
                        if non_empty_mask.sum() > 0:
                             tx_f = tx_flat[non_empty_mask]
                             rx_f = rx_flat[non_empty_mask]
                             mine_network.eval() # 确保在评估模式
                             batch_mi = mine_network(tx_f, rx_f).item()
                    except Exception as mi_err:
                        print(f"\n警告: 计算 DeepSC 互信息时出错 (SNR={snr_db}, Batch={batch_idx}): {mi_err}")
                current_snr_deepsc_mi.append(batch_mi) # MI 是整个批次的估计值

            # --- 7.2 评估传统基线 (仅评估前 N 个批次) ---
            if local_baselines_available and batch_idx < baseline_eval_max_batches:
                batch_np = batch.numpy() # 使用原始 batch (包含 <START>)

                for method_name in baseline_methods_to_compare:
                    source_coder_name, channel_coder_name = method_name.split('_')
                    source_coder = baselines[source_coder_name]
                    channel_coder = baselines[channel_coder_name]

                    batch_baseline_bleu_scores = []
                    batch_baseline_sim_scores = []
                    num_samples_processed_baseline = 0

                    try:
                        # 1. 源编码 (输入是原始句子，包含 <START>)
                        source_encoded_bits, original_source_bit_lengths = source_coder.encode(batch_np)
                        if source_encoded_bits.size == 0: continue # 如果源编码结果为空则跳过

                        # 2. 信道编码
                        channel_coded_output = channel_coder.encode(source_encoded_bits)
                        if channel_coded_output.size == 0: continue # 如果信道编码结果为空则跳过

                        # 3. 模拟信道 & 4. 信道解码
                        channel_decoded_bits = np.array([]) # 初始化
                        if channel_coder_name == 'turbo':
                             llr, noise_var_turbo = simulate_traditional_channel_llr(channel_coded_output, snr_db)
                             # Turbo 解码器内部会使用存储的 _current_original_lengths (编码时记录)
                             channel_decoded_bits = channel_coder.decode(llr, noise_var=noise_var_turbo)
                        elif channel_coder_name == 'rs':
                             try:
                                  # RS 需要硬判决符号输入
                                  # 模拟通道作用在 *比特* 上 (简化, BPSK + AWGN)
                                  llr, _ = simulate_traditional_channel_llr(channel_coded_output, snr_db)
                                  hard_bits = (llr < 0).astype(np.uint8)
                                  # 打包成字节
                                  received_symbols = pack_bits_to_bytes(hard_bits)
                                  # RS 解码器需要知道原始比特长度才能正确截断
                                  # RS 编码器内部会存储 _current_original_bit_lengths
                                  channel_decoded_bits = channel_coder.decode(received_symbols)
                             except Exception as e_rs:
                                  print(f"\n警告: RS 处理过程出错 (SNR={snr_db}, Batch={batch_idx}): {e_rs}")
                                  traceback.print_exc()
                                  channel_decoded_bits = np.array([]) # 保证后续判断 size
                        else:
                             print(f"警告: 未知的信道编码器名称 '{channel_coder_name}'")
                             continue

                        if channel_decoded_bits.size == 0: continue # 如果信道解码结果为空则跳过

                        # 5. 源解码 (需要原始比特长度列表 original_source_bit_lengths)
                        decoded_sentences_indices = source_coder.decode(channel_decoded_bits, original_source_bit_lengths)

                        # 6. 计算指标
                        str_pred_baseline, str_ref_baseline = [], []
                        valid_indices_count = 0
                        for i, decoded_indices in enumerate(decoded_sentences_indices):
                            if not decoded_indices: continue # 跳过空解码结果
                            num_samples_processed_baseline += 1
                            ref_indices_np = batch_np[i, 1:] # 参考，去掉 <START>

                            # BLEU: 比较解码结果和目标
                            dec_tensor = torch.tensor([decoded_indices], dtype=torch.long)
                            ref_tensor = torch.from_numpy(ref_indices_np.reshape(1, -1)).long()

                            pred_len_b = dec_tensor.size(1)
                            target_len_b = ref_tensor.size(1)
                            if pred_len_b == 0 or target_len_b == 0: continue

                            # 对齐长度
                            if pred_len_b > target_len_b:
                                dec_tensor_aligned = dec_tensor[:, :target_len_b]
                            elif pred_len_b < target_len_b:
                                padding_b = torch.full((1, target_len_b - pred_len_b), pad_idx, dtype=torch.long)
                                dec_tensor_aligned = torch.cat([dec_tensor, padding_b], dim=1)
                            else:
                                dec_tensor_aligned = dec_tensor
                            # 计算单个样本的 BLEU
                            batch_baseline_bleu_scores.append(bleu_score(dec_tensor_aligned, ref_tensor))

                            # Sentence Similarity
                            dec_str = ' '.join(vocab.decode(decoded_indices, stop_at_end=True))
                            ref_str = ' '.join(vocab.decode(ref_indices_np.tolist(), stop_at_end=True))
                            if dec_str and ref_str:
                                str_pred_baseline.append(dec_str)
                                str_ref_baseline.append(ref_str)
                                valid_indices_count += 1

                        # 计算批次相似度
                        if str_pred_baseline and str_ref_baseline:
                            try:
                                sim_scores_list = sentence_similarity(str_pred_baseline, str_ref_baseline, device=device, max_length=cfg.model.get('max_len', 32))
                                batch_baseline_sim_scores.extend(sim_scores_list) # 加入每个样本的分数
                            except Exception as sim_err_b:
                                print(f"\n警告: 计算基线 '{method_name}' 句子相似度时出错 (SNR={snr_db}, Batch={batch_idx}): {sim_err_b}")
                                # 不添加分数

                        # 累加结果 (存储每个样本的分数，最后求平均)
                        if batch_baseline_bleu_scores:
                             current_snr_baseline_results[method_name]['bleu'].extend(batch_baseline_bleu_scores)
                        if batch_baseline_sim_scores:
                             current_snr_baseline_results[method_name]['sim'].extend(batch_baseline_sim_scores)
                        current_snr_baseline_results[method_name]['count'] += 1 # 记录成功评估的批次数
                        current_snr_baseline_results[method_name]['total_samples'] += num_samples_processed_baseline # 记录总共处理的有效样本数

                    except Exception as e_baseline:
                        print(f"\n警告: 在 SNR={snr_db} dB, Batch={batch_idx} 评估基线 '{method_name}' 时出错: {e_baseline}")
                        traceback.print_exc() # 可选：打印详细堆栈

            # --- 7.3 更新进度条 ---
            pbar_metrics = {
                'BLEU': f"{batch_bleu_avg:.3f}",
                'Sim': f"{np.mean(batch_sim_list):.3f}" if 'batch_sim_list' in locals() and batch_sim_list else "N/A",
                'MI': f"{batch_mi:.3f}"
            }
            if local_baselines_available and batch_idx < baseline_eval_max_batches:
                first_baseline = baseline_methods_to_compare[0]
                if current_snr_baseline_results[first_baseline]['bleu']:
                    # 显示的是当前批次的基线平均BLEU
                    last_batch_baseline_bleu_avg = np.mean(batch_baseline_bleu_scores) if 'batch_baseline_bleu_scores' in locals() and batch_baseline_bleu_scores else 0.0
                    pbar_metrics[f"{first_baseline[:3]}BLEU"] = f"{last_batch_baseline_bleu_avg:.3f}"

            pbar.set_postfix(pbar_metrics)


        # --- 单个 SNR 评估结束 ---
        pbar.close() # 关闭当前 SNR 的进度条

        # 计算并记录 DeepSC 平均指标 (对整个数据集求平均)
        avg_deepsc_bleu = np.mean(current_snr_deepsc_bleu) if current_snr_deepsc_bleu else 0.0 # 对批次平均值求平均
        avg_deepsc_sim = np.mean(current_snr_deepsc_sim) if current_snr_deepsc_sim else 0.0 # 对所有样本分数求平均
        avg_deepsc_mi = np.mean(current_snr_deepsc_mi) if current_snr_deepsc_mi else 0.0 # 对批次MI估计值求平均
        results['deepsc_bleu'].append(avg_deepsc_bleu)
        results['deepsc_sim'].append(avg_deepsc_sim)
        results['deepsc_mi'].append(avg_deepsc_mi)

        # 计算并记录基线平均指标 (对评估过的样本求平均)
        if local_baselines_available:
            for method in baseline_methods_to_compare:
                # BLEU: 对所有评估样本的 BLEU 分数求平均
                avg_baseline_bleu = np.mean(current_snr_baseline_results[method]['bleu']) if current_snr_baseline_results[method]['bleu'] else 0.0
                # Sim: 对所有评估样本的 Sim 分数求平均
                avg_baseline_sim = np.mean(current_snr_baseline_results[method]['sim']) if current_snr_baseline_results[method]['sim'] else 0.0
                results[f'{method}_bleu'].append(avg_baseline_bleu)
                results[f'{method}_sim'].append(avg_baseline_sim)
                # 如果 count 为 0，表示该基线在此 SNR 下未能成功评估
                if current_snr_baseline_results[method]['count'] == 0:
                    print(f"  信息: 基线 '{method}' 在 SNR={snr_db} dB 下未能成功评估任何批次。")

        # 打印当前 SNR 汇总
        print(f"  SNR={snr_db} dB 汇总:")
        print(f"    DeepSC: BLEU={avg_deepsc_bleu:.4f}, Sim={avg_deepsc_sim:.4f}, MI={avg_deepsc_mi:.4f}")
        if local_baselines_available:
            for method in baseline_methods_to_compare:
                bleu_val = results[f'{method}_bleu'][-1] # 获取刚添加的平均值
                sim_val = results[f'{method}_sim'][-1]
                # 只有当成功评估了至少一个批次时才打印结果
                if current_snr_baseline_results[method]['count'] > 0:
                    print(f"    {method.replace('_','+').upper()}: BLEU={bleu_val:.4f}, Sim={sim_val:.4f} (基于 {current_snr_baseline_results[method]['total_samples']} 样本 from {current_snr_baseline_results[method]['count']} 批次)")
                else:
                    print(f"    {method.replace('_','+').upper()}: (未成功评估)")

    # ---------- 8. 保存结果与可视化 ----------
    print("\n--- 8. 保存结果并生成图表 ---")
    # 使用 Hydra 的运行时输出目录
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) if hydra.core.hydra_config.HydraConfig.initialized() else Path('evaluation_results_compare')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  结果将保存到: {output_dir.resolve()}")

    # 保存 CSV
    results_df = pd.DataFrame(results)
    # 清理检查点名称以用作文件前缀
    output_prefix = ckpt_path.stem.replace('.ckpt','').replace('.pt','')
    if not output_prefix or output_prefix == 'best_model': # 如果是通用名称，添加更多信息
        model_type_prefix = "strict" if is_strict_version else "standard"
        output_prefix = f"deepsc_{model_type_prefix}_{cfg.data.channel}"
    csv_filename = output_dir / f'{output_prefix}_baseline_comparison.csv'
    try:
        results_df.to_csv(csv_filename, index=False, float_format='%.5f')
        print(f"  详细结果已保存到 CSV 文件: {csv_filename}")
    except Exception as e_csv:
        print(f"  错误: 保存 CSV 文件失败: {e_csv}")

    # --- 绘图 ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid') # 使用兼容的样式
        plot_channel_name = channel_model.__class__.__name__ if channel_model else cfg.data.channel

        # 绘制 BLEU 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(results['snr'], results['deepsc_bleu'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC')
        if local_baselines_available:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(baseline_methods_to_compare)))
            markers = ['s', '^', 'x', 'd']
            for i, method in enumerate(baseline_methods_to_compare):
                if any(results[f'{method}_bleu']): # 仅当有有效数据点时才绘制
                    plt.plot(results['snr'], results[f'{method}_bleu'],
                             marker=markers[i % len(markers)], linestyle='--', linewidth=1.5, markersize=5,
                             color=colors[i], label=method.replace('_', '+').upper())
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('BLEU-1 Score', fontsize=12)
        plt.title(f'BLEU Score vs. SNR Comparison ({plot_channel_name})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.xticks(snrs_db)
        plt.ylim(bottom=-0.05, top=1.05) # 稍微扩展y轴范围
        plt.tight_layout()
        bleu_plot_filename = output_dir / f'{output_prefix}_bleu_comparison.png'
        plt.savefig(bleu_plot_filename, dpi=300, bbox_inches='tight')
        print(f"  BLEU 比较图已保存: {bleu_plot_filename}")
        plt.close()

        # 绘制句子相似度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(results['snr'], results['deepsc_sim'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC')
        if local_baselines_available:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(baseline_methods_to_compare)))
            markers = ['s', '^', 'x', 'd']
            for i, method in enumerate(baseline_methods_to_compare):
                if any(results[f'{method}_sim']):
                    plt.plot(results['snr'], results[f'{method}_sim'],
                             marker=markers[i % len(markers)], linestyle='--', linewidth=1.5, markersize=5,
                             color=colors[i], label=method.replace('_', '+').upper())
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Sentence Similarity', fontsize=12)
        plt.title(f'Sentence Similarity vs. SNR Comparison ({plot_channel_name})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.xticks(snrs_db)
        plt.ylim(bottom=-0.05, top=1.05) # 相似度范围 [0, 1]
        plt.tight_layout()
        sim_plot_filename = output_dir / f'{output_prefix}_similarity_comparison.png'
        plt.savefig(sim_plot_filename, dpi=300, bbox_inches='tight')
        print(f"  句子相似度比较图已保存: {sim_plot_filename}")
        plt.close()

        # 绘制互信息曲线 (仅 DeepSC)
        if results['deepsc_mi'] and any(m != 0.0 for m in results['deepsc_mi']): # 确保有非零 MI 数据
            plt.figure(figsize=(10, 6))
            plt.plot(results['snr'], results['deepsc_mi'], marker='o', linestyle='-', linewidth=2, markersize=6, label='DeepSC MI Lower Bound', color='purple')
            plt.xlabel('SNR (dB)', fontsize=12)
            plt.ylabel('Mutual Information Lower Bound (nats)', fontsize=12)
            plt.title(f'DeepSC Mutual Information vs. SNR ({plot_channel_name})', fontsize=14)
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.legend(fontsize=10)
            plt.xticks(snrs_db)
            plt.tight_layout()
            mi_plot_filename = output_dir / f'{output_prefix}_mi_curve.png'
            plt.savefig(mi_plot_filename, dpi=300, bbox_inches='tight')
            print(f"  DeepSC 互信息曲线图已保存: {mi_plot_filename}")
            plt.close()
        else:
            print("  跳过绘制互信息曲线（无有效数据或 MINE 未加载）。")

    except Exception as e_plot:
        print(f"  错误: 生成图表时失败: {e_plot}")
        traceback.print_exc()

    print(f"\n评估完成！结果已保存到目录: {output_dir.resolve()}")

# ---------- 程序入口点 ----------
if __name__ == '__main__':
    main()