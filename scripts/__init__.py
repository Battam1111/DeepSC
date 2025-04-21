# ================================================================================
# 文件: /home/star/Yanjun/PolyUCourse/DeepSC/scripts/__init__.py
# (已修改，添加环境变量设置以解决 tokenizers 并行警告)
# --------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
scripts 包的初始化
=================

作用：
1. **自动把项目根目录加入 sys.path**
   允许在 *任意工作目录* 下执行：
       python -m scripts.train
       python -m scripts.evaluate
   等命令而无需手动设置 PYTHONPATH 或“cd ..”。
   如果根目录已存在于 sys.path，则不会重复插入。

2. **设置 TOKENIZERS_PARALLELISM 环境变量**
   解决 Hugging Face tokenizers 在 DataLoader 使用多进程 (num_workers > 0) 时
   可能出现的关于 fork 后并行性的警告，并避免潜在的死锁风险。
   将其设置为 'false' 会禁用 tokenizers 内部的并行，通常对整体性能影响不大。
"""

from __future__ import annotations
import sys
import os # 导入 os 模块
from pathlib import Path

# --- 设置 Tokenizers 并行性环境变量 ---
# 必须在 transformers 或 tokenizers 被导入之前设置
# 设置为 'false' 可以避免在 DataLoader worker (fork 产生) 中出现死锁风险和警告
# 设置为 'true' 则表示明确允许，可能会保留并行但有潜在风险（不推荐，除非明确需要）
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# print("[scripts/__init__.py] 设置 TOKENIZERS_PARALLELISM=false") # 可选：用于调试确认

# --- 添加项目根目录到 sys.path ---
# scripts/__init__.py 所在目录:  <proj_root>/scripts
_THIS_DIR = Path(__file__).resolve().parent
# 项目根目录:              <proj_root>
_PROJECT_ROOT = _THIS_DIR.parent

# 将项目根目录放到 sys.path 最前面，确保 import deepsc 可以成功
root_str = str(_PROJECT_ROOT)
if root_str not in sys.path:
    # 置于最前，可覆盖可能存在的同名第三方包
    sys.path.insert(0, root_str)
    # print(f"[scripts/__init__.py] 添加项目根目录到 sys.path: {root_str}") # 可选：调试信息

# 可选：暴露一个常量，方便脚本中引用
PROJECT_ROOT = _PROJECT_ROOT

# ------------------- 说明 -------------------
# 为什么选择在 scripts/__init__.py 而不是每个脚本顶部修改 sys.path 或环境变量？
#   1) python -m scripts.xxx 时，解释器会首先导入 scripts 包，
#      因而这里的修改最早生效，后续 import 都能受益。
#   2) 集中处理一次，避免每个脚本重复写模板代码。
# -------------------------------------------