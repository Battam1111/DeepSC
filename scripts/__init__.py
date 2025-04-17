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

2. 如果根目录已存在于 sys.path，则不会重复插入。
"""

from __future__ import annotations
import sys
from pathlib import Path

# scripts/__init__.py 所在目录:  <proj_root>/scripts
_THIS_DIR = Path(__file__).resolve().parent
# 项目根目录:                  <proj_root>
_PROJECT_ROOT = _THIS_DIR.parent

# 将项目根目录放到 sys.path 最前面，确保 import deepsc 可以成功
root_str = str(_PROJECT_ROOT)
if root_str not in sys.path:
    # 置于最前，可覆盖可能存在的同名第三方包
    sys.path.insert(0, root_str)

# 可选：暴露一个常量，方便脚本中引用
PROJECT_ROOT = _PROJECT_ROOT

# ------------------- 说明 -------------------
# 为什么选择在 scripts/__init__.py 而不是每个脚本顶部修改 sys.path？
#   1) python -m scripts.xxx 时，解释器会首先导入 scripts 包，
#      因而这里的修改最早生效，后续 import 都能受益。
#   2) 集中处理一次，避免每个脚本重复写模板代码。
# -------------------------------------------
