# -*- coding: utf-8 -*-
"""
统一随机种子工具
==================================================
在训练脚本最开头调用：
    from deepsc.utils.seed import set_global_seed
    set_global_seed(42, deterministic=True)
"""

import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic: bool = False):
    """
    设定 Python / NumPy / PyTorch 的随机种子，使结果可复现
    :param seed:      任意整数
    :param deterministic: True 时强制 cuDNN 进入确定性模式（会略降速）
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN 每次选择确定算法，关闭 benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
