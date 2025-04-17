# -*- coding: utf-8 -*-
"""
EuroParl 数据加载（改进版，支持两种 pkl 格式）
=============================================
1. mmap 惰性加载：若是“多条记录逐条 dump”格式（每句一条 pickle），
   则不一次性载入，避免爆内存；
2. 单次全量加载：若是“一次性 dump 整个列表”格式（List[List[int]]），
   自动 fallback 到内存列表，按元素逐句访问；
3. map‐style Dataset：支持 __len__ 和 __getitem__；
4. 分桶批次（BucketBatch）：将相似长度句子放同一批，降低 padding 开销；
   若无 torchdata.bucketbatch，则退化到原生 DataLoader + collate_fn。

使用方式与原来完全一致，不需要额外修改脚本。
"""

from __future__ import annotations
import pickle, mmap
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入 torchdata 的 IterableWrapper 与 bucketbatch
try:
    from torchdata.datapipes.iter import IterableWrapper
    from torchdata.datapipes.iter.combining import bucketbatch
except ImportError:
    try:
        from torchdata.datapipes.iter import IterableWrapper, bucketbatch
    except ImportError:
        IterableWrapper = None
        bucketbatch = None


class EuroParlDataset(Dataset):
    """
    map‐style Dataset，支持 __len__ 和 __getitem__。
    兼容两种 pkl 文件格式：
      1) 多条记录逐条 dump：List[int]，分多次 pickle.dump
      2) 一次性 dump 整个列表：List[List[int]]，一次 pickle.dump

    Args:
        pkl_path: pickle 文件路径
    """
    def __init__(self, pkl_path):
        super().__init__()
        self.pkl_path = Path(pkl_path)
        self._f = open(self.pkl_path, 'rb')          # 常驻文件句柄
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        # —— 第一步：用 mmap 扫描所有 pickle dump 的偏移量 —— #
        offsets: List[int] = []
        with open(self.pkl_path, 'rb') as f, \
             mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            while True:
                pos = mm.tell()
                try:
                    pickle.load(mm)
                    offsets.append(pos)
                except EOFError:
                    break

        # —— fallback：若只有 1 个 offset，尝试判断是不是“一次 load 列表”格式 —— #
        if len(offsets) == 1:
            try:
                with open(self.pkl_path, 'rb') as f:
                    obj = pickle.load(f)
                # 如果读出的是 List[List[int]]，切换到内存列表分支
                if (isinstance(obj, list)
                        and len(obj) > 0
                        and all(isinstance(item, (list, tuple)) for item in obj)):
                    self.data = obj            # 内存列表，每个元素是一条句子的 int 序列
                    self.use_mmap = False
                    return
            except Exception:
                # 任意异常都忽略，回到 mmap 分支
                pass

        # —— 常规分支：多记录 mmap 加载 —— #
        self.offsets = offsets
        self.use_mmap = True

    def __len__(self) -> int:
        if not self.use_mmap:
            return len(self.data)
        else:
            return len(self.offsets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回一个 1D LongTensor，形状 [seq_len]。
        """
        if not self.use_mmap:
            sent = self.data[idx]
        else:
            self._mm.seek(self.offsets[idx])
            sent = pickle.load(self._mm)
        return torch.tensor(sent, dtype=torch.long)


def make_dataloader(
    pkl_path: str,
    batch_size: int,
    pad_idx: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    创建 DataLoader：
      - 优先使用 torchdata.bucketbatch 分桶；
      - 若环境中未安装/不支持，则退化为普通 DataLoader + collate_fn。

    参数:
      pkl_path    : EuroParl pickle 文件路径（见 EuroParlDataset 文档）
      batch_size  : 每批样本数量
      pad_idx     : PAD token 的索引，用于填充
      num_workers : DataLoader 并行 worker 数
      shuffle     : 是否打乱序列（仅影响采样顺序，不影响 batch 内句长排序）

    返回:
      torch.utils.data.DataLoader
    """
    ds = EuroParlDataset(pkl_path)

    # 自定义 collate_fn：按批内最长序列做 padding
    def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(x.size(0) for x in batch)
        out = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        for i, seq in enumerate(batch):
            out[i, : seq.size(0)] = seq
        return out

    # 如果可用 IterableWrapper + bucketbatch，则优先使用 DataPipe 分桶
    if IterableWrapper is not None and bucketbatch is not None:
        dp = IterableWrapper(range(len(ds)))
        if shuffle:
            dp = dp.shuffle()
        # map 索引到实际句子 Tensor
        dp = dp.map(lambda i: ds[i])
        # 分桶 batch：相似长度自动聚合，再传给 collate_fn
        dp = bucketbatch(
            dp,
            batch_size=batch_size,
            bucket_size_multiplier=50,
            sort_key=lambda seq: seq.size(0)
        )
        return DataLoader(
            dp,
            batch_size=None,       # DataPipe 已经是批了
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    # fallback：普通 DataLoader(map-style)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
