# -*- coding: utf-8 -*-
"""
词表工具：加载、构建、编码、解码
================================

本模块提供 Vocab 类，用于：
1. 从 JSON 文件加载词表；
2. 将任意 token 列表编码为索引列表（encode）；
3. 将索引列表解码为 token 列表（decode）；
4. 将词表保存回 JSON 文件。
"""

import json
from pathlib import Path
from typing import List, Dict, Union

# 专用 token 及其固定索引
SPECIAL_TOKENS: Dict[str, int] = {
    '<PAD>':   0,   # 填充符
    '<START>': 1,   # 序列起始符
    '<END>':   2,   # 序列终止符
    '<UNK>':   3,   # 未知词
}


class Vocab:
    """
    词表管理类

    属性:
        token2idx (Dict[str, int]): 从 token 到索引的映射
        idx2token (Dict[int, str]): 从索引到 token 的映射（反向映射）
    """

    def __init__(self, token2idx: Dict[str, int]):
        """
        :param token2idx: 自定义词表（包含 SPECIAL_TOKENS）
        """
        self.token2idx = token2idx
        # 反转映射，供 decode 使用
        self.idx2token = {idx: tok for tok, idx in token2idx.items()}

    # ---------- 基本方法 ----------
    def __len__(self) -> int:
        """
        :return: 词表大小
        """
        return len(self.token2idx)

    def encode(self, tokens: List[str], allow_unk: bool = True) -> List[int]:
        """
        将 token 列表转为索引列表

        :param tokens:     要编码的 token 列表
        :param allow_unk:  是否将未登录词映射为 '<UNK>'，否则抛出 KeyError
        :return:           对应的索引列表
        """
        res: List[int] = []
        for t in tokens:
            if t not in self.token2idx:
                if allow_unk:
                    t = '<UNK>'
                else:
                    raise KeyError(f'词汇表不包含 token: "{t}"')
            res.append(self.token2idx[t])
        return res

    def decode(self,
               idxes: List[int],
               stop_at_end: bool = True
               ) -> List[str]:
        """
        将索引列表转为 token 列表

        :param idxes:       要解码的索引列表
        :param stop_at_end: 遇到 '<END>' 时是否停止后续解码
        :return:             对应的 token 列表
        """
        res: List[str] = []
        for i in idxes:
            tok = self.idx2token.get(i, '<UNK>')
            res.append(tok)
            if stop_at_end and tok == '<END>':
                break
        return res

    # ---------- IO 操作 ----------
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocab":
        """
        从 JSON 文件加载词表

        JSON 文件格式示例:
        {
          "token_to_idx": {
            "<PAD>": 0,
            "<START>": 1,
            ...
          }
        }

        :param path: JSON 文件路径（str 或 pathlib.Path）
        :return:     Vocab 实例
        """
        path = Path(path)
        with path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        token2idx = obj.get('token_to_idx')
        if token2idx is None or not isinstance(token2idx, dict):
            raise ValueError(f"{path} 中未找到 'token_to_idx' 字段或格式错误")
        # 将 key: str, value: 数字 转为正确类型
        token2idx = {str(k): int(v) for k, v in token2idx.items()}
        return cls(token2idx)

    def save(self, path: Union[str, Path]) -> None:
        """
        将当前词表保存为 JSON 文件，字段名为 'token_to_idx'

        :param path: 保存路径（str 或 pathlib.Path）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {'token_to_idx': self.token2idx}
        with path.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

