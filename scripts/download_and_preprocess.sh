#!/bin/bash
# -*- coding: utf-8 -*-
# 数据下载和预处理脚本
# 下载并预处理欧洲议会数据集，生成训练集、测试集和词表

set -e  # 出错即退出

# 创建数据目录
mkdir -p data/europarl

# 检查是否需要下载原始数据
if [ ! -f "data/europarl/europarl-v7.fr-en.en" ]; then
    echo "下载欧洲议会数据集..."
    wget -P data/tmp http://www.statmt.org/europarl/v7/fr-en.tgz
    
    echo "解压数据集..."
    mkdir -p data/tmp
    tar -xzf data/tmp/fr-en.tgz -C data/tmp
    
    echo "移动英文文件..."
    mv data/tmp/europarl-v7.fr-en.en data/europarl/
    
    echo "清理临时文件..."
    rm -rf data/tmp
else
    echo "原始数据已存在，跳过下载步骤"
fi

# 检查是否需要预处理
if [ ! -f "data/europarl/train_data.pkl" ] || [ ! -f "data/europarl/test_data.pkl" ] || [ ! -f "data/europarl/vocab.json" ]; then
    echo "预处理数据集..."
    python -m scripts.preprocess
else
    echo "预处理数据已存在，跳过预处理步骤"
fi

echo "数据准备完成！"