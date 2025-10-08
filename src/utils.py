#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/7 14:59 
* Project: InfantWeight 
* File: utils.py
* IDE: PyCharm 
* Function:
"""
import os
import random

import pandas as pd
import torch.distributed as dist
import numpy as np
import torch
from torch.optim import AdamW, SGD, LBFGS
from muon import MuonWithAuxAdam
from model import MLP, KAN


def get_optimizer(model, optimizer, lr):
    """根据参数初始化优化器"""
    if optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer == 'Muon':   # TODO: bugs need to fix.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group(backend='nccl', rank=0, world_size=1)
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        nonhidden_params = [*model.head.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=lr, weight_decay=0.01),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif optimizer == 'LBFGS':  # The effect is extremely poor, so it was disabled.
        # optimizer = LBFGS(model.parameters(), lr=lr)
        raise NotImplementedError("The effect of LBFGS optimizer is extremely poor, so it was disabled.")
    else:
        raise ValueError("Invalid optimizer. Please choose from ['AdamW', 'SGD', 'Muon', 'LBFGS'].")
    return optimizer


def get_model(args, input_dim):
    """根据参数初始化模型"""
    if args.model == 'MLP':
        model = MLP(input_dim=input_dim, hidden_layers=args.hidden_layers, dropout_rate=args.dropout, init_type=args.init_type)
    elif args.model == 'KAN':
        model = KAN(layers_hidden=[input_dim] + args.hidden_layers + [1])
    else:
        raise ValueError(f"Invalid model: {args.model}. Please choose from ['MLP', KAN].")
    return model


def set_seed(seed):
    """设置随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_distribution(y_train, y_test, y_bins_train, y_bins_test):
    """
    检查训练集和测试集的分箱类别分布及统计信息。

    Args:
        y_train (pd.Series): 训练集真实目标值（连续数值，如出生体重）
        y_test (pd.Series): 测试集真实目标值
        y_bins_train (pd.Series): 训练集分箱类别
        y_bins_test (pd.Series): 测试集分箱类别
    """
    def stats_summary(y, y_bins, name="数据集"):
        df = pd.DataFrame({"value": y, "bin": y_bins})
        print(f"\n{name} 样本总数: {len(y)}")
        print(f"{name} 按类别统计：")
        summary = df.groupby("bin")["value"].agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            max="max"
        )
        summary["ratio"] = summary["count"] / len(y)
        print(summary)

    print("==== 数据分布检查 ====")
    stats_summary(y_train, y_bins_train, "训练集")
    stats_summary(y_test, y_bins_test, "测试集")
