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
import torch.distributed as dist
import numpy as np
import torch
from torch.optim import AdamW, SGD
from muon import MuonWithAuxAdam
from model import MLP, KAN


def get_optimizer(model, optimizer, lr):
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
    else:
        raise ValueError("Invalid optimizer. Please choose from ['AdamW', 'SGD', 'Muon'].")
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
