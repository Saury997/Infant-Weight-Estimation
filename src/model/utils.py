#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/16 22:27 
* Project: Infant-Weight-Estimation 
* File: utils.py
* IDE: PyCharm 
* Function:
"""
from typing import Optional

from torch import nn


def _activation(name: str) -> nn.Module:
    name = (name or 'relu').lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu' or name == 'swish':
        return nn.SiLU()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _norm1d(kind: Optional[str], num_channels: int) -> nn.Module:
    if not kind:
        return nn.Identity()
    k = kind.lower()
    if k in ['bn', 'batch', 'batchnorm']:
        return nn.BatchNorm1d(num_channels)
    if k in ['ln', 'layer', 'layernorm']:
        return nn.LayerNorm(normalized_shape=num_channels)
    if k in ['none', 'identity', 'id']:
        return nn.Identity()
    raise ValueError(f"Unsupported norm kind: {kind}")


def _init_weights(m: nn.Module, init_type: str = 'xavier'):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        init_type = init_type.lower()
        if init_type == 'uniform':
            nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        elif init_type == 'normal':
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(m.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")
        if m.bias is not None:
            nn.init.zeros_(m.bias)