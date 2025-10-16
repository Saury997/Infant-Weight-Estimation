#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/16 22:25 
* Project: Infant-Weight-Estimation 
* File: cnn.py
* IDE: PyCharm 
* Function:
"""
from typing import List, Tuple

import torch
from torch import nn

from src.model.utils import _activation, _init_weights, _norm1d


class CNNRegressor1D(nn.Module):
    """Compact 1D CNN for regression.

    Handles (B, F) by reshaping to (B, 1, F).
    Handles (B, T, D) by transposing to (B, D, T).
    """
    def __init__(
        self,
        num_features: int,
        in_channels: int = 1,
        conv_channels: Tuple[int] = (16, 32),
        kernel_sizes: Tuple[int] = (5, 3),
        strides: Tuple[int] = (1, 1),
        norm: str = 'batch',
        act: str = 'relu',
        dropout: float = 0.1,
        mlp_hidden: int = 64,
        init_type: str = 'xavier'
    ) -> None:
        """
        Args:
            num_features (int): 输入特征数量。
            in_channels (int): 输入通道数。
            conv_channels (tuple): 卷积核数量。
            kernel_sizes (tuple): 卷积核大小。
            strides (tuple): 卷积步长。
            norm (str): 批量归一化方式，可选 'batch', 'layer', 'instance'。
            act (str): 激活函数，可选 'relu', 'leaky_relu', 'elu'。
            dropout (float): Dropout 概率。
            mlp_hidden (int): MLP 的隐藏层大小。
            init_type (str): 参数初始化方式，可选 'xavier', 'kaiming', 'uniform'。
        """
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(strides), "conv_channels, kernel_sizes, strides must align"
        layers: List[nn.Module] = []
        c_in = in_channels
        self.sequence_mode = False  # set True when seeing (B,T,D)
        for c_out, k, s in zip(conv_channels, kernel_sizes, strides):
            pad = k // 2
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=pad, bias=False),
                _norm1d(norm, c_out),
                _activation(act),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c_in, mlp_hidden),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        self.apply(_init_weights)
        self.num_features = num_features
        self.in_channels = in_channels
        self.apply(lambda m: _init_weights(m, init_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, F) or (B, T, D) or (B, C, L)
        if x.dim() == 2:  # (B, F)
            x = x.unsqueeze(1)  # (B, 1, F)
        elif x.dim() == 3:
            B, A, B_or_T = x.shape
            # If given (B, T, D) we want (B, D, T); heuristic: treat last dim as feature dim if smaller than 64? Better: assume (B, T, D)
            # We'll assume input is (B, T, D)
            x = x.transpose(1, 2)  # (B, D, T)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        out = self.head(x).squeeze(-1)
        return out