#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/7 13:03 
* Project: InfantWeight 
* File: mlp.py
* IDE: PyCharm 
* Function:
"""
import torch.nn as nn

from src.model.utils import _init_weights, _norm1d, _activation


class MLP(nn.Module):
    """
    一个标准的多层感知机模型，用于回归任务。
    模型结构: Input -> Linear -> BatchNorm -> ReLU -> Dropout -> ... -> Output
    """

    def __init__(self,
                 input_dim: int,
                 hidden_layers: list,
                 dropout_rate: float = 0.2,
                 init_type: str = 'xavier',
                 norm: str = 'batch',
                 act: str = 'relu'
    ) -> None:
        """
        初始化模型。

        Args:
            input_dim (int): 输入特征的数量。
            hidden_layers (list): 一个列表，包含每个隐藏层中的神经元数量。
                                  例如: [128, 64, 32]
            dropout_rate (float): Dropout 的概率。
            init_type (str): 参数初始化方式，可选 'xavier', 'kaiming', 'uniform' 等。
            norm (str): 批量归一化方式，可选 'batch', 'layer', 'instance' 等。
            act (str): 激活函数，可选 'relu', 'leaky_relu', 'elu' 等。
        """
        super(MLP, self).__init__()

        layers = []
        in_features = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(_norm1d(norm, h_dim))
            layers.append(_activation(act))
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_dim
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)
        self.apply(lambda m: _init_weights(m, init_type))

    def regularization_loss(self):
        """
        计算模型正则化损失。
        """
        return sum(map(lambda m: sum(map(lambda p: p.abs().sum(), m.parameters())), self.model.modules()))

    def forward(self, x):
        """定义前向传播路径。"""
        return self.model(x)