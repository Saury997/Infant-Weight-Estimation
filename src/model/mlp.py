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
import torch.nn.init as init


class MLP(nn.Module):
    """
    一个标准的多层感知机模型，用于回归任务。
    模型结构: Input -> Linear -> BatchNorm -> ReLU -> Dropout -> ... -> Output
    """

    def __init__(self, input_dim: int, hidden_layers: list, dropout_rate:float=0.2, init_type:str='xavier'):
        """
        初始化模型。

        Args:
            input_dim (int): 输入特征的数量。
            hidden_layers (list): 一个列表，包含每个隐藏层中的神经元数量。
                                  例如: [128, 64, 32]
            init_type (str): 参数初始化方式，可选 'xavier', 'kaiming', 'uniform' 等。
        """
        super(MLP, self).__init__()

        layers = []
        # 输入层
        in_features = input_dim

        # 动态构建隐藏层
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # Trick: 批量归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Trick: Dropout
            in_features = h_dim

        # 输出层
        layers.append(nn.Linear(in_features, 1))

        # 使用 Sequential 容器组合所有层
        self.model = nn.Sequential(*layers)

        # 参数初始化
        self.init_type = init_type
        self._initialize_weights()

    def regularization_loss(self):
        """
        计算模型正则化损失。
        """
        return sum(map(lambda m: sum(map(lambda p: p.abs().sum(), m.parameters())), self.model.modules()))

    def _initialize_weights(self):
        """
        根据指定的初始化方式初始化网络权重。
        """
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif self.init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_type == 'uniform':
                    init.uniform_(m.weight, -0.1, 0.1)
                elif self.init_type == 'normal':
                    init.normal_(m.weight, mean=0, std=0.1)

                # 偏置项初始化为0
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """定义前向传播路径。"""
        return self.model(x)