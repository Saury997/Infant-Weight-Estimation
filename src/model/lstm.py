#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/16 22:33 
* Project: Infant-Weight-Estimation 
* File: lstm.py
* IDE: PyCharm 
* Function:
"""
from typing import Optional

import torch
from torch import nn

from src.model.utils import _activation, _init_weights


class LSTMRegressor(nn.Module):
    """
    LSTM Regressor
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        mlp_hidden: int = 64,
        act: str = 'relu',
        init_type: str = 'xavier',
        raw_input_dim: Optional[int] = None,  # if actual last-dim != input_dim
        use_lazy_proj: bool = True
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if raw_input_dim and raw_input_dim != input_dim:
            self.proj = nn.Linear(raw_input_dim, input_dim)
        else:
            self.proj = nn.LazyLinear(input_dim) if use_lazy_proj else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )
        last_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(last_dim, mlp_hidden),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        self.apply(lambda m: _init_weights(m, init_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, F, 1)
        x = self.proj(x)
        o, (h_n, c_n) = self.lstm(x)
        last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.lstm.bidirectional else h_n[-1]
        return self.head(last).squeeze(-1)