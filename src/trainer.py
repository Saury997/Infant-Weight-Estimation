#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/7 13:07 
* Project: InfantWeight 
* File: trainer.py
* IDE: PyCharm 
* Function:
"""
import os

# trainer.py

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import copy


class Trainer:
    """
    一个封装了训练和评估逻辑的类。
    """

    def __init__(self, model, criterion, optimizer, args):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = args.device
        self.model = model.to(self.device)
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_val_loss, self.best_val_mae = float('inf'), float('inf')
        self.model_name = args.model
        self.args = args

    def train_one_epoch(self, train_loader):
        """训练一个 epoch。"""
        self.model.train()  # 设置为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def evaluate(self, data_loader):
        """在验证集或测试集上评估模型。"""
        self.model.eval()  # 设置为评估模式
        running_loss = 0.0
        running_mae = 0.0

        with torch.no_grad():  # 在评估期间不计算梯度
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 计算 MAE (平均绝对误差)
                mae = torch.abs(outputs - targets).mean()

                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_mae = running_mae / len(data_loader.dataset)
        return epoch_loss, epoch_mae

    def fit(self, train_loader, val_loader, epochs, patience, save_root):
        """
        完整训练流程，包含早停和学习率调度。

        Args:
            patience (int): 早停的耐心值。
            save_root (str):最佳模型保存根路径。
        """
        # Trick: 使用学习率调度器
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=patience // 2, verbose=True)

        patience_counter = 0
        save_path = os.path.join(save_root, self.model.__class__.__name__)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_mae = self.evaluate(val_loader)

            elapsed = time.time() - start_time

            print(f'Epoch {epoch + 1}/{epochs} [{elapsed:.0f}s] - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}g')

            scheduler.step(val_loss)

            # Trick: 早停逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_val_mae = val_loss, val_mae
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                print(f'Validation loss decreased.')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs with no improvement.')
                break

        # 加载性能最好的模型权重
        self.model.load_state_dict(self.best_model_wts)
        print("Finished training. Loaded best model weights.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        torch.save(self.best_model_wts, os.path.join(save_path, f'best{timestamp}-val_mae{self.best_val_mae:.0f}g-{self.args.hidden_layers}.pth'))