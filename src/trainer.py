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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.args = args

    def train_one_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()  # 设置为训练模式
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.args.log_transform:
                targets = torch.exp(targets)

            if isinstance(self.optimizer, torch.optim.LBFGS):
                def closure():
                    self.optimizer.zero_grad()
                    outputs = torch.exp(self.model(inputs))
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    return loss

                # LBFGS 的 step 需要 closure
                loss = self.optimizer.step(closure)
                running_loss += loss.item() * inputs.size(0)

            else:
                self.optimizer.zero_grad()
                if self.args.log_transform:
                    outputs = torch.exp(self.model(inputs))
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
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

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.args.log_transform:
                    outputs, targets = torch.exp(self.model(inputs)), torch.exp(targets)
                else:
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
        如果 val_loader 为 None，则不进行验证和早停，直接训练指定 epochs。
        """
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=patience // 2)
        patience_counter = 0
        save_path_dir = os.path.join(save_root, self.args.model)
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_info = f"{self.args.hidden_layers}-epochs{self.args.epochs}-lr{self.args.lr}"

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader)
            elapsed = time.time() - start_time

            if val_loader:
                val_loss, val_mae = self.evaluate(val_loader)
                print(f'Epoch {epoch + 1}/{epochs} [{elapsed:.0f}s] - '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}g')
                scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss, self.best_val_mae = val_loss, val_mae
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    print('Validation loss decreased.')
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs with no improvement.')
                    break
            else:
                print(f'Epoch {epoch + 1}/{epochs} [{elapsed:.0f}s] - Train Loss: {train_loss:.4f}')
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_wts)
        print("Finished training. Loaded best model weights.")

        if val_loader:
            model_filename = f'best_{timestamp}-val_mae{self.best_val_mae:.0f}g-{model_info}.pth'
        else:
            model_filename = f'final_{timestamp}-{model_info}.pth'

        if self.args.save:
            torch.save(self.best_model_wts, os.path.join(save_path_dir, model_filename))