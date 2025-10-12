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
import torch
from tqdm import tqdm
from loguru import logger
import time
import copy


class Trainer:
    """
    一个封装了训练和评估逻辑的类。
    """

    def __init__(self, model, criterion, optimizer, config, writer, fold):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = config.others.device
        self.epochs = config.training.epochs
        self.model = model.to(self.device)
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_val_loss, self.best_val_mae = float('inf'), float('inf')
        self.training_cfg = config.training
        self.writer = writer
        self.fold = fold
        self.log_transform = config.data.log_transform

    def train_one_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()  # 设置为训练模式
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Fold {self.fold} Epoch {epoch + 1}/{self.epochs} [T]",
                            leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.log_transform:
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
                if self.log_transform:
                    outputs = torch.exp(self.model(inputs))
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) + self.model.regularization_loss()
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
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.log_transform:
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

    def fit(self, train_loader, val_loader, save_root, scheduler):
        """
        完整训练流程，包含早停和学习率调度。
        如果 val_loader 为 None，则不进行验证和早停，直接训练指定 epochs。
        """
        patience_counter = 0

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            elapsed = time.time() - start_time

            if val_loader:
                val_loss, val_mae = self.evaluate(val_loader)
                logger.info(f"Fold {self.fold} | Epoch {epoch+1:03d}/{self.epochs} [{elapsed:.2f}s] -> Train Loss: {train_loss:.2f}, "
                            f"Val Loss: {val_loss:.2f}, Val MAE: {val_mae:.2f}g")
                if self.writer:
                    log_prefix = f'Fold-{self.fold}/'
                    self.writer.add_scalar(f'{log_prefix}Loss/Train', train_loss, epoch)
                    self.writer.add_scalar(f'{log_prefix}Loss/Validation', val_loss, epoch)
                    self.writer.add_scalar(f'{log_prefix}MAE/Validation', val_mae, epoch)
                    self.writer.add_scalar(f'{log_prefix}Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss, self.best_val_mae = val_loss, val_mae
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.training_cfg.patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch+1}.")
                    break
            else:
                logger.info(f"Final Train | Epoch {epoch+1:03d}/{self.epochs} [{elapsed:.2f}s] -> Train Loss: {train_loss:.4f}")
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                if self.writer:
                    self.writer.add_scalar('Final/Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Final/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

        self.model.load_state_dict(self.best_model_wts)
        logger.info(f"Finished training Fold {self.fold}. Loaded best model weights.")

        if val_loader:
            model_filename = f'fold{self.fold}_best_model.pth'
        else:
            model_filename = 'final_model.pth'

        torch.save(self.best_model_wts, os.path.join(save_root, model_filename))
        logger.success(f"Best model for Fold {self.fold} saved.")