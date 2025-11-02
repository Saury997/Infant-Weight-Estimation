import os
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import json

from utils import evaluate_regression, single_metrics


class Trainer:
    """
    一个封装了训练与评估逻辑的 Trainer 类，支持：
    - log 空间训练与反变换
    - 自动计算多种误差指标
    - 保存最优模型与指标文件
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
        self.targets_are_already_log = config.data.log_transform
        self.y_mean, self.y_std = None, None
        self.best_metrics = None

    # -------------------------------------------------------------------------
    def _compute_y_stats(self, train_loader):
        """计算训练集的 log-target 均值与标准差"""
        if not self.log_transform:
            return

        all_targets = []
        for _, targets in train_loader:
            all_targets.append(targets.view(-1, 1).to(self.device))

        if len(all_targets) == 0:
            raise ValueError("Empty train_loader when computing y stats.")

        all_targets = torch.cat(all_targets, dim=0)
        y_log = all_targets if self.targets_are_already_log else torch.log(all_targets)
        self.y_mean, self.y_std = y_log.mean(), y_log.std()

        if torch.isclose(self.y_std, torch.tensor(0., device=self.device)):
            logger.warning("y_std is zero, forcing to 1.0.")
            self.y_std = torch.tensor(1.0, device=self.device)

        logger.info(f"Fold {self.fold}: log-space y mean={self.y_mean:.6f}, std={self.y_std:.6f}")

    # -------------------------------------------------------------------------
    def _prepare_targets(self, targets, training=True):
        """log空间标准化或反标准化"""
        if not self.log_transform:
            return targets

        if training:
            y_log = targets if self.targets_are_already_log else torch.log(targets)
            if self.y_mean is None or self.y_std is None:
                self.y_mean, self.y_std = y_log.mean(), y_log.std()
            return (y_log - self.y_mean) / self.y_std
        else:
            y_pred_log = targets * self.y_std + self.y_mean
            return torch.exp(y_pred_log)

    # -------------------------------------------------------------------------
    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Fold {self.fold} Epoch {epoch+1}/{self.epochs} [T]", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets_norm = self._prepare_targets(targets, training=True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets_norm)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(train_loader.dataset)

    # -------------------------------------------------------------------------
    def evaluate(self, data_loader):
        """
        在验证集或测试集上评估模型
        """
        self.model.eval()
        running_loss = 0.0
        all_outputs, all_targets = [], []

        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                if self.log_transform:
                    outputs = self._prepare_targets(outputs, training=False)
                    targets = torch.exp(targets) if self.targets_are_already_log else targets
                else:
                    outputs = outputs
                    targets = targets

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

                if self.log_transform:
                    targets_norm = self._prepare_targets(targets, training=True)
                    loss = self.criterion(outputs, targets_norm)
                else:
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        y_pred = torch.cat(all_outputs, dim=0).numpy().flatten()
        y_true = torch.cat(all_targets, dim=0).numpy().flatten()

        metrics = evaluate_regression(y_true, y_pred)

        # 3800g 以上样本指标
        idx_3800 = np.where(y_true >= 3800)[0]
        if len(idx_3800) > 0:
            sub_y_true = y_true[idx_3800]
            sub_y_pred = y_pred[idx_3800]
            metrics_3800g = evaluate_regression(sub_y_true, sub_y_pred)
        else:
            metrics_3800g = None

        sample_metrics = []
        for i in range(len(y_true)):
            sample_metrics.append(single_metrics(np.array([y_true[i]]), np.array([y_pred[i]])))

        return epoch_loss, metrics, metrics_3800g, sample_metrics, torch.tensor(y_pred)

    # -------------------------------------------------------------------------
    def fit(self, train_loader, val_loader, save_root, scheduler):
        """主训练流程 + 模型保存 + 指标保存"""
        patience_counter = 0
        if self.log_transform:
            self._compute_y_stats(train_loader)

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            elapsed = time.time() - start_time

            if val_loader:
                val_loss, metrics, _, _, _ = self.evaluate(val_loader)

                logger.info(
                    f"Fold {self.fold} | Epoch {epoch+1}/{self.epochs} [{elapsed:.2f}s] "
                    f"Loss(T/V) {train_loss:.2f}/{val_loss:.2f} | "
                    f"MAE {metrics['MAE']:.2f} MAPE {metrics['MAPE']:.2f} RMSE {metrics['RMSE']:.2f} R2 {metrics['R2']:.2f} "
                    f"Err(Sys/Ran) {metrics['SystematicError']:+.2f}/{metrics['RandomError']:+.2f}% "
                    f"w10p {metrics['With10pct']:.2f}%"
                )

                if self.writer:
                    prefix = f'Fold-{self.fold}/'
                    self.writer.add_scalar(f'{prefix}Loss/Train', train_loss, epoch)
                    self.writer.add_scalar(f'{prefix}Loss/Val', val_loss, epoch)
                    self.writer.add_scalar(f'{prefix}MAE/Val', metrics['MAE'], epoch)
                    self.writer.add_scalar(f'{prefix}LR', self.optimizer.param_groups[0]['lr'], epoch)

                scheduler.step(val_loss)

                if metrics['MAE'] < self.best_val_mae:
                    self.best_val_mae, self.best_val_loss = metrics['MAE'], val_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.best_metrics = metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.training_cfg.patience:
                    logger.warning(f"Early stopping at epoch {epoch+1}.")
                    break
            else:
                _, metrics, _, _, _ = self.evaluate(train_loader)
                logger.info(
                    f"Train {epoch+1}/{self.epochs} [{elapsed:.2f}s] Loss {train_loss:.2f} | "
                    f"MAE {metrics['MAE']:.2f} MAPE {metrics['MAPE']:.2f} RMSE {metrics['RMSE']:.2f} R2 {metrics['R2']:.2f} "
                    f"Err(Sys/Ran) {metrics['SystematicError']:+.2f}/{metrics['RandomError']:+.2f}% "
                    f"w10p {metrics['With10pct']:.2f}%"
                )
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # === 保存模型与指标 ===
        self.model.load_state_dict(self.best_model_wts)
        model_path = os.path.join(save_root, f'fold{self.fold}_best_model.pth')
        torch.save(self.best_model_wts, model_path)
        logger.success(f"Best model saved: {model_path}")
        return self.best_metrics
