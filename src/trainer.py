import os
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import json


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
    def evaluate(self, data_loader, save_root=None):
        """
        在验证集或测试集上评估模型，返回：
          - epoch_loss_log: log 空间 loss
          - epoch_mse_g: 原始 g 空间 MSE
          - epoch_mae_g: 原始 g 空间 MAE
          - metrics: 各类误差指标（整体 + 3800g以上 + 分体重段）
          - y_pred_orig: 原始单位的预测结果 tensor
        """
        self.model.eval()
        running_loss_log = 0.0
        running_mae = 0.0
        running_mse = 0.0
        all_outputs_orig, all_targets_orig = [], []

        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs_norm = self.model(inputs)

                # === 还原回原始体重(g) ===
                if self.log_transform:
                    outputs_orig = self._prepare_targets(outputs_norm, training=False)
                    targets_orig = torch.exp(targets) if self.targets_are_already_log else targets
                else:
                    outputs_orig = outputs_norm
                    targets_orig = targets

                all_outputs_orig.append(outputs_orig.detach().cpu())
                all_targets_orig.append(targets_orig.detach().cpu())

                # log 空间 loss
                if self.log_transform:
                    targets_norm = self._prepare_targets(targets, training=True)
                    loss_log = self.criterion(outputs_norm, targets_norm)
                else:
                    loss_log = self.criterion(outputs_orig, targets_orig)

                mae_g = torch.abs(outputs_orig - targets_orig).mean()
                mse_g = torch.pow(outputs_orig - targets_orig, 2).mean()

                running_loss_log += loss_log.item() * inputs.size(0)
                running_mae += mae_g.item() * inputs.size(0)
                running_mse += mse_g.item() * inputs.size(0)

        # === 汇总 ===
        epoch_loss_log = running_loss_log / len(data_loader.dataset)
        epoch_mae_g = running_mae / len(data_loader.dataset)
        epoch_mse_g = running_mse / len(data_loader.dataset)

        y_pred = torch.cat(all_outputs_orig, dim=0).numpy().flatten()
        y_true = torch.cat(all_targets_orig, dim=0).numpy().flatten()

        # =====================================================
        #               指标计算逻辑
        # =====================================================
        def within_pct(y_true, y_pred, pct):
            return float(np.mean(np.abs(y_pred - y_true) <= pct / 100.0 * y_true) * 100.0)

        # 全体样本指标
        relative_errors_pct = (y_pred - y_true) / (y_true + 1e-8) * 100
        systematic_error = float(np.mean(relative_errors_pct))
        random_error = float(np.std(relative_errors_pct, ddof=0))
        mape = float(np.mean(np.abs(relative_errors_pct)))
        mae_g = float(np.mean(np.abs(y_pred - y_true)))
        rmse_g = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        metrics = {
            "系统误差(%)": systematic_error,
            "随机误差(%)": random_error,
            "MAPE(%)": mape,
            "MAE(g)": mae_g,
            "RMSE(g)": rmse_g,
            "Within5%": within_pct(y_true, y_pred, 5),
            "Within10%": within_pct(y_true, y_pred, 10),
            "Within15%": within_pct(y_true, y_pred, 15),
            "样本数量": len(y_true)
        }

        # 3800g 以上样本指标
        idx_3800 = np.where(y_true >= 3800)[0]
        if len(idx_3800) > 0:
            sub_y_true = y_true[idx_3800]
            sub_y_pred = y_pred[idx_3800]
            rel_err_3800 = (sub_y_pred - sub_y_true) / (sub_y_true + 1e-8) * 100
            metrics[">=3800g"] = {
                "系统误差(%)": float(np.mean(rel_err_3800)),
                "随机误差(%)": float(np.std(rel_err_3800, ddof=0)),
                "MAPE(%)": float(np.mean(np.abs(rel_err_3800))),
                "MAE(g)": float(np.mean(np.abs(sub_y_pred - sub_y_true))),
                "RMSE(g)": float(np.sqrt(np.mean((sub_y_pred - sub_y_true) ** 2))),
                "Within5%": within_pct(sub_y_true, sub_y_pred, 5),
                "Within10%": within_pct(sub_y_true, sub_y_pred, 10),
                "Within15%": within_pct(sub_y_true, sub_y_pred, 15),
                "样本数量": len(sub_y_true)
            }
        else:
            metrics[">=3800g"] = None

        # 分体重段指标
        metrics["分体重段指标"] = {}
        bins = [(0, 2500), (2500, 3800), (3800, np.inf)]
        labels = ["小体重", "正常体重", "大体重"]
        for (lb, ub), label in zip(bins, labels):
            idx = np.where((y_true >= lb) & (y_true < ub))[0]
            if len(idx) > 0:
                sub_y_true = y_true[idx]
                sub_y_pred = y_pred[idx]
                rel_err = (sub_y_pred - sub_y_true) / (sub_y_true + 1e-8) * 100
                metrics["分体重段指标"][label] = {
                    "系统误差(%)": float(np.mean(rel_err)),
                    "随机误差(%)": float(np.std(rel_err, ddof=0)),
                    "MAPE(%)": float(np.mean(np.abs(rel_err))),
                    "MAE(g)": float(np.mean(np.abs(sub_y_pred - sub_y_true))),
                    "RMSE(g)": float(np.sqrt(np.mean((sub_y_pred - sub_y_true) ** 2))),
                    "样本数量": len(sub_y_true)
                }
            else:
                metrics["分体重段指标"][label] = None

        # 保存 metrics
        if save_root is not None:
            metrics_path = os.path.join(save_root, f"metrics_fold{self.fold}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)

        return epoch_loss_log, epoch_mse_g, epoch_mae_g, metrics, torch.tensor(y_pred)

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
                val_loss_log, val_loss_g, val_mae_g, metrics, _ = self.evaluate(val_loader, save_root=save_root)

                logger.info(
                    f"Fold {self.fold} | Epoch {epoch+1}/{self.epochs} [{elapsed:.2f}s] -> "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss_log:.6f}, "
                    f"MAE: {val_mae_g:.2f}g, 系统误差: {metrics['系统误差(%)']:.2f}%, 随机误差: {metrics['随机误差(%)']:.2f}%"
                )

                if self.writer:
                    prefix = f'Fold-{self.fold}/'
                    self.writer.add_scalar(f'{prefix}Loss/Train', train_loss, epoch)
                    self.writer.add_scalar(f'{prefix}Loss/Val', val_loss_log, epoch)
                    self.writer.add_scalar(f'{prefix}MAE/Val', val_mae_g, epoch)
                    self.writer.add_scalar(f'{prefix}LR', self.optimizer.param_groups[0]['lr'], epoch)

                scheduler.step(val_loss_log)

                if val_mae_g < self.best_val_mae:
                    self.best_val_mae, self.best_val_loss = val_mae_g, val_loss_log
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.best_metrics = metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.training_cfg.patience:
                    logger.warning(f"Early stopping at epoch {epoch+1}.")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} Train Loss: {train_loss:.6f}")
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # === 保存模型与指标 ===
        self.model.load_state_dict(self.best_model_wts)
        model_path = os.path.join(save_root, f'fold{self.fold}_best_model.pth')
        torch.save(self.best_model_wts, model_path)
        logger.success(f"✅ Best model saved: {model_path}")

        if save_root is not None and self.best_metrics is not None:
            metrics_path = os.path.join(save_root, f"best_metrics_fold{self.fold}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.best_metrics, f, ensure_ascii=False, indent=4)
            logger.success(f"✅ Metrics saved: {metrics_path}")

        return self.best_metrics
