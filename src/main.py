#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/09
* Project: InfantWeight
* File: main.py
* Function: Main training script for fetal weight prediction models.
  Supports multiple model architectures (MLP, KAN, Fourier-KAN, etc.) and training modes (k-fold cross-validation or simple validation).
  Handles data loading, preprocessing, model training, evaluation, and result logging with TensorBoard support.
"""
import os
import time
from types import SimpleNamespace
import warnings

import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np
import argparse

from data_loader import load_and_preprocess_data
from trainer import Trainer
from utils import get_optimizer, set_seed, get_model, check_distribution, setup_logger, get_scheduler

warnings.filterwarnings("ignore")


def load_config(config_path):
    """
    加载并解析 YAML 配置文件.
    将嵌套的字典转换为可以通过点操作符访问的命名空间对象.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if not isinstance(d, dict):
            return d
        namespace = SimpleNamespace()
        for key, value in d.items():
            attr_name = key.replace('-', '_')
            namespace.__setattr__(attr_name, dict_to_namespace(value))
        return namespace

    return dict_to_namespace(config_dict)


def main():
    parser = argparse.ArgumentParser(description='Fetal Weight Prediction Model Training')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to the YAML configuration file. Default: config.yaml')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    if config.others.device == 'auto':
        config.others.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {config.others.device}")
    set_seed(config.others.random_seed)
    if not os.path.exists(config.others.save_root):
        os.makedirs(config.others.save_root)

    # 日志文件保存路径
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_{config.model.name}-{config.model.hidden_layers}_lr-{config.training.lr}_bs-{config.training.batch_size}"
    run_dir = os.path.join(config.others.save_root, run_name)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    log_file_path = os.path.join(run_dir, 'run.log')
    config_file_path = os.path.join(run_dir, 'config.yaml')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    setup_logger(log_file_path)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    logger.info(f"Experiment run directory: {run_dir}")

    # 超参数保存
    with open(args.config, 'r', encoding='utf-8') as f_in, open(config_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())
    logger.info(f"Configuration for this run saved to {config_file_path}")
    logger.info(f"Experiments are conducted on {config.model.name}.")

    # --- 1. 数据加载和预处理 ---
    logger.info("Loading and preprocessing all data...")
    X, y, y_bins, input_dim = load_and_preprocess_data(
        file_path=config.data.path,
        target_column=config.data.target_column,
        feat_eng=config.data.feature_engineering,
        binning=config.data.binning,
        log_transform=config.data.log_transform
    )

    # --- 2. 划分出最终的测试集 ---
    if config.data.binning:
        X_train_val, X_test, y_train_val, y_test, y_bins_train, y_bins_test = train_test_split(
            X, y, y_bins,
            test_size=config.data.test_size,
            random_state=config.others.random_seed,
            stratify=y_bins
        )
        check_distribution(y_train_val, y_test, y_bins_train, y_bins_test)
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=config.data.test_size,
            random_state=config.others.random_seed
        )
    logger.info(f"Data loaded. Input dimension: {input_dim}")
    logger.info(f"Samples for training/validation: {len(X_train_val)}, Held-out test samples: {len(X_test)}")

    if X is None:
        raise ValueError("DataLoadError! Please check your data file path.")

    # --- 3. 根据参数选择执行K-折交叉验证或简单验证 ---
    if config.training.use_kfold:
        # --- K-Fold Cross-Validation ---
        logger.info(f"----- Starting {config.training.k_folds}-Fold Cross-Validation -----")
        kfold = KFold(n_splits=config.training.k_folds, shuffle=True, random_state=config.others.random_seed)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train_val)):
            fold_num = fold + 1
            logger.info(f"----- FOLD {fold_num}/{config.training.k_folds} -----")

            # --- 数据准备 ---
            X_train, X_val = X_train_val.iloc[train_ids], X_train_val.iloc[val_ids]
            y_train, y_val = y_train_val.iloc[train_ids], y_train_val.iloc[val_ids]

            if config.data.standardize:
                final_scaler = StandardScaler()
                X_train = final_scaler.fit_transform(X_train)
                X_val = final_scaler.transform(X_val)
            else:
                X_train = X_train.values
                X_val = X_val.values

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1))

            train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

            # --- 模型初始化 ---
            model = get_model(config.model, input_dim).to(config.others.device)
            criterion = nn.MSELoss()
            optimizer = get_optimizer(model, config.training.optimizer, config.training.lr)
            scheduler = get_scheduler(optimizer, config.training.scheduler)

            # --- 训练 ---
            trainer = Trainer(model, criterion, optimizer, config, writer, fold=fold_num)
            trainer.fit(train_loader, val_loader, checkpoints_dir, scheduler)

            fold_results.append(trainer.best_val_mae)
            logger.success(f"Fold {fold + 1} Best Validation MAE: {trainer.best_val_mae:.2f}g")

        # --- 打印交叉验证总结 ---
        avg_mae = np.mean(fold_results)
        std_mae = np.std(fold_results)
        logger.success(f"K-Fold CV Summary: Average Validation MAE = {avg_mae:.2f}g ± {std_mae:.2f}g")

        # 将 config 对象转为字典以记录超参数
        def config_to_dict(cfg):
            if not isinstance(cfg, SimpleNamespace):
                return cfg
            return {key: config_to_dict(value) for key, value in cfg.__dict__.items()}

        # 展平嵌套的配置字典
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v) if isinstance(v, list) else v))
            return dict(items)

        hparams_dict = flatten_dict(config_to_dict(config))
        metrics_dict = {"Metrics/CV_Avg_MAE": float(avg_mae), "Metrics/CV_Std_MAE": float(std_mae)}
        writer.add_hparams(hparams_dict, metrics_dict)
        logger.success("Hyperparameters and metrics saved to TensorBoard.")

    # --- 4. 训练最终模型 ---
    logger.info("----- Training Final Model on All Training Data -----")

    # --- 数据准备 ---
    if config.data.standardize:
        final_scaler = StandardScaler()
        X_train_val = final_scaler.fit_transform(X_train_val)
        X_test = final_scaler.transform(X_test)
    else:
        X_train_val = X_train_val.values

    y_train_val_tensor = torch.tensor(y_train_val.values, dtype=torch.float32).view(-1, 1)

    final_train_dataset = TensorDataset(torch.tensor(X_train_val, dtype=torch.float32), y_train_val_tensor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=config.training.batch_size, shuffle=True)

    # --- 模型初始化与训练 ---
    final_model = get_model(config.model, input_dim).to(config.others.device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(final_model, config.training.optimizer, config.training.lr)
    scheduler = get_scheduler(optimizer, config.training.scheduler)

    final_trainer = Trainer(final_model, criterion, optimizer, config, writer, fold='Final')
    final_trainer.fit(final_train_loader, None, checkpoints_dir, scheduler)

    # --- 5. 在最终测试集上评估 ---
    logger.info("----- Evaluating Final Model on Held-Out Test Set -----")
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    test_loss, test_mae = final_trainer.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)

    logger.success(f"Final Test Set Performance -> MSE: {test_loss:.4f}, RMSE: {test_rmse:.2f}g, MAE: {test_mae:.2f}g")
    writer.close()


if __name__ == "__main__":
    main()