#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang & Lanxiang Ma
* Date: 2025/10/09
* Project: InfantWeight
* File: main.py
* Function: Main training script for fetal weight prediction models.
  Supports multiple model architectures (MLP, KAN, Fourier-KAN, etc.) and training modes (k-fold cross-validation or simple validation).
  Handles data loading, preprocessing, model training, evaluation, and result logging with TensorBoard support.
  This script orchestrates the entire machine learning pipeline, from configuration loading and data preparation
  to model training, hyperparameter tuning (implicitly via k-fold), final evaluation on a held-out test set,
  and visualization of results.
"""
import os
import time
from types import SimpleNamespace
import warnings
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np
import argparse
import pandas as pd  # <-- 新增导入

from data_loader import load_and_preprocess_data
from trainer import Trainer
from utils import get_optimizer, set_seed, get_model, check_distribution, setup_logger, get_scheduler
from plot import result_plot

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> SimpleNamespace:
    """
    加载并解析 YAML 配置文件。
    将嵌套的字典结构转换为可以通过点操作符访问属性的 `SimpleNamespace` 对象。
    这使得配置参数的访问更加方便，例如 `config.training.epochs`。

    Args:
        config_path (str): YAML 配置文件的路径。

    Returns:
        SimpleNamespace: 包含所有配置参数的命名空间对象。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if not isinstance(d, dict):
            return d
        namespace = SimpleNamespace()
        for key, value in d.items():
            attr_name = key.replace('-', '_')  # 替换 '-' 为 '_'，以便点操作符访问
            namespace.__setattr__(attr_name, dict_to_namespace(value))
        return namespace

    return dict_to_namespace(config_dict)


def main():
    """
    主训练和评估函数。

    该函数执行以下主要步骤：
    1. 解析命令行参数，加载 YAML 配置文件。
    2. 初始化设备、随机种子、创建实验目录并设置日志。
    3. 加载并预处理数据，包括特征工程、对数变换等。
    4. 将数据划分为训练/验证集和独立的测试集。如果配置了分箱，则进行分层抽样。
    5. 根据配置选择训练模式：
        - 如果 `config.training.use_kfold` 为 True，则执行 K-折交叉验证。
          在每个折叠中，模型会进行训练和验证，并记录验证集上的性能。
        - 否则，将跳过 K-折验证（但仍会进行后续的最终模型训练和测试）。
    6. 使用所有训练/验证数据训练一个最终模型。
       该模型会保存最佳权重。
    7. 在独立的测试集上评估最终模型的性能，并计算详细的误差指标。
    8. 可选地生成结果可视化图表（例如，预测值 vs 真实值散点图，Bland-Altman 图）并保存。
    9. 将超参数和交叉验证结果（如果进行了）记录到 TensorBoard。
    10. 将训练集和测试集的原始特征、真实值和预测值保存为 CSV 文件。
    """
    parser = argparse.ArgumentParser(description='Fetal Weight Prediction Model Training')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to the YAML configuration file. Default: config.yaml')
    args = parser.parse_args()

    # --- 1. 加载配置并初始化环境 ---
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found at '{args.config}'")
        return

    if config.others.device == 'auto':
        config.others.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {config.others.device}")
    set_seed(config.others.random_seed)  # 设置随机种子以保证可复现性

    if not os.path.exists(config.others.save_root):
        os.makedirs(config.others.save_root)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    layers_info = 'default'
    if hasattr(config.model, 'hidden_layers'):
        layers_info = config.model.hidden_layers
    elif hasattr(config.model, 'params') and hasattr(config.model.params, 'conv_channels'):
        layers_info = config.model.params.conv_channels
    elif hasattr(config.model, 'params') and hasattr(config.model.params, 'num_neurons'):
        layers_info = config.model.params.num_neurons

    run_name = f"{timestamp}_{config.model.name}-{layers_info}_lr-{config.training.lr}_bs-{config.training.batch_size}"
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

    # 保存当前配置文件的副本
    with open(args.config, 'r', encoding='utf-8') as f_in, open(config_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())
    logger.info(f"Configuration for this run saved to {config_file_path}")
    logger.info(f"Experiments are conducted on {config.model.name}.")

    # --- 2. 数据加载和预处理 ---
    logger.info("Loading and preprocessing all data...")
    X, y, y_bins, input_dim = load_and_preprocess_data(
        file_path=config.data.path,
        target_column=config.data.target_column,
        feat_eng=config.data.feature_engineering,
        binning=config.data.binning,
        log_transform=config.data.log_transform
    )

    # --- 3. 划分出最终的测试集 ---
    if config.data.binning:
        # 如果需要分箱（通常用于处理目标变量分布不均的情况），则进行分层抽样
        X_train_val, X_test, y_train_val, y_test, y_bins_train, y_bins_test = train_test_split(
            X, y, y_bins,
            test_size=config.data.test_size,
            random_state=config.others.random_seed,
            stratify=y_bins  # 依据分箱结果进行分层抽样
        )
        check_distribution(y_train_val, y_test, y_bins_train, y_bins_test)  # 检查分层抽样后的分布
    else:
        # 否则进行普通随机抽样
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=config.data.test_size,
            random_state=config.others.random_seed
        )
    logger.info(f"Data loaded. Input dimension: {input_dim}")
    logger.info(f"Samples for training/validation: {len(X_train_val)}, Held-out test samples: {len(X_test)}")

    if X is None:
        raise ValueError("DataLoadError! Please check your data file path.")

    # --- 4. 根据参数选择执行 K-折交叉验证 ---
    if config.training.use_kfold:
        logger.info(f"----- Starting {config.training.k_folds}-Fold Cross-Validation -----")
        kfold = KFold(n_splits=config.training.k_folds, shuffle=True, random_state=config.others.random_seed)
        fold_results = []  # 存储每折的验证 MAE

        for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train_val)):
            fold_num = fold + 1
            logger.info(f"----- FOLD {fold_num}/{config.training.k_folds} -----")

            # --- K-Fold 数据准备 ---
            X_train, X_val = X_train_val.iloc[train_ids], X_train_val.iloc[val_ids]
            y_train, y_val = y_train_val.iloc[train_ids], y_train_val.iloc[val_ids]

            # 数据标准化
            if config.data.standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            else:
                X_train = X_train.values
                X_val = X_val.values

            # 转换为 TensorDataset 和 DataLoader
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1))

            train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

            # --- 模型初始化 ---
            model = get_model(config.model, input_dim).to(config.others.device)
            criterion = nn.MSELoss()  # 使用均方误差作为损失函数
            optimizer = get_optimizer(model, config.training.optimizer, config.training.lr)
            scheduler = get_scheduler(optimizer, config.training.scheduler)

            # --- 训练 ---
            trainer = Trainer(model, criterion, optimizer, config, writer, fold=fold_num)
            trainer.fit(train_loader, val_loader, checkpoints_dir, scheduler)

            fold_results.append(trainer.best_val_mae)
            logger.success(f"Fold {fold + 1} Best Validation MAE: {trainer.best_val_mae:.2f}g")

        # --- K-Fold 结果总结 ---
        avg_mae = np.mean(fold_results)
        std_mae = np.std(fold_results)
        logger.success(f"K-Fold CV Summary: Average Validation MAE = {avg_mae:.2f}g ± {std_mae:.2f}g")

        # 将 config 对象转为字典以记录超参数到 TensorBoard
        def config_to_dict(cfg: Any) -> Dict[str, Any]:
            if not isinstance(cfg, SimpleNamespace):
                return cfg
            return {key: config_to_dict(value) for key, value in cfg.__dict__.items()}

        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
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

    # --- 5. 训练最终模型 (使用所有训练/验证数据) ---
    logger.info("----- Training Final Model on All Training Data -----")

    # --- 最终模型数据准备 ---
    if config.data.standardize:
        final_scaler = StandardScaler()
        X_train_val_scaled = final_scaler.fit_transform(X_train_val)
        X_test_scaled = final_scaler.transform(X_test)
    else:
        X_train_val_scaled = X_train_val.values
        X_test_scaled = X_test.values

    y_train_val_tensor = torch.tensor(y_train_val.values, dtype=torch.float32).view(-1, 1)

    final_train_dataset = TensorDataset(torch.tensor(X_train_val_scaled, dtype=torch.float32), y_train_val_tensor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=config.training.batch_size, shuffle=True)

    # --- 最终模型初始化与训练 ---
    final_model = get_model(config.model, input_dim).to(config.others.device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(final_model, config.training.optimizer, config.training.lr)
    scheduler = get_scheduler(optimizer, config.training.scheduler)

    # 在所有训练数据上训练，不使用验证集 (或使用所有数据作为训练集)
    final_trainer = Trainer(final_model, criterion, optimizer, config, writer, fold='Final')
    final_trainer.fit(final_train_loader, None, checkpoints_dir, scheduler)  # val_loader 传入 None 表示不进行验证

    # --- 6. 在最终测试集上评估 ---
    logger.info("----- Evaluating Final Model on Held-Out Test Set -----")
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    test_loss, metrics, _, y_pred_test = final_trainer.evaluate(test_loader)
    logger.success(
        f"Final Test Set Performance -> RMSE: {metrics['RMSE']:.2f}g, MAE: {metrics['MAE']:.2f}g, "
        f"R2: {metrics['R2']:.2f}, MAPE: {metrics['MAPE']:.2f}, SystematicError: {metrics['SystematicError']:.2f}, "
        f"RandomError: {metrics['RandomError']:.2f}%, With10pct: {metrics['With10pct']:.2f}%"
    )

    # --- 7. 结果可视化 ---
    if config.others.plot:
        logger.info("Generating result plots...")
        final_model.eval()
        with torch.no_grad():
            y_pred_train_norm = final_model(
                torch.tensor(X_train_val_scaled, dtype=torch.float32, device=config.others.device))
            # 将训练集预测值反变换回原始单位
            y_pred_train_orig = final_trainer._prepare_targets(y_pred_train_norm, training=False).detach().cpu()

        # 散点图
        fig, _ = result_plot(y_train_val, y_pred_train_orig, y_test, y_pred_test.to('cpu'),
                             model_name=config.model.name)
        fig_path = os.path.join(run_dir, f'result_visualization-{config.model.name}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Result visualization saved to: {fig_path}")

    # --- 8. 保存训练集和测试集预测结果到CSV ---
    logger.info("Saving training and test set predictions to CSV files...")

    train_results_df = X_train_val.copy()
    train_results_df[config.data.target_column + '_True'] = y_train_val
    train_results_df[config.data.target_column + '_Predicted'] = y_pred_train_orig.numpy().flatten()
    train_predictions_path = os.path.join(run_dir, 'train_predictions.csv')
    train_results_df.to_csv(train_predictions_path, index=True)
    logger.success(f"Train set predictions saved to: {train_predictions_path}")

    test_results_df = X_test.copy()
    test_results_df[config.data.target_column + '_True'] = y_test
    test_results_df[config.data.target_column + '_Predicted'] = y_pred_test.numpy().flatten()
    test_predictions_path = os.path.join(run_dir, 'test_predictions.csv')
    test_results_df.to_csv(test_predictions_path, index=True)
    logger.success(f"Test set predictions saved to: {test_predictions_path}")

    writer.close()


if __name__ == "__main__":
    main()
