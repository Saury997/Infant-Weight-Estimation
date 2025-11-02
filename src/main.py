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
import pandas as pd

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


def config_to_hparams_dict(config: SimpleNamespace) -> Dict[str, Any]:
    """
    将嵌套的 config (SimpleNamespace) 转换为扁平化的字典，用于 TensorBoard HParams。
    """
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
                if isinstance(v, list) or v is None:
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
        return dict(items)

    config_dict = config_to_dict(config)
    return flatten_dict(config_dict)


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
    set_seed(config.others.random_seed)

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

    mode_tag = "KFold-CrossTest" if config.training.use_kfold else "Standard-Test"
    run_name = f"{timestamp}_{config.model.name}_{layers_info}_{mode_tag}"
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
    logger.info(f"Using device: {config.others.device}")
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

    if config.training.use_kfold:
        logger.info(f"----- Starting {config.training.k_folds}-Fold Cross-Testing Mode -----")

        kfold_stratify = y_bins if config.data.binning else None
        kfold = KFold(n_splits=config.training.k_folds, shuffle=True, random_state=config.others.random_seed)

        all_fold_metrics = []

        for fold, (train_ids, test_ids_fold) in enumerate(kfold.split(X, kfold_stratify)):
            fold_num = fold + 1
            logger.info(f"----- FOLD {fold_num}/{config.training.k_folds} -----")

            # --- K-Fold 数据准备 ---
            X_train, X_test_fold = X.iloc[train_ids], X.iloc[test_ids_fold]
            y_train, y_test_fold = y.iloc[train_ids], y.iloc[test_ids_fold]

            if config.data.standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test_fold = scaler.transform(X_test_fold)
            else:
                X_train = X_train.values
                X_test_fold = X_test_fold.values

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
            test_dataset_fold = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32),
                                              torch.tensor(y_test_fold.values, dtype=torch.float32).view(-1, 1))

            train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=config.training.batch_size, shuffle=False)

            # --- 模型初始化与训练 ---
            model = get_model(config.model, input_dim).to(config.others.device)
            criterion = nn.MSELoss()
            optimizer = get_optimizer(model, config.training.optimizer, config.training.lr)
            scheduler = get_scheduler(optimizer, config.training.scheduler)

            trainer = Trainer(model, criterion, optimizer, config, writer, fold=fold_num)

            trainer.fit(train_loader, test_loader_fold, checkpoints_dir, scheduler)

            # --- 在该折的测试集上评估最佳模型 ---
            test_loss, metrics, _, _, _ = trainer.evaluate(test_loader_fold)
            all_fold_metrics.append(metrics)
            logger.success(
                f"Fold {fold + 1} Test Performance -> RMSE: {metrics['RMSE']:.2f}g, MAE: {metrics['MAE']:.2f}g, R2: {metrics['R2']:.2f}"
            )

        # --- K-Fold 交叉测试结果总结 ---
        avg_metrics = pd.DataFrame(all_fold_metrics).mean()
        std_metrics = pd.DataFrame(all_fold_metrics).std()

        logger.success("----- Average Cross-Test Performance -----")
        for key in avg_metrics.index:
            logger.success(f"Average {key}: {avg_metrics[key]:.2f} ± {std_metrics[key]:.2f}")

        # --- 记录超参数和平均测试结果 ---
        hparams_dict = config_to_hparams_dict(config)
        metrics_dict = {f"Metrics/Avg_Test_{key}": val for key, val in avg_metrics.items()}
        writer.add_hparams(hparams_dict, metrics_dict)
        logger.success("Hyperparameters and average test metrics saved to TensorBoard.")

        # ----------------------------------------------------------------------
        # 流程 B: 标准训练/测试 (use_kfold = False)
        # ----------------------------------------------------------------------
    else:
        logger.info("----- Starting Standard Train/Test Split Mode -----")

        # --- 3. 划分出最终的训练集和测试集 ---
        stratify_bins = y_bins if config.data.binning else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=config.data.test_size,
            random_state=config.others.random_seed,
            stratify=stratify_bins
        )
        if config.data.binning:
            y_bins_train, y_bins_test = train_test_split(y_bins, test_size=config.data.test_size,
                                                         random_state=config.others.random_seed, stratify=y_bins)
            check_distribution(y_train_val, y_test, y_bins_train, y_bins_test)

        logger.info(f"Data split. Samples for training: {len(X_train_val)}, Held-out test samples: {len(X_test)}")

        # --- 4. 训练最终模型 (使用所有训练/验证数据) ---
        logger.info("----- Training Final Model on All Training Data -----")

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

        final_model = get_model(config.model, input_dim).to(config.others.device)
        criterion = nn.MSELoss()
        optimizer = get_optimizer(final_model, config.training.optimizer, config.training.lr)
        scheduler = get_scheduler(optimizer, config.training.scheduler)

        final_trainer = Trainer(final_model, criterion, optimizer, config, writer, fold='Final')
        # val_loader 传入 None，表示不进行验证，只在训练集上训练
        final_trainer.fit(final_train_loader, None, checkpoints_dir, scheduler)

        # --- 5. 在最终测试集上评估 ---
        logger.info("----- Evaluating Final Model on Held-Out Test Set -----")
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        test_loss, metrics, _, sample_metrics, y_pred_test = final_trainer.evaluate(test_loader)
        logger.success("----- Final Test Set Performance -----")
        for key, value in metrics.items():
            logger.success(f"Test {key}: {value:.2f}")

        # --- 6. 结果可视化 ---
        if config.others.plot:
            logger.info("Generating result plots...")
            final_model.eval()
            with torch.no_grad():
                y_pred_train_norm = final_model(
                    torch.tensor(X_train_val_scaled, dtype=torch.float32, device=config.others.device))
                y_pred_train_orig = final_trainer._prepare_targets(y_pred_train_norm, training=False).detach().cpu()

            fig, _ = result_plot(y_train_val, y_pred_train_orig, y_test, y_pred_test.to('cpu'),
                                 model_name=config.model.name)
            fig_path = os.path.join(run_dir, f'result_visualization-{config.model.name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Result visualization saved to: {fig_path}")

        # --- 7. 保存训练集和测试集预测结果到CSV ---
        logger.info("Saving training and test set predictions to CSV files...")

        train_res_df = X_train_val.copy()
        train_res_df[config.data.target_column + '_True'] = y_train_val
        train_res_df[config.data.target_column + '_Predicted'] = y_pred_train_orig.numpy().flatten()
        train_pred_path = os.path.join(run_dir, 'train_pred.csv')
        train_res_df.to_csv(train_pred_path, index=False)
        logger.success(f"Train set predictions saved to: {train_pred_path}")

        test_res_df = X_test.copy()
        test_res_df[config.data.target_column + '_True'] = y_test
        test_res_df[config.data.target_column + '_Predicted'] = y_pred_test.numpy().flatten()
        sample_metrics_df = pd.DataFrame(sample_metrics)
        for col in sample_metrics_df.columns:
            test_res_df[f'sample_{col}'] = sample_metrics_df[col].values
        test_pred_path = os.path.join(run_dir, 'test_pred.csv')
        test_res_df.to_csv(test_pred_path, index=False)
        logger.success(f"Test set predictions saved to: {test_pred_path}")

        # --- 8. 记录超参数和单次测试结果 ---
        hparams_dict = config_to_hparams_dict(config)
        metrics_dict = {f"Metrics/Test_{key}": val for key, val in metrics.items()}
        writer.add_hparams(hparams_dict, metrics_dict)
        logger.success("Hyperparameters and test metrics saved to TensorBoard.")

        # --- 流程结束 ---
        writer.close()
        logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
