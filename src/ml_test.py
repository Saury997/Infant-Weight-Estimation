#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/16
* Project: InfantWeight
* File: ml_test.py
* Function: Machine learning regression benchmark testing script for multiple algorithms.
* Usage: `python ml_test.py --config ../configs/ml.yaml`.
"""
import argparse
import os
import time
import shutil
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split

from data_loader import load_and_preprocess_data
from src.plot import result_plot
from utils import set_seed, setup_logger, evaluate_regression, build_regressor, single_metrics
from main import load_config


def main():
    parser = argparse.ArgumentParser(description='Multi-Algorithm ML Baselines for REGRESSION')
    parser.add_argument('--config', type=str, default='../configs/ml.yaml',
                        help='Path to the ml.yaml configuration file.')
    args = parser.parse_args()

    # --- 1. 读取配置 ---
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found '{args.config}'")
        return

    # --- 2. 创建本次运行的总目录 ---
    set_seed(config.others.random_seed)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    mode_tag = "KFold-CrossTest" if config.training.use_kfold else "Standard-Test"
    run_name = f"{timestamp}_ML-Multi-Regression_{mode_tag}"
    run_dir = os.path.join(config.others.save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- 3. 设置总日志并备份配置 ---
    log_file = os.path.join(run_dir, 'run.log')
    setup_logger(log_file)
    logger.info(f"Multi-algorithm baseline testing started. Results will be saved in: {run_dir}")
    logger.info(f"Selected Mode: {'K-Fold Cross-Testing' if config.training.use_kfold else 'Standard Train/Test'}")

    cfg_copy_path = os.path.join(run_dir, 'config_backup.yaml')
    shutil.copyfile(args.config, cfg_copy_path)
    logger.info(f"Configuration file backed up to: {cfg_copy_path}")

    # --- 4. 数据加载与预处理 ---
    logger.info("Loading and preprocessing data...")
    X, y, y_bins, input_dim = load_and_preprocess_data(
        file_path=config.data.path,
        target_column=config.data.target_column,
        feat_eng=config.data.feature_engineering,
        binning=config.data.binning,
        log_transform=config.data.log_transform,
    )
    logger.info(f"Data loading completed. Total samples: {len(X)}, Input dim: {input_dim}")

    algos = config.ml.algorithms
    logger.info(f"Will run the following algorithms: {algos}")

    # ----------------------------------------------------------------------
    # 流程 A: K-折交叉测试 (use_kfold = True)
    # ----------------------------------------------------------------------
    if config.training.use_kfold:
        all_cross_test_results = []
        k = config.training.k_folds
        logger.info(f"Starting {k}-fold cross-testing for all algorithms...")

        kfold_stratify = y_bins if config.data.binning and y_bins is not None else None
        kf = KFold(n_splits=k, shuffle=True, random_state=config.others.random_seed)

        for algo in algos:
            logger.info(f"---------- Starting {k}-fold Cross-Test for: {algo} ----------")
            algo_dir = os.path.join(run_dir, algo)  # 算法子目录
            os.makedirs(algo_dir, exist_ok=True)

            fold_metrics = []

            for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X, kfold_stratify)):
                X_tr, X_te_fold = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te_fold = y.iloc[tr_idx], y.iloc[te_idx]

                model = build_regressor(algo, config.ml, config.data.standardize)
                model.fit(X_tr, y_tr)
                y_te_pred = model.predict(X_te_fold)
                y_te_true = np.exp(y_te_fold.values) if config.data.log_transform else y_te_fold.values
                if config.data.log_transform: y_te_pred = np.exp(y_te_pred)

                metrics = evaluate_regression(y_te_true, y_te_pred)
                fold_metrics.append(metrics)

                model_path = os.path.join(algo_dir, f'model_fold_{fold_idx + 1}.joblib')
                joblib.dump(model, model_path)

            # 计算 K-折 交叉测试的平均值和标准差
            keys = list(fold_metrics[0].keys())
            avg = {f"{k}": np.mean([m[k] for m in fold_metrics]) for k in keys}
            std = {f"{k}_std": np.std([m[k] for m in fold_metrics]) for k in keys}

            ct_row = {"Algorithm": algo, **avg, **std}
            all_cross_test_results.append(ct_row)
            logger.success(f"[{algo} Cross-Test Summary] -> " + ', '.join(
                [f"{k}:{val:.2f}±{std_dev:.2f}" for k, val, std_dev in zip(avg.keys(), avg.values(), std.values())]))

        # 保存交叉测试汇总结果到总目录
        ct_summary_df = pd.DataFrame(all_cross_test_results)
        ct_summary_path = os.path.join(run_dir, '_summary_cross_test.csv')
        ct_summary_df.to_csv(ct_summary_path, index=False)
        logger.success(f"All algorithms cross-testing summary report saved: {ct_summary_path}")
        logger.opt(raw=True).info(
            "\n--- Cross-Testing Results Summary ---\n" + ct_summary_df.round(4).to_string(index=False) + "\n")

    # ----------------------------------------------------------------------
    # 流程 B: 标准训练/测试 (use_kfold = False)
    # ----------------------------------------------------------------------
    else:
        logger.info("Starting standard train/test split mode...")

        # --- 5. 划分最终测试集 ---
        if config.data.binning and y_bins is not None:
            X_train_val, X_test, y_train_val, y_test, _, _ = train_test_split(
                X, y, y_bins, test_size=config.data.test_size,
                random_state=config.others.random_seed, stratify=y_bins
            )
        else:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=config.data.test_size, random_state=config.others.random_seed
            )
        logger.info(f"Data split completed. Train set: {len(X_train_val)}, Test set: {len(X_test)}")

        all_test_results = []

        for algo in algos:
            logger.info(f"---------- Starting to process algorithm: {algo} ----------")
            algo_dir = os.path.join(run_dir, algo)
            os.makedirs(algo_dir, exist_ok=True)

            # --- Final Train & Test ---
            logger.info(f"Training final {algo} model and evaluating on test set...")
            final_model = build_regressor(algo, config.ml, config.data.standardize)
            final_model.fit(X_train_val, y_train_val)

            # 保存模型
            model_path = os.path.join(algo_dir, 'model.joblib')
            joblib.dump(final_model, model_path)
            logger.info(f"Model saved to: {model_path}")

            y_test_pred = final_model.predict(X_test)

            # 准备绘图和保存CSV所需的数据
            y_tr_pred = final_model.predict(X_train_val)
            y_tr_true = np.exp(y_train_val.values) if config.data.log_transform else y_train_val.values
            y_test_true = np.exp(y_test.values) if config.data.log_transform else y_test.values

            if config.data.log_transform:
                y_test_pred = np.exp(y_test_pred)
                y_tr_pred = np.exp(y_tr_pred)  # y_tr_pred 也需要反变换

            if config.others.plot:
                fig, _ = result_plot(
                    y_tr_true, y_tr_pred,
                    y_test_true, y_test_pred,
                    model_name=algo,
                    xlabel="True Values",
                    ylabel="Predicted Values",
                    panel_tag=None,
                    show_top_hist=True,
                    show_right_hist=True,
                    show_bottom_residual=True,
                    bins=20,
                    figsize=(2.6, 3.2), dpi=300
                )
                fig_path = os.path.join(algo_dir, f'result_visualization-{algo}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Visualization saved to: {fig_path}")

            test_metrics = evaluate_regression(y_test_true, y_test_pred)
            test_row = {"Algorithm": algo, **{f"Test_{k}": v for k, v in test_metrics.items()}}
            all_test_results.append(test_row)
            logger.success(f"[{algo} TEST] -> " + ', '.join([f"{k}:{v:.4f}" for k, v in test_metrics.items()]))

            # 计算并保存单个样本的评估指标
            sample_metrics_list = []
            for i in range(len(y_test_true)):
                sample_metrics = single_metrics(
                    np.array([y_test_true[i]]),
                    np.array([y_test_pred[i]])
                )
                sample_metrics_list.append(sample_metrics)

            predictions_df = pd.DataFrame({'y_true': y_test_true, 'y_pred': y_test_pred})
            sample_metrics_df = pd.DataFrame(sample_metrics_list)
            for col in sample_metrics_df.columns:
                predictions_df[f'sample_{col}'] = sample_metrics_df[col].values
            pred_path = os.path.join(algo_dir, 'test_pred.csv')
            predictions_df.to_csv(pred_path, index=False)
            logger.info(f"Test set predictions and sample metrics saved to: {pred_path}")
            logger.info(f"---------- Algorithm {algo} processing completed ----------\n")

        # 保存标准测试汇总结果到总目录
        test_summary_df = pd.DataFrame(all_test_results)
        test_summary_path = os.path.join(run_dir, '_summary_test.csv')
        test_summary_df.to_csv(test_summary_path, index=False)
        logger.success(f"All algorithms final test summary report saved: {test_summary_path}")
        logger.opt(raw=True).info(
            "\n--- Final Test Results Summary ---\n" + test_summary_df.round(4).to_string(index=False) + "\n")

    logger.success("All tasks completed!")


if __name__ == '__main__':
    main()