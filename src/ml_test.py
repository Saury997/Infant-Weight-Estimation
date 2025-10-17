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
from utils import set_seed, setup_logger, evaluate_regression, build_regressor
from main import load_config


def main():
    parser = argparse.ArgumentParser(description='Multi-Algorithm ML Baselines for REGRESSION')
    parser.add_argument('--config', type=str, default='../configs/ml.yaml',
                        help='Path to the ml.yaml configuration file.')
    args = parser.parse_args()

    # 1) 读取配置
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found '{args.config}'")
        return

    # 2) 创建本次运行的【总目录】
    set_seed(config.others.random_seed)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_ML-Multi-Regression"
    run_dir = os.path.join(config.others.save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 3) 设置【总日志】并备份配置
    log_file = os.path.join(run_dir, 'run.log')
    setup_logger(log_file)
    logger.info(f"Multi-algorithm baseline testing started. Results will be saved in: {run_dir}")


    cfg_copy_path = os.path.join(run_dir, 'config_backup.yaml')
    shutil.copyfile(args.config, cfg_copy_path)
    logger.info(f"Configuration file backed up to: {cfg_copy_path}")

    # 4) 数据加载与预处理（一次性完成）
    logger.info("Loading and preprocessing data...")
    X, y, y_bins, input_dim = load_and_preprocess_data(
        file_path=config.data.path,
        target_column=config.data.target_column,
        feat_eng=config.data.feature_engineering,
        binning=config.data.binning,
        log_transform=config.data.log_transform,
    )

    # 5) 划分最终测试集
    if config.data.binning and y_bins is not None:
        X_train_val, X_test, y_train_val, y_test, _, _ = train_test_split(
            X, y, y_bins, test_size=config.data.test_size,
            random_state=config.others.random_seed, stratify=y_bins
        )
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=config.data.test_size, random_state=config.others.random_seed
        )
    logger.info(f"Data loading completed. Train/Validation set: {len(X_train_val)}, Final test set: {len(X_test)}, Input dim: {input_dim}")

    # 6) 遍历算法列表，依次训练和评估
    all_cv_results = []
    all_test_results = []
    algos = config.ml.algorithms

    logger.info(f"Will run the following algorithms: {algos}")

    for algo in algos:
        logger.info(f"---------- Starting to process algorithm: {algo} ----------")

        # 为当前算法创建独立的子目录
        algo_dir = os.path.join(run_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

        # --- K-Fold CV ---
        if config.training.use_kfold:
            k = config.training.k_folds
            logger.info(f"Performing {k}-fold cross-validation for {algo}...")
            kf = KFold(n_splits=k, shuffle=True, random_state=config.others.random_seed)
            fold_metrics = []

            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train_val)):
                X_tr, X_va = X_train_val.iloc[tr_idx], X_train_val.iloc[va_idx]
                y_tr, y_va = y_train_val.iloc[tr_idx], y_train_val.iloc[va_idx]

                model = build_regressor(algo, config.ml, config.data.standardize)
                model.fit(X_tr, y_tr)
                y_va_pred = model.predict(X_va)

                y_va_true = np.exp(y_va.values) if config.data.log_transform else y_va.values
                if config.data.log_transform: y_va_pred = np.exp(y_va_pred)

                metrics = evaluate_regression(y_va_true, y_va_pred)
                fold_metrics.append(metrics)

            keys = list(fold_metrics[0].keys())
            avg = {f"CV_{k}": np.mean([m[k] for m in fold_metrics]) for k in keys}
            std = {f"CV_{k}_std": np.std([m[k] for m in fold_metrics]) for k in keys}
            cv_row = {"Algorithm": algo, **avg, **std}
            all_cv_results.append(cv_row)
            logger.success(f"[{algo} CV-Summary] -> " + ', '.join([f"{k}:{v:.4f}" for k, v in avg.items()]))

        # --- Final Train & Test ---
        logger.info(f"Training final {algo} model and evaluating on test set...")
        final_model = build_regressor(algo, config.ml, config.data.standardize)
        final_model.fit(X_train_val, y_train_val)

        # 保存模型
        model_path = os.path.join(algo_dir, 'model.joblib')
        joblib.dump(final_model, model_path)
        logger.info(f"Model saved to: {model_path}")

        y_test_pred = final_model.predict(X_test)

        y_tr_pred = final_model.predict(X_train_val)
        y_tr_true = np.exp(y_train_val.values) if config.data.log_transform else y_train_val.values
        y_test_true = np.exp(y_test.values) if config.data.log_transform else y_test.values
        if config.data.log_transform: y_test_pred = np.exp(y_test_pred)

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

        # 保存预测结果
        predictions_df = pd.DataFrame({'y_true': y_test_true, 'y_pred': y_test_pred})
        pred_path = os.path.join(algo_dir, 'test_predictions.csv')
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Test set predictions saved to: {pred_path}")

        test_metrics = evaluate_regression(y_test_true, y_test_pred)
        test_row = {"Algorithm": algo, **{f"Test_{k}": v for k, v in test_metrics.items()}}
        all_test_results.append(test_row)
        logger.success(f"[{algo} TEST] -> " + ', '.join([f"{k}:{v:.4f}" for k, v in test_metrics.items()]))
        logger.info(f"---------- Algorithm {algo} processing completed ----------\n")

    # 7) 保存【汇总结果】到总目录
    if all_cv_results:
        cv_summary_df = pd.DataFrame(all_cv_results)
        cv_summary_path = os.path.join(run_dir, '_summary_cv.csv')
        cv_summary_df.to_csv(cv_summary_path, index=False)
        logger.success(f"All algorithms cross-validation summary report saved: {cv_summary_path}")
        logger.opt(raw=True).info("\n--- Cross-validation Results Summary ---\n" + cv_summary_df.round(4).to_string(index=False) + "\n")

    if all_test_results:
        test_summary_df = pd.DataFrame(all_test_results)
        test_summary_path = os.path.join(run_dir, '_summary_test.csv')
        test_summary_df.to_csv(test_summary_path, index=False)
        logger.success(f"All algorithms final test summary report saved: {test_summary_path}")
        logger.opt(raw=True).info("\n--- Final Test Results Summary ---\n" + test_summary_df.round(4).to_string(index=False) + "\n")

    logger.success("All tasks completed!")


if __name__ == '__main__':
    main()
