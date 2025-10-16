#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/16 20:50 
* Project: Infant-Weight-Estimation 
* File: ml_search.py
* IDE: PyCharm 
* Function:
"""
import argparse
import os
import time
import json
import warnings
import joblib
from types import SimpleNamespace
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_data
from utils import set_seed, setup_logger, build_regressor, evaluate_regression  # 复用统一模型工厂
from main import load_config

warnings.filterwarnings("ignore")


def _make_run_dirs(save_root: str, algo_list: List[str]) -> Tuple[str, str, str, str, str]:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_HP-Search_{'-'.join(algo_list)}"
    run_dir = os.path.join(save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'run.log')
    results_dir = os.path.join(run_dir, 'results')
    models_dir = os.path.join(run_dir, 'best_models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    cfg_copy_path = os.path.join(run_dir, 'ml_search.yaml')
    return run_dir, log_file_path, results_dir, models_dir, cfg_copy_path


ALGO_SECTION_KEY = {
    'KNN': 'knn',
    'RandomForest': 'random_forest',
    'ExtraTrees': 'extra_trees',
    'SVR': 'svr',
    'Ridge': 'ridge',
    'Lasso': 'lasso',
    'ElasticNet': 'elasticnet',
    'KernelRidge': 'kernel_ridge',
    'GBDT': 'gbdt',
    'Linear': 'linear',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm',
    'CatBoost': 'catboost',
}


def _make_cfg_ml_for_algo(algo: str, base_params: Dict[str, Any]) -> SimpleNamespace:
    """为 utils.build_regressor 构造只包含当前算法参数的 cfg_ml 命名空间。
    例如 algo='KNN' -> cfg_ml.knn = SimpleNamespace(**base_params)
    """
    sec = ALGO_SECTION_KEY[algo]
    def _to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
        # 递归把 dict 转 SimpleNamespace，保证与 utils.build_regressor 的 vars(cfg_ml.xxx) 兼容
        ns = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ns[k] = _to_namespace(v)
            else:
                ns[k] = v
        return SimpleNamespace(**ns)
    return SimpleNamespace(**{sec: _to_namespace(base_params)})


def _needs_scaler(algo_name: str, standardize_flag: bool) -> bool:
    # 与 utils.build_regressor 一致的标准化策略（KNN/SVR/Ridge/Lasso/ElasticNet 强制）
    force = {'KNN', 'SVR', 'Ridge', 'Lasso', 'ElasticNet'}
    return standardize_flag or (algo_name in force)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Multiple ML Regressors (reuse utils.build_regressor)')
    parser.add_argument('--config', type=str, default='../configs/ml_search.yaml',
                        help='配置文件，建议单独为搜索准备（例如 ../configs/ml_search.yaml）')
    args = parser.parse_args()

    # 读取配置
    config = load_config(args.config)
    set_seed(config.others.random_seed)

    algo_list: List[str] = list(config.ml_search.algorithms)
    run_dir, log_file, results_dir, models_dir, cfg_copy_path = _make_run_dirs(config.others.save_root, algo_list)
    setup_logger(log_file)
    logger.info(f"Hyperparameter Search Run Dir: {run_dir}")

    # 备份配置
    with open(args.config, 'r', encoding='utf-8') as f_in, open(cfg_copy_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())

    # 加载数据
    logger.info("Loading and preprocessing all data ...")
    X, y, y_bins, input_dim = load_and_preprocess_data(
        file_path=config.data.path,
        target_column=config.data.target_column,
        feat_eng=config.data.feature_engineering,
        binning=config.data.binning,
        log_transform=config.data.log_transform,
    )

    # 划分测试集
    if config.data.binning and (y_bins is not None):
        X_train_val, X_test, y_train_val, y_test, _, _ = train_test_split(
            X, y, y_bins,
            test_size=config.data.test_size,
            random_state=config.others.random_seed,
            stratify=y_bins
        )
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=config.data.test_size,
            random_state=config.others.random_seed
        )

    logger.success(f"Data loaded. Input dim: {input_dim}, Train/Val: {len(X_train_val)}, Test: {len(X_test)}")

    # 搜索器全局设置
    scoring = config.ml_search.scoring           # 例如 'neg_root_mean_squared_error'
    mode = config.ml_search.mode                 # 'grid' or 'random'
    search_n_jobs = config.ml_search.search_n_jobs
    refit_flag = config.ml_search.refit
    return_train_score = config.ml_search.return_train_score
    try:
        random_iter = config.ml_search.random_iter
    except AttributeError:
        random_iter = None

    kf = KFold(n_splits=config.training.k_folds, shuffle=True, random_state=config.others.random_seed)

    # 动态读取每个算法的搜索空间（避免 getattr，使用 vars）
    ml_search_dict = vars(config.ml_search)

    summary_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for algo in algo_list:
        logger.info("=" * 80)
        logger.info(f"[Search Algo] {algo}")
        sec_key = ALGO_SECTION_KEY[algo]
        algo_cfg = ml_search_dict[sec_key]  # SimpleNamespace

        # base_params & search_space 都是 dict-like（或 SimpleNamespace）
        bp = algo_cfg.base_params
        ss = algo_cfg.search_space
        base_params = dict(vars(bp)) if isinstance(bp, SimpleNamespace) else dict(bp)
        search_space = dict(vars(ss)) if isinstance(ss, SimpleNamespace) else dict(ss)

        cfg_ml = _make_cfg_ml_for_algo(algo, base_params)
        pipe: Pipeline = build_regressor(algo, cfg_ml, standardize=config.data.standardize)

        if _needs_scaler(algo, config.data.standardize) and 'scaler' not in dict(pipe.named_steps):
            pipe = Pipeline([('scaler', StandardScaler())] + list(pipe.steps))

        param_grid = {f"model__{k}": v for k, v in search_space.items()}

        # 选择搜索器
        if mode == 'grid':
            searcher = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=kf,
                n_jobs=search_n_jobs,
                refit=refit_flag,
                return_train_score=return_train_score,
                verbose=0,
            )
        else:
            if random_iter is None:
                raise ValueError("ml_search.mode == 'random' 时必须提供 ml_search.random_iter")
            searcher = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                n_iter=random_iter,
                scoring=scoring,
                cv=kf,
                n_jobs=search_n_jobs,
                refit=refit_flag,
                return_train_score=return_train_score,
                random_state=config.others.random_seed,
                verbose=0,
            )

        # 执行搜索
        t0 = time.time()
        searcher.fit(X_train_val, y_train_val)
        search_time = time.time() - t0

        # 收集最优结果
        best_score = float(searcher.best_score_)
        best_params_prefixed = dict(searcher.best_params_)

        best_params = {k.split('model__', 1)[1] if k.startswith('model__') else k: v for k, v in best_params_prefixed.items()}

        # 另存 cv_results_
        cv_results_df = pd.DataFrame(searcher.cv_results_)
        cv_path = os.path.join(results_dir, f'{algo}_cv_results.csv')
        cv_results_df.to_csv(cv_path, index=False)
        logger.success(f"{algo}: best_cv_score={best_score:.6f}, time={search_time:.1f}s, saved cv_results -> {cv_path}")

        # 在训练+验证集上重训最佳模型，并在测试集评估
        best_model = searcher.best_estimator_
        best_model.fit(X_train_val, y_train_val)
        y_pred_test = best_model.predict(X_test)
        if config.data.log_transform:
            y_pred_test = np.exp(y_pred_test)
            y_test_true = np.exp(y_test.values)
        else:
            y_test_true = y_test.values
        metrics = evaluate_regression(y_test_true, y_pred_test)

        # 记录汇总
        summary_rows.append({
            'Algo': algo,
            'BestCV': best_score,
            'SearchTimeSec': round(search_time, 2),
            'BestParams': json.dumps(best_params, ensure_ascii=False)
        })
        tr = {'Algo': algo}
        tr.update({f'Test_{k}': float(v) for k, v in metrics.items()})
        test_rows.append(tr)

        # 保存最佳模型（可选）
        if config.ml_search.save_best_model:
            model_path = os.path.join(models_dir, f'{algo}_best.joblib')
            joblib.dump(best_model, model_path)
            logger.info(f"Saved best model -> {model_path}")

    # 保存汇总 CSV
    best_summary_df = pd.DataFrame(summary_rows)
    best_summary_path = os.path.join(results_dir, 'best_params_summary.csv')
    best_summary_df.to_csv(best_summary_path, index=False)

    test_summary_df = pd.DataFrame(test_rows)
    test_summary_path = os.path.join(results_dir, 'best_models_test_metrics.csv')
    test_summary_df.to_csv(test_summary_path, index=False)

    logger.success(f"Best params summary saved: {best_summary_path}")
    logger.success(f"Test metrics summary saved: {test_summary_path}")

    # meta
    meta = {
        'algorithms': algo_list,
        'mode': config.ml_search.mode,
        'scoring': scoring,
        'k_folds': config.training.k_folds,
        'results': {
            'best_params_summary_csv': os.path.abspath(best_summary_path),
            'best_models_test_metrics_csv': os.path.abspath(test_summary_path)
        }
    }
    meta_path = os.path.join(results_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.success(f"Meta saved: {meta_path}")


if __name__ == '__main__':
    main()
