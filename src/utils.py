#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/7 14:59 
* Project: InfantWeight 
* File: utils.py
* IDE: PyCharm 
* Function:
"""
import os
import random
import sys
from types import SimpleNamespace

import pandas as pd
import torch.distributed as dist
import numpy as np
import torch
from loguru import logger
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW, SGD, LBFGS
from muon import MuonWithAuxAdam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_optimizer(model: torch.nn.Module, optimizer: str, lr: float) -> torch.optim.Optimizer:
    """根据参数初始化优化器"""
    if optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer == 'Muon':   # TODO: bugs need to fix.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group(backend='nccl', rank=0, world_size=1)
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        nonhidden_params = [*model.head.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=lr, weight_decay=0.01),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        logger.warning("The Muon optimizer has bugs to fix.")
    elif optimizer == 'LBFGS':
        # optimizer = LBFGS(model.parameters(), lr=lr)
        logger.warning("The effect of LBFGS optimizer is extremely poor, so it was disabled.")
        raise NotImplementedError("The effect of LBFGS optimizer is extremely poor, so it was disabled.")
    else:
        logger.error(f"Invalid optimizer: {optimizer}. Please choose from ['AdamW', 'SGD', 'Muon', 'LBFGS'].")
        raise ValueError("Invalid optimizer. Please choose from ['AdamW', 'SGD', 'Muon', 'LBFGS'].")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: SimpleNamespace):
    """根据参数初始化学习率调度器"""
    if scheduler_cfg.name == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_cfg.patience)
        logger.info(f"Using ReduceLROnPlateau scheduler with patience={scheduler_cfg.patience}.")
    elif scheduler_cfg.name == 'CosineAnnealingWarmRestarts':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        # T_0: 初始重启周期长度，T_mult: 周期倍增因子，eta_min: 最小学习率
        scheduler = CosineAnnealingWarmRestarts(optimizer, scheduler_cfg.T_0, T_mult=scheduler_cfg.T_mult, eta_min=scheduler_cfg.eta_min)
        logger.info(f"Using CosineAnnealingWarmRestarts scheduler with T_0={scheduler_cfg.T_0}, T_mult={scheduler_cfg.T_mult}, eta_min={scheduler_cfg.eta_min}.")
    elif scheduler_cfg.name == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        # step_size: 学习率衰减的步长，gamma: 衰减系数
        scheduler = StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
        logger.info(f"Using StepLR scheduler with step_size={scheduler_cfg.step_size}, gamma={scheduler_cfg.gamma}.")
    elif scheduler_cfg.name == 'MultiStepLR':
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=scheduler_cfg.milestones, gamma=scheduler_cfg.gamma)
        logger.info(f"Using MultiStepLR scheduler with milestones={scheduler_cfg.milestones}, gamma={scheduler_cfg.gamma}.")
    elif scheduler_cfg.name == 'ExponentialLR':
        from torch.optim.lr_scheduler import ExponentialLR
        # gamma: 指数衰减系数
        scheduler = ExponentialLR(optimizer, gamma=scheduler_cfg.gamma)
        logger.info(f"Using ExponentialLR scheduler with gamma={scheduler_cfg.gamma}.")
    elif scheduler_cfg.name == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # T_max: 最大迭代次数，eta_min: 最小学习率
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg.T_max, eta_min=scheduler_cfg.eta_min)
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={scheduler_cfg.T_max}, eta_min={scheduler_cfg.eta_min}.")
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_cfg.scheduler}. Please choose from ['ReduceLROnPlateau', "
                         f"'CosineAnnealingWarmRestarts', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR'].")
    return scheduler


def get_model(model_cfg: SimpleNamespace, input_dim: int) -> torch.nn.Module:
    """根据参数初始化模型"""
    params = vars(model_cfg.params)
    if model_cfg.name == 'MLP':
        from model.mlp import MLP
        model = MLP(input_dim=input_dim, **params)
    elif model_cfg.name == 'CNN':
        from model.cnn import CNNRegressor1D
        model = CNNRegressor1D(num_features=input_dim, **params)
    elif model_cfg.name == 'RNN':
        from model.rnn import RNNRegressor
        model = RNNRegressor(**params)
    elif model_cfg.name == 'LSTM':
        from model.lstm import LSTMRegressor
        model = LSTMRegressor(**params)
    elif model_cfg.name == 'KAN':
        from model.kan import KAN
        model = KAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'Fourier-KAN':
        from model.fourier_kan import FourierKAN
        model = FourierKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'DropKAN':
        from model.drop_kan import DropKAN
        model = DropKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'Wavelet-KAN':
        from model.wavelet_kan import WaveletKAN
        model = WaveletKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'Jacobi-KAN':
        from model.jacobi_kan import JacobiKAN
        model = JacobiKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'Taylor-KAN':
        from model.taylor_kan import TaylorKAN
        model = TaylorKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    elif model_cfg.name == 'Cheby-KAN':
        from model.cheby_kan import ChebyKAN
        model = ChebyKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1])
    elif model_cfg.name == 'F-KAN':
        from model.fkan import FKAN
        model = FKAN(layers_hidden=[input_dim] + model_cfg.hidden_layers + [1], **params)
    else:
        raise ValueError(f"Invalid model: {model_cfg.name}. Please choose from "
                         f"['MLP', 'KAN', 'Taylor-KAN', 'Fourier-KAN', 'Wavelet-KAN', 'Jacobi-KAN', 'Cheby-KAN'].")
    return model


def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_distribution(y_train: pd.Series, y_test: pd.Series, y_bins_train: pd.Series, y_bins_test: pd.Series) -> None:
    """
    检查训练集和测试集的分箱类别分布及统计信息。

    Args:
        y_train (pd.Series): 训练集真实目标值（连续数值，如出生体重）
        y_test (pd.Series): 测试集真实目标值
        y_bins_train (pd.Series): 训练集分箱类别
        y_bins_test (pd.Series): 测试集分箱类别
    """
    def stats_summary(y, y_bins, name="数据集"):
        df = pd.DataFrame({"value": y, "bin": y_bins})
        logger.info(f"\n{name} 样本总数: {len(y)}")
        logger.info(f"{name} 按类别统计：")
        summary = df.groupby("bin")["value"].agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            max="max"
        )
        summary["ratio"] = summary["count"] / len(y)
        logger.opt(raw=True).info(str(summary))  # 使用 raw=True 避免添加额外的日志格式前缀

    logger.info("==== 数据分布检查 ====")
    stats_summary(y_train, y_bins_train, "训练集")
    stats_summary(y_test, y_bins_test, "测试集")


def setup_logger(log_file_path: str) -> None:
    """初始化日志记录器"""
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <6}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file_path, rotation="10 MB", level="DEBUG")
    logger.info("Logger initialized.")


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归任务的各项评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def build_regressor(algo: str, cfg_ml, standardize:bool=False) -> Pipeline:
    """根据算法名称和配置构建一个 scikit-learn Pipeline"""
    steps = []

    if standardize or algo in ['KNN', 'SVR', 'Ridge', 'Lasso', 'ElasticNet']:
        steps.append(('scaler', StandardScaler()))

    algo = algo.strip()
    if algo == 'KNN':
        base = KNeighborsRegressor(**vars(cfg_ml.knn))
    elif algo == 'RandomForest':
        base = RandomForestRegressor(**vars(cfg_ml.random_forest))
    elif algo == 'ExtraTrees':
        base = ExtraTreesRegressor(**vars(cfg_ml.extra_trees))
    elif algo == 'XGBoost':
        base = XGBRegressor(**vars(cfg_ml.xgboost))
    elif algo == 'LightGBM':
        base = LGBMRegressor(**vars(cfg_ml.lightgbm))
    elif algo == 'CatBoost':
        base = CatBoostRegressor(**vars(cfg_ml.catboost))
    elif algo == 'Linear':
        base = LinearRegression(**vars(cfg_ml.linear))
    elif algo == 'Ridge':
        base = Ridge(**vars(cfg_ml.ridge))
    elif algo == 'Lasso':
        base = Lasso(**vars(cfg_ml.lasso))
    elif algo == 'ElasticNet':
        base = ElasticNet(**vars(cfg_ml.elasticnet))
    elif algo == 'SVR':
        base = SVR(**vars(cfg_ml.svr))
    elif algo == 'KernelRidge':
        base = KernelRidge(**vars(cfg_ml.kernel_ridge))
    elif algo == 'GBDT':
        base = GradientBoostingRegressor(**vars(cfg_ml.gbdt))
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    steps.append(('model', base))
    return Pipeline(steps)