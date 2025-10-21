#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang & Lanxiang Ma
* Date: 2025/10/7 13:04 
* Project: InfantWeight 
* File: data_loader.py
* IDE: PyCharm 
* Function: Data loading and preprocessing module for fetal weight prediction.
  Provides feature engineering capabilities and data preparation functions including target variable binning and log transformation.
"""
import pandas as pd
import numpy as np
from loguru import logger


def feature_engineering(df):
    """
    对数据集进行详尽的特征工程。

    Args:
        df (pd.DataFrame): 原始数据框。

    Returns:
        pd.DataFrame: 经过特征工程后的数据框。
    """
    logger.info("Performing advanced feature engineering...")

    # --- I. 创建生理学综合指标 ---
    height_m = df['孕妇身高'] / 100
    height_m[height_m == 0] = np.nan
    df['孕妇BMI'] = df['孕妇入院体重'] / (height_m ** 2)

    # --- II. 创建比例特征 ---
    df['HC_AC_Ratio'] = df['头围'] / df['腹围'].replace(0, np.nan)
    df['FL_AC_Ratio'] = df['股骨长'] / df['腹围'].replace(0, np.nan)
    df['BPD_FL_Ratio'] = df['双顶径'] / df['股骨长'].replace(0, np.nan)
    df['BPD_HC_Ratio'] = df['双顶径'] / df['头围'].replace(0, np.nan)

    # --- III. 创建交互与多项式特征 ---
    poly_features = ['双顶径', '头围', '腹围', '股骨长', '孕周(超声检查时)']
    for col in poly_features:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_cub'] = df[col] ** 3

    df['BPD_x_AC'] = df['双顶径'] * df['腹围']
    df['BPD_x_FL'] = df['双顶径'] * df['股骨长']
    df['HC_x_AC'] = df['头围'] * df['腹围']
    df['AC_x_FL'] = df['腹围'] * df['股骨长']
    df['GA_x_AC'] = df['孕周(超声检查时)'] * df['腹围']
    df['GA_x_BPD'] = df['孕周(超声检查时)'] * df['双顶径']

    # --- IV. 清理工作 ---
    df = df.drop(columns=['孕妇身高', '孕妇入院体重'])

    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    return df


def load_and_preprocess_data(file_path, target_column, feat_eng=False, binning=False, log_transform=False):
    """
    加载、进行高级特征工程、预处理数据并创建 PyTorch DataLoaders。

    Args:
        file_path (str): 数据集文件路径 (e.g., 'my_dataset.xlsx')。
        target_column (str): 目标变量的列名 (e.g., '出生体重')。
        feat_eng (bool, optional): 是否进行高级特征工程。默认为 True。
        binning (bool, optional): 是否进行目标变量的分箱。默认为 False。
        log_transform (bool, optional): 是否对目标变量进行对数变换。默认为 False。

    Returns:
        tuple: 包含 X, y, y_bins 和 input_dim。
    """
    # 1. 加载数据
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        logger.error(f"错误: 数据文件 '{file_path}' 未找到。请确保文件在正确的路径下。")
        return None, None, None, None

    # 2. 剔除无关或导致数据泄漏的列
    df = df.drop(columns=['ID'])
    df = df.drop(columns=['羊水指数'])

    # 3. 执行高级特征工程
    if feat_eng:
        df = feature_engineering(df)

    # 4. 划分特征 (X) 和目标 (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 5. 对目标变量进行分箱（单位 g）
    if binning:
        logger.info("Binning target variable in original g units...")
        bins = list(np.arange(1470, 4800 + 250, 250))
        labels = list(range(len(bins) - 1))
        y_bins = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

        min_count = 2
        counts = y_bins.value_counts().sort_index()

        while counts.min() < min_count:
            idx_min = counts.idxmin()
            if idx_min == counts.index[0]:
                counts.iloc[1] += counts.iloc[0]
                counts = counts.iloc[1:]
            elif idx_min == counts.index[-1]:
                counts.iloc[-2] += counts.iloc[-1]
                counts = counts.iloc[:-1]
            else:
                counts.iloc[idx_min-1] += counts.iloc[idx_min]
                counts = counts.drop(idx_min)

        valid_bins = counts.index
        y_bins = y_bins.apply(lambda b: b if b in valid_bins else (
            valid_bins[0] if b < valid_bins[0] else valid_bins[-1]
        ))

    else:
        y_bins = None

    # 6. 对目标变量进行 log 空间变换
    if log_transform:
        y = np.log(y)

    input_dim = X.shape[1]
    return X, y, y_bins, input_dim
