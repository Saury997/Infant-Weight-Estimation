#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2025/10/7 13:04 
* Project: InfantWeight 
* File: data_loader.py
* IDE: PyCharm 
* Function:
"""
import pandas as pd
import numpy as np


def feature_engineering(df):
    """
    对数据集进行详尽的特征工程。

    Args:
        df (pd.DataFrame): 原始数据框。

    Returns:
        pd.DataFrame: 经过特征工程后的数据框。
    """
    print("Performing advanced feature engineering...")

    # --- I. 创建生理学综合指标 ---
    # 计算孕妇BMI (注意单位转换: cm -> m)
    # 为避免除以0的错误，对身高做下界保护
    height_m = df['孕妇身高'] / 100
    height_m[height_m == 0] = np.nan  # 将身高为0的视为无效值
    df['孕妇BMI'] = df['孕妇入院体重'] / (height_m ** 2)

    # --- II. 创建比例特征 ---
    # 为避免除以0，分母为0时结果设为0或NaN
    df['HC_AC_Ratio'] = df['头围'] / df['腹围'].replace(0, np.nan)
    df['FL_AC_Ratio'] = df['股骨长'] / df['腹围'].replace(0, np.nan)
    df['BPD_FL_Ratio'] = df['双顶径'] / df['股骨长'].replace(0, np.nan)
    df['BPD_HC_Ratio'] = df['双顶径'] / df['头围'].replace(0, np.nan)

    # --- III. 创建交互与多项式特征 ---
    # 定义需要创建高次项的核心生物测量指标
    poly_features = ['双顶径', '头围', '腹围', '股骨长', '孕周(超声检查时)']

    for col in poly_features:
        # 创建平方项
        df[f'{col}_sq'] = df[col] ** 2
        # 创建立方项 (对体重这类三维量纲的预测可能更重要)
        df[f'{col}_cub'] = df[col] ** 3

    # 创建核心交互项 (模拟经典估重公式)
    df['BPD_x_AC'] = df['双顶径'] * df['腹围']
    df['BPD_x_FL'] = df['双顶径'] * df['股骨长']
    df['HC_x_AC'] = df['头围'] * df['腹围']
    df['AC_x_FL'] = df['腹围'] * df['股骨长']

    # 孕周与核心指标的交互
    df['GA_x_AC'] = df['孕周(超声检查时)'] * df['腹围']
    df['GA_x_BPD'] = df['孕周(超声检查时)'] * df['双顶径']

    # --- IV. 清理工作 ---
    # 创建了BMI后，原始的身高体重可以考虑移除，以减少共线性
    df = df.drop(columns=['孕妇身高', '孕妇入院体重'])

    # 用中位数填充所有过程中可能产生的NaN值
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

    Returns:
        tuple: 包含 X, y 和 input_dim。
    """
    # 1. 加载数据
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_path}' 未找到。请确保文件在正确的路径下。")
        return None, None, None, None, None

    # 2. 剔除无关或导致数据泄漏的列
    df = df.drop(columns=['ID'])

    # 3. 执行高级特征工程
    if feat_eng:
        print("Performing advanced feature engineering...")
        df = feature_engineering(df)

    # 4. 划分特征 (X) 和目标 (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if log_transform:
        y = np.log(y)

    # 5. 对目标变量进行分箱
    if binning:
        print("Binning target variable...")
        bins = list(np.arange(1470, 4800 + 500, 500))
        labels = list(range(len(bins) - 1))
        y_bins = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    else:
        y_bins = None

    # 获取输入特征的维度
    input_dim = X.shape[1]

    return X, y, y_bins, input_dim