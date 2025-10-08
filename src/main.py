# main.py
import os
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

from data_loader import load_and_preprocess_data
from trainer import Trainer
from utils import get_optimizer, set_seed, get_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Fetal Weight Prediction Model Training')

    # data hyperparameters
    parser.add_argument('--data-path', type=str, default='../data/my_dataset.xlsx',
                        help='Path to the dataset file')
    parser.add_argument('--target-column', type=str, default='出生体重',
                        help='Target column name in the dataset')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Proportion of the dataset to hold out for the final test set')

    # model hyperparameters
    parser.add_argument('--model', type=str, default='MLP', help='Model architecture')
    parser.add_argument('--init-type', type=str, default='xavier',
                        choices=['uniform', 'normal', 'xavier', 'kaiming'], help='Initialization type for model weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for the model')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes')

    # training hyperparameters
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'SGD', 'Muon'], help='Optimizer for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--save-root', type=str, default='../ckpt',
                        help='Path to save the best model')
    parser.add_argument('--use-kfold', action='store_true',
                        help='Enable K-Fold Cross-Validation. If not set, uses a simple train/val split.')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for K-Fold Cross-Validation')

    # others
    parser.add_argument('--random-seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_seed)
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    print(f"Using device: {args.device}")

    # --- 1. 数据加载和预处理 ---
    print("Loading and preprocessing data...")
    X, y, input_dim = load_and_preprocess_data(
        file_path=args.data_path,
        target_column=args.target_column,
        random_state=args.random_seed
    )

    # --- 2. 划分出最终的测试集 ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Data loaded. Input dimension: {input_dim}")
    print(f"Total samples for training/validation: {len(X_train_val)}")
    print(f"Held-out test samples: {len(X_test)}")

    if X is None:
        raise ValueError("DataLoadError! Please check your data file path.")

    # --- 3. 根据参数选择执行K-折交叉验证或简单验证 ---
    if args.use_kfold:
        # --- K-Fold Cross-Validation ---
        print(f"\n----- Starting {args.k_folds}-Fold Cross-Validation -----")
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train_val)):
            print(f"\n----- FOLD {fold + 1}/{args.k_folds} -----")

            # --- 数据准备 ---
            X_train, X_val = X_train_val.iloc[train_ids], X_train_val.iloc[val_ids]
            y_train, y_val = y_train_val.iloc[train_ids], y_train_val.iloc[val_ids]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                          torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
            val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                        torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1))

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # --- 模型初始化 ---
            model = get_model(args, input_dim).to(args.device)
            criterion = nn.MSELoss()
            optimizer = get_optimizer(model, args.optimizer, args.lr)

            # --- 训练 ---
            trainer = Trainer(model, criterion, optimizer, args)
            trainer.fit(train_loader, val_loader, args.epochs, args.patience, args.save_root)

            fold_results.append(trainer.best_val_mae)
            print(f"Fold {fold + 1} Best Validation MAE: {trainer.best_val_mae:.2f}g")

        # --- 打印交叉验证总结 ---
        avg_mae = np.mean(fold_results)
        std_mae = np.std(fold_results)
        print("\n--- K-Fold Cross-Validation Summary ---")
        print(f"Average Validation MAE across {args.k_folds} folds: {avg_mae:.2f}g ± {std_mae:.2f}g")
        print("---------------------------------------\n")

    # --- 4. 训练最终模型 ---
    print("----- Training Final Model on All Training Data -----")

    # --- 数据准备 ---
    final_scaler = StandardScaler()
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    y_train_val_tensor = torch.tensor(y_train_val.values, dtype=torch.float32).view(-1, 1)

    final_train_dataset = TensorDataset(torch.tensor(X_train_val_scaled, dtype=torch.float32), y_train_val_tensor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 模型初始化与训练 ---
    final_model = get_model(args, input_dim).to(args.device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(final_model, args.optimizer, args.lr)

    final_trainer = Trainer(final_model, criterion, optimizer, args)
    final_trainer.fit(final_train_loader, None, args.epochs, args.patience, args.save_root)

    # --- 5. 在最终测试集上评估 ---
    print("\n----- Evaluating Final Model on Held-Out Test Set -----")
    X_test_scaled = final_scaler.transform(X_test)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loss, test_mae = final_trainer.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)

    print("\n--- Final Test Set Performance ---")
    print(f"Mean Squared Error (MSE): {test_loss:.4f}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}g")
    print(f"Mean Absolute Error (MAE): {test_mae:.2f}g")


if __name__ == "__main__":
    main()