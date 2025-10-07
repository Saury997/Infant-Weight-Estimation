# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

from data_loader import load_and_preprocess_data
from model import MLP, KAN
from trainer import Trainer
from utils import get_optimizer, set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Fetal Weight Prediction Model Training')

    # data hyperparameters
    parser.add_argument('--data-path', type=str, default='../data/my_dataset.xlsx',
                        help='Path to the dataset file')
    parser.add_argument('--target-column', type=str, default='出生体重',
                        help='Target column name in the dataset')

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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--save-root', type=str, default='../ckpt',
                        help='Path to save the best model')

    # others
    parser.add_argument('--random-seed', type=int, default=42,
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

    # --- 数据加载和预处理 ---
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset, scaler, input_dim = load_and_preprocess_data(
        file_path=args.data_path,
        target_column=args.target_column,
        random_state=args.random_seed
    )

    if train_dataset is None:
        raise ValueError("数据加载失败，请检查数据文件路径和数据格式。")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Data loaded. Input dimension: {input_dim}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # --- 模型、损失函数和优化器 ---
    print("Initializing model...")
    if args.model == 'MLP':
        print("Using MLP model.")
        model = MLP(input_dim=input_dim, hidden_layers=[128, 64], dropout_rate=args.dropout, init_type=args.init_type)
    elif args.model == 'KAN':
        print("Using KAN model.")
        model = KAN(layers_hidden=[input_dim] + args.hidden_layers + [1])
    else:
        raise ValueError(f"Invalid model: {args.model}. Please choose from ['MLP', KAN].")

    # MSE 对大误差的惩罚更重
    criterion = nn.MSELoss()

    # 优化器
    optimizer = get_optimizer(model, args.optimizer, args.lr)

    # --- 训练模型 ---
    print("Starting training...")
    trainer = Trainer(model, criterion, optimizer, args)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_root=args.save_root
    )

    # --- 在测试集上评估最终模型 ---
    print("\nEvaluating the best model on the test set...")
    # trainer.model 中已经加载了验证集上表现最好的模型权重
    test_loss, test_mae = trainer.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)  # RMSE 是 MSE 的平方根，单位也与目标一致

    print("--- Test Set Performance ---")
    print(f"Mean Squared Error (MSE): {test_loss:.4f}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}g")
    print(f"Mean Absolute Error (MAE): {test_mae:.2f}g")


if __name__ == "__main__":
    main()