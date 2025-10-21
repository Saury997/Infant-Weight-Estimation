#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang & Lanxiang Ma
* Date: 2025/10/17 10:16 
* Project: Infant-Weight-Estimation 
* File: plot.py
* IDE: PyCharm 
* Function:
"""
from typing import Tuple, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score

matplotlib.use('Agg')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    "axes.unicode_minus": False,
})


def result_plot(
    y_true_train: np.ndarray, y_pred_train: np.ndarray,
    y_true_test: np.ndarray, y_pred_test: np.ndarray,
    model_name: str = "XGB",
    xlabel: str = r"True value",
    ylabel: str = r"Predicted value",
    panel_tag: str = None,
    show_top_hist: bool = True,
    show_right_hist: bool = True,
    show_bottom_residual: bool = True,
    bins: int = 20,
    figsize: tuple = (2.6, 3.2),
    dpi: int = 300
) -> Tuple[Figure, tuple[Axes, Optional[Axes], Optional[Axes], Optional[Axes]]]:
    """
    紧凑风格 Pred vs True 主图 + （可选）顶部直方图 + （可选）右侧直方图 + （可选）底部残差图。
    主图：x=Predicted, y=True；底部残差：y = Pred - True，x 复用主图（共享 x 轴）。
    """
    rc = {
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.0, "axes.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 7,
        "savefig.pad_inches": 0.02,
    }
    c_train, c_test, c_line = "#8FB7FF", "#F4B183", "#6E6E6E"

    # -------- 数据与范围 --------
    y_true_train = np.asarray(y_true_train).ravel()
    y_pred_train = np.asarray(y_pred_train).ravel()
    y_true_test  = np.asarray(y_true_test).ravel()
    y_pred_test  = np.asarray(y_pred_test).ravel()

    x_all = np.concatenate([y_pred_train, y_pred_test])  # 主图 x 用预测
    y_all = np.concatenate([y_true_train, y_true_test])  # 主图 y 用真实
    lo = min(x_all.min(), y_all.min()); hi = max(x_all.max(), y_all.max())
    pad = 0.02 * (hi - lo + 1e-12); lo -= pad; hi += pad

    # -------- 布局：有/无直方与残差时的行高/列宽 --------
    from matplotlib import gridspec
    top_h    = 0.65 if show_top_hist else 0.0
    main_h   = 5.6
    resid_h  = 1.0 if show_bottom_residual else 0.0
    right_w  = 0.60 if show_right_hist else 0.0

    with mpl.rc_context(rc):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(
            3, 2,
            height_ratios=[top_h, main_h, resid_h],
            width_ratios=[6.0, right_w],
            hspace=0.03, wspace=0.02
        )

        ax_main = fig.add_subplot(gs[1, 0])
        ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main) if show_top_hist else None
        ax_rgt  = fig.add_subplot(gs[1, 1], sharey=ax_main) if show_right_hist else None
        ax_res  = fig.add_subplot(gs[2, 0], sharex=ax_main) if show_bottom_residual else None

        # -------- 主散点（x=Pred, y=True）--------
        ax_main.scatter(y_pred_train, y_true_train, s=3, color=c_train, label="Train")
        ax_main.scatter(y_pred_test,  y_true_test,  s=3, color=c_test,  label="Test")

        # x=y 参考线
        ax_main.plot([lo, hi], [lo, hi], "--", color=c_line, linewidth=0.9, dashes=(3, 2))

        # 线性回归线（y = a x + b）
        def _fit_line(x, y):
            a, b = np.polyfit(x, y, deg=1)
            x_min, x_max = np.min(x), np.max(x)
            xp = np.linspace(x_min, x_max, 100)
            yp = a * xp + b
            return xp, yp
        xp_tr, yp_tr = _fit_line(y_true_train, y_pred_train)
        xp_te, yp_te = _fit_line(y_true_test,  y_pred_test)

        ax_main.plot(yp_tr, xp_tr, color="#1f77b4", linewidth=0.8)
        ax_main.plot(yp_te, xp_te, color="#ff7f0e", linewidth=0.8)

        # 轴 & 比例
        ax_main.set_xlim(lo, hi); ax_main.set_ylim(lo, hi)
        ax_main.set_aspect('equal', adjustable='box')
        ax_main.set_ylabel(ylabel)

        ax_main.spines['left'].set_visible(True)
        ax_main.spines['bottom'].set_visible(True)
        ax_main.spines['top'].set_visible(False if show_top_hist else True)
        ax_main.spines['right'].set_visible(False if show_right_hist else True)

        ax_main.tick_params(axis="x", labelbottom=False)
        ax_main.tick_params(length=2.5, pad=1.5)

        r2_tr = r2_score(y_true_train, y_pred_train)
        r2_te = r2_score(y_true_test,  y_pred_test)
        ax_main.text(0.65, 0.38, model_name,
                     transform=ax_main.transAxes, ha="left", va="center", color='black')
        ax_main.text(0.56, 0.15,
                     f"Train $R^2$: {r2_tr:.3f}\nTest  $R^2$: {r2_te:.3f}",
                     transform=ax_main.transAxes, ha="left", va="bottom",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, boxstyle="round,pad=0.2"))

        if panel_tag is not None:
            ax_main.text(-0.1, 1.1, panel_tag, transform=ax_main.transAxes,
                         fontweight="bold", ha="left", va="bottom")

        ax_main.legend(loc="upper left", handlelength=1.0, borderaxespad=0.2)

        # -------- 顶部直方图 --------
        if ax_top is not None:
            ax_top.hist(y_pred_train, bins=bins, color=c_train, edgecolor='white', linewidth=0.5)
            ax_top.hist(y_pred_test,  bins=bins, color=c_test,  edgecolor='white', linewidth=0.5)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["left"].set_visible(False)

            ax_top.tick_params(axis="x",
                               top=False, bottom=True,
                               labeltop=False, labelbottom=False,
                               length=2.0)
            ax_top.tick_params(axis="y", left=False, right=False, labelleft=False, length=0)

        # -------- 右侧直方图 --------
        if ax_rgt is not None:
            ax_rgt.hist(y_true_train, bins=bins, color=c_train,
                        orientation="horizontal", edgecolor='white', linewidth=0.5)
            ax_rgt.hist(y_true_test,  bins=bins, color=c_test,
                        orientation="horizontal", edgecolor='white', linewidth=0.5)
            ax_rgt.spines["right"].set_visible(False)
            ax_rgt.spines["top"].set_visible(False)
            ax_rgt.spines["bottom"].set_visible(False)

            ax_rgt.tick_params(axis="y",
                                 right=False, left=True,
                                 labelright=False, labelleft=False,
                                 length=2.0)
            ax_rgt.tick_params(axis="x",
                                 bottom=False, top=False,
                                 labelbottom=False, labeltop=False,
                                 length=2.0)

        # -------- 底部残差图（样式仿示图）--------
        if ax_res is not None:
            res_tr = y_pred_train - y_true_train
            res_te = y_pred_test  - y_true_test
            ax_res.scatter(y_pred_train, res_tr, s=3, color=c_train)
            ax_res.scatter(y_pred_test,  res_te, s=3, color=c_test)

            ax_res.axhline(0.0, linestyle="--", color=c_line, linewidth=0.9, dashes=(3, 2))

            r_all = np.concatenate([res_tr, res_te])
            r_rng = np.nanpercentile(np.abs(r_all), 98)
            r_pad = 0.1 * (r_rng + 1e-12)
            ax_res.set_ylim(-r_rng - r_pad, r_rng + r_pad)

            ax_res.set_xlabel(xlabel)
            ax_res.tick_params(length=2.0, pad=0.5)
            ax_res.set_ylabel("Errors", labelpad=1.5)

            for s in ("top", "right", "left", "bottom"):
                ax_res.spines[s].set_visible(True)

        plt.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.96,
                            wspace=0.02, hspace=0.02)
        return fig, (ax_main, ax_top, ax_rgt, ax_res)


def bland_altman_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name="Model",
                      xlabel="Mean of True & Predicted", ylabel="Difference (Pred - True)",
                      figsize=(4, 3), dpi=300):
    """
    绘制 Bland-Altman 图
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(mean_vals, diff_vals, s=5, alpha=0.6)
    plt.axhline(mean_diff, color='gray', linestyle='--', label='Mean Diff')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Bland-Altman: {model_name}")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n_tr, n_te = 300, 180
    xt = rng.normal(-1.5, 0.5, n_tr); yt = 0.98*xt + rng.normal(0, 0.06, n_tr)
    xv = rng.normal(-1.3, 0.6, n_te);  yv = 0.90*xv + rng.normal(0, 0.10, n_te)

    fig, _ = result_plot(
        xt, yt, xv, yv,
        model_name="XGBoost",
        xlabel=r"Predicted value",
        ylabel=r"True value",
        panel_tag="a",
        show_top_hist=True,
        show_right_hist=True,
        show_bottom_residual=True,
        figsize=(2.6, 3.2), dpi=300
    )
    plt.show()