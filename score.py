import os
import json
import yaml
import pandas as pd
from glob import glob

# === 1. 收集数据 ===
records = []
root = r"D:\CV\Infant-Weight-Estimation\sensitivity_analysis"

for json_path in glob(os.path.join(root, "**", "checkpoints", "metrics_foldFinal.json"), recursive=True):
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(json_path)), "config.yaml")

    with open(json_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # 只保留顶层指标（数值类型）和 >=3800g
    top_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    g3800_metrics = {}
    if ">=3800g" in metrics and isinstance(metrics[">=3800g"], dict):
        g3800_metrics = {f"{k}_3800+": v for k, v in metrics[">=3800g"].items() if isinstance(v, (int, float))}

    # 读取 families
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    families = "+".join(cfg['model']['params']['families'])

    # 汇总
    record = {"families": families}
    record.update(top_metrics)
    record.update(g3800_metrics)
    records.append(record)

df = pd.DataFrame(records)


# === 2. 指标归一化 ===
def normalize(series, higher_better=False):
    if higher_better:
        return (series - series.min()) / (series.max() - series.min())
    else:
        return (series.max() - series) / (series.max() - series.min())


for col in df.columns:
    if "families" in col: continue
    if "Within" in col:
        df[col + "_norm"] = normalize(df[col], higher_better=True)
    else:
        df[col + "_norm"] = normalize(df[col], higher_better=False)

# === 3. 计算加权得分（按照你的原逻辑）===
weights = {
    "系统误差(%)": 0.12, "随机误差(%)": 0.12, "MAPE(%)": 0.12,
    "MAE(g)": 0.12, "RMSE(g)": 0.12,
    "Within5%": 0.08, "Within10%": 0.08, "Within15%": 0.08,
}

score_all = sum(df[f"{k}_norm"] * w for k, w in weights.items())
score_3800 = sum(df[f"{k}_norm"] * w for k, w in weights.items() if "(>3800g)" not in k)

df["Score_all"] = score_all
df["Score_3800+"] = score_3800
df["Total_Score"] = 0.7 * df["Score_all"] + 0.3 * df["Score_3800+"]

# === 4. 排序与输出 ===
df = df.sort_values("Total_Score", ascending=False)
df.to_csv(r"D:\CV\Infant-Weight-Estimation\模型评分结果\model_performance_ranking.csv", index=False, encoding="utf-8-sig")
print(df.head(10))
