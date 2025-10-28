import os
import json
import yaml
import pandas as pd

# 根目录
ROOT_DIR = r"D:\CV\Infant-Weight-Estimation\sensitivity_analysis"
OUTPUT_CSV = os.path.join(ROOT_DIR, "metrics_summary.csv")

records = []

for root, dirs, files in os.walk(ROOT_DIR):
    if "checkpoints" in dirs:
        ckpt_dir = os.path.join(root, "checkpoints")
        json_path = os.path.join(ckpt_dir, "metrics_foldFinal.json")
        config_path = os.path.join(root, "config.yaml")

        if os.path.exists(json_path) and os.path.exists(config_path):
            try:
                # 读取 JSON
                with open(json_path, "r", encoding="utf-8") as jf:
                    metrics = json.load(jf)

                # 读取 YAML
                with open(config_path, "r", encoding="utf-8") as yf:
                    config = yaml.safe_load(yf)

                families = config.get("model", {}).get("params", {}).get("families", [])

                # 将 families 转成字符串方便展示
                families_str = "+".join(families)

                # 合并成一行
                record = {"families": families_str, "path": root}
                if isinstance(metrics, dict):
                    record.update(metrics)

                records.append(record)
                print(f"✅ Processed: {families_str}")
            except Exception as e:
                print(f"⚠️ Failed to process {root}: {e}")

# 汇总成 DataFrame
if records:
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n🎯 汇总完成，共 {len(records)} 条记录，已保存至：\n{OUTPUT_CSV}")
else:
    print("❌ 未找到任何 metrics_foldFinal.json 文件。")
