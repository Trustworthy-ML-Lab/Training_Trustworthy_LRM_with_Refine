import os
import json
import pandas as pd

# Define models and datasets
models = [
    "ReFIne-qwen3-1.7b",
    "ReFIne-qwen3-4b",
    "ReFIne-qwen3-8b",
    "ReFIne-qwen3-1.7b-sft-only",
    "ReFIne-qwen3-4b-sft-only",
    "ReFIne-qwen3-8b-sft-only",
]
datasets = ["AIME2024", "MATH-500", "gpqa", "gsm8k"]
base_dir = "evaluate_commitment_faithfulness_results"

# Collect data
data = {}
for model in models:
    row = {}
    for ds in datasets:
        json_path = os.path.join(base_dir, ds, model, "10_runs.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            agg = d.get("aggregate", {})
            u = agg.get("understanding_mean")
            f_mean = agg.get("facts_mean")
            p = agg.get("plan_mean")
            if u is not None and f_mean is not None and p is not None:
                row[ds] = f"{u:.2f} / {f_mean:.2f} / {p:.2f}"
            else:
                row[ds] = ""
        else:
            row[ds] = ""
    data[model] = row

# Create DataFrame
df = pd.DataFrame.from_dict(data, orient="index", columns=datasets)

# ——— Ensure no truncation ———
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# ————————————————

print("commitment faithfulness (understanding / facts / plan):")
print(df)
