import os
import glob
import json
import pandas as pd

# ——— CONFIGURE THESE ———
all_models = [
    "ReFIne-qwen3-1.7b",
    "Plain-qwen3-1.7b",
    "ReFIne-qwen3-4b",
    "Plain-qwen3-4b",
    "ReFIne-qwen3-8b",
    "Plain-qwen3-8b",
]
datasets = ["AIME2024", "gpqa", "MATH-500", "gsm8k"]
root_dir = "evaluate_results"
# ——————————————


def load_aggregate(dataset: str, model: str):
    """
    Find the *_runs.json under
      root_dir / dataset / model
    and return its ['aggregate'] dict, or None if not found.
    """
    path = os.path.join(root_dir, dataset, model, "*_runs.json")
    files = glob.glob(path)
    if not files:
        return None
    with open(files[0], "r") as f:
        summary = json.load(f)
    return summary.get("aggregate")


def group_models_by_size(models):
    groups = {"1.7B": [], "4B": [], "8B": []}
    for m in models:
        if "1.7b" in m:
            groups["1.7B"].append(m)
        elif "4b" in m:
            groups["4B"].append(m)
        elif "8b" in m:
            groups["8B"].append(m)
    return groups


def build_tables(model_list, label):
    acc_dict = {}
    len_dict = {}

    for m in model_list:
        acc_row = {}
        len_row = {}
        for d in datasets:
            agg = load_aggregate(d, m)
            if agg is None:
                # no file → leave blank
                acc_row[d] = ""
                len_row[d] = ""
            else:
                acc_row[d] = f"{agg['mean_accuracy']:.2f}% ± {agg['std_accuracy']:.2f}%"
                len_row[d] = f"{agg['mean_length']:.1f} ± {agg['std_length']:.1f}"
        acc_dict[m] = acc_row
        len_dict[m] = len_row

    acc_df = pd.DataFrame.from_dict(acc_dict, orient="index", columns=datasets)
    len_df = pd.DataFrame.from_dict(len_dict, orient="index", columns=datasets)
    acc_df.index.name = "Model"
    len_df.index.name = "Model"

    print(f"\n=== {label} Models ===\n")
    print("► Accuracy (mean ± std %)\n")
    print(acc_df, "\n")
    print("► Reasoning Length (mean ± std tokens)\n")
    print(len_df, "\n")


# ——— MAIN ———
model_groups = group_models_by_size(all_models)
for size_label, models_in_group in model_groups.items():
    build_tables(models_in_group, size_label)
