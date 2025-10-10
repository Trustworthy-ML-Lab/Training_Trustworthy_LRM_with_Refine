import os
import glob
import json

import numpy as np
import pandas as pd

# ─── DISPLAY SETTINGS ──────────────────────────────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ——— CONFIG ———
models = [
    "ReFIne-qwen3-1.7b",
    "ReFIne-qwen3-1.7b-sft-only",
    "ReFIne-qwen3-4b",
    "ReFIne-qwen3-4b-sft-only",
    "ReFIne-qwen3-8b",
    "ReFIne-qwen3-8b-sft-only",
]
datasets = ["AIME2024", "gpqa", "MATH-500", "gsm8k"]
root_dir = "evaluate_results"
# —————————————————

expected_tags = [
    "<understanding>", "</understanding>",
    "<facts>",         "</facts>",
    "<plan>",          "</plan>",
    "<think>",         "</think>",
    "<final_answer>",  "</final_answer>",
    "<self_assessment>",  "</self_assessment>",
]
think_subtags = ["<understanding>", "<facts>", "<plan>"]

# container for (model, metric) → { dataset: "mean±std" or "" }
results = {}

for m in models:
    for metric in ["Structure", "Understanding", "Facts", "Plan"]:
        results[(m, metric)] = {}

    for d in datasets:
        # find the runs file
        pattern = os.path.join(root_dir, d, m, "*_runs.json")
        files = glob.glob(pattern)
        if not files:
            print(f"⚠️  Skipping {m} @ {d}: no runs file found.")
            # leave all metrics blank for this dataset
            results[(m, "Structure")][d] = ""
            results[(m, "Understanding")][d] = ""
            results[(m, "Facts")][d] = ""
            results[(m, "Plan")][d] = ""
            continue

        data = json.load(open(files[0], "r", encoding="utf-8"))
        struct_rates = []
        sub_rates = {tag: [] for tag in think_subtags}

        # iterate runs
        for run in data.get("runs", []):
            recs = run.get("records", [])
            recs = [r for r in recs if r.get("reasoning_length", 0) <= 31000]
            total = len(recs)
            if total == 0:
                # nothing to compute for this run
                continue

            s_ok = 0
            counts = {tag: 0 for tag in think_subtags}

            for rec in recs:
                text = rec.get("full_response", "")
                low = text.lower()

                # 1) full-structure? (respect exact tag order)
                pos, ok = 0, True
                for tag in expected_tags:
                    idx = text.find(tag, pos)
                    if idx < 0:
                        ok = False
                        break
                    pos = idx + len(tag)
                if ok:
                    s_ok += 1

                # 2) each subtag inside <think>…</think>
                if "<think>" in low and "</think>" in low:
                    body = low.split("<think>", 1)[1].split("</think>", 1)[0]
                    for tag in think_subtags:
                        if tag in body:
                            counts[tag] += 1

            struct_rates.append(s_ok / total * 100.0)
            for tag in think_subtags:
                sub_rates[tag].append(counts[tag] / total * 100.0)

        # compute mean±std for structure (blank if no data)
        if struct_rates:
            ms, ss = np.mean(struct_rates), np.std(struct_rates, ddof=1) if len(struct_rates) > 1 else (np.mean(struct_rates), 0.0)
            results[(m, "Structure")][d] = f"{ms:.2f}% ± {ss:.2f}%"
        else:
            results[(m, "Structure")][d] = ""

        # and for each sub‐metric
        for tag in think_subtags:
            rates = sub_rates[tag]
            name = tag.strip("<>").capitalize()
            if rates:
                mr = np.mean(rates)
                sr = np.std(rates, ddof=1) if len(rates) > 1 else 0.0
                results[(m, name)][d] = f"{mr:.2f}% ± {sr:.2f}%"
            else:
                results[(m, name)][d] = ""

# build MultiIndex DataFrame
index = pd.MultiIndex.from_tuples(results.keys(), names=["Model", "Metric"])
df = pd.DataFrame.from_dict(results, orient="index", columns=datasets)
df.index = index

print("\n► Compliance rates by model & dataset (mean ± std %)\n")
print(df)
