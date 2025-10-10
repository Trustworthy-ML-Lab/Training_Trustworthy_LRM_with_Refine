import os
import glob
import json
import pandas as pd

# Ensure full-width display of wide tables
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ——— CONFIG ———
comparison_dir = "evaluate_readability_results"

# Define choice labels (fixed column order)
choice_labels = {
    1: "ReFIne clearly better",
    2: "ReFIne slightly better",
    3: "equal quality",
    4: "Plain slightly better",
    5: "Plain clearly better",
}
columns = list(choice_labels.values())

# Static list of datasets in desired order (rows)
datasets = ["AIME2024", "MATH-500", "gpqa", "gsm8k"]

# Discover all comparison pairs across datasets
pairs = set()
for d in datasets:
    pair_dirs = glob.glob(os.path.join(comparison_dir, d, "*_vs_*"))
    for pd_path in pair_dirs:
        pairs.add(os.path.basename(pd_path))

def load_choices(json_path):
    """Load counts per choice and total from a single *_runs.json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = {i: 0 for i in choice_labels}
    total = 0
    for run in data.get('runs', []):
        for rec in run.get('records', []):
            choice = rec.get('comparison_choice')
            if isinstance(choice, int) and choice in choice_labels:
                counts[choice] += 1
                total += 1
    return counts, total

# If nothing found, still show one placeholder table
if not pairs:
    pairs = {"No_pairs_found"}

# For each comparison pair, print a table (always prints)
for pair in sorted(pairs):
    table_rows = []
    for d in datasets:
        # Initialize row with blanks for all columns
        row = {col: '' for col in columns}

        pattern = os.path.join(comparison_dir, d, pair, "*_runs.json")
        files = glob.glob(pattern)

        if files:
            agg_counts = {i: 0 for i in choice_labels}
            agg_total = 0
            for fp in files:
                counts, total = load_choices(fp)
                for i in choice_labels:
                    agg_counts[i] += counts[i]
                agg_total += total

            if agg_total > 0:
                for i, label in choice_labels.items():
                    pct = agg_counts[i] / agg_total * 100.0
                    row[label] = f"{pct:.2f}%"

        row['_dataset'] = d
        table_rows.append(row)

    # Build DataFrame even if all rows are blank
    df = pd.DataFrame(table_rows, dtype=str)
    df = df[['_dataset'] + columns].rename(columns={'_dataset': 'Dataset'})
    df.set_index('Dataset', inplace=True)

    print(f"\n=== Comparison: {pair} ===\n")
    print(df.to_string(), "\n")
