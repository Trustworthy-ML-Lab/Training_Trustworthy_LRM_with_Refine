import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

# ─── DISPLAY SETTINGS ──────────────────────────────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ─── EDIT THESE ────────────────────────────────────────────────────────────────
DATASETS = ["AIME2024", "MATH-500", "gpqa", "gsm8k"]

# Structured/Reasoning models (lives under evaluate_reasoning_model_results/)
STRUCT_MODELS = [
    "ReFIne-qwen3-1.7b",
    "ReFIne-qwen3-4b",
    "ReFIne-qwen3-8b",
]

# Baseline models (lives under evaluate_normal_model_confidence/)
BASELINE_MODELS = [
    "Plain-qwen3-1.7b",
    "Plain-qwen3-4b",
    "Plain-qwen3-8b",
]

REASONING_ROOT = Path("evaluate_results")
BASELINE_ROOT  = Path("evaluate_plain_model_confidence_results")

# Threshold for excluding truncated generations from coverage
MAX_REASONING_LEN = 31000
# ────────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal‐width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ids  = np.digitize(probs, bins, right=True)
    ece  = 0.0
    N    = len(probs)
    for b in range(1, n_bins + 1):
        mask = (ids == b)
        if mask.sum() > 0:
            ece += (mask.sum() / N) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)

def extract_confidence(text: str) -> float | None:
    """
    Parse 'Confidence: X/10' where X∈{0,…,10}. Normalize to p=X/10 in [0,1].
    """
    m = re.search(r"Confidence:\s*(\d+)\s*/\s*10\b", text)
    if not m:
        return None
    score = int(m.group(1))
    score = max(0, min(10, score))  # clamp
    return score / 10.0

def is_truncated(rec: dict) -> bool:
    """
    An example is considered truncated if reasoning_length > MAX_REASONING_LEN.
    If the field is missing or unparsable, we treat it as not truncated (eligible).
    """
    rl = rec.get("reasoning_length", None)
    try:
        return (rl is not None) and (int(rl) > MAX_REASONING_LEN)
    except Exception:
        return False

def evaluate_file(path: Path) -> list[dict]:
    """
    Returns a list of dicts, one per run:
      [{run_id, AUROC, Brier, ECE, Coverage}, …]

    Coverage is computed ONLY over *eligible* (non-truncated) examples:
      Coverage = (# eligible examples with a parsed confidence) / (# eligible examples)
    """
    raw = json.loads(path.read_text())
    out = []
    for run in raw["runs"]:
        ps, ys = [], []

        total_all = 0              # all records (debug only)
        total_trunc = 0            # truncated records (debug only)
        eligible_total = 0         # denominator for coverage
        with_conf_eligible = 0     # numerator for coverage

        for rec in run["records"]:
            total_all += 1
            truncated = is_truncated(rec)
            if truncated:
                total_trunc += 1
                # We exclude truncated examples from coverage entirely.
                # (Do not count in numerator or denominator.)
                # We also won't add them to ps/ys, which affects metrics only if
                # they happened to include a confidence before truncation.
                continue

            eligible_total += 1
            p = extract_confidence(rec.get("full_response", ""))
            if p is not None:
                with_conf_eligible += 1
                ps.append(p)
                ys.append(1 if rec.get("correct", False) else 0)

        # DEBUG summary for the run
        print(
            f"{path.parent.parent.name}/{path.parent.name} run {run['run_id']}:",
            f"N_all={total_all},",
            f"N_truncated={total_trunc},",
            f"N_eligible={eligible_total},",
            f"N_with_conf={with_conf_eligible},",
            f"coverage={with_conf_eligible/eligible_total:.3f}" if eligible_total > 0 else "coverage=–",
            f"p_min={np.min(ps) if ps else '–'}",
            f"p_max={np.max(ps) if ps else '–'}",
            f"p_mean={np.mean(ps) if ps else '–'}",
            f"acc={np.mean(ys) if ys else '–'}"
        )

        coverage = (with_conf_eligible / eligible_total) if eligible_total > 0 else np.nan

        rec_dict = {
            "run_id": run["run_id"],
            "Coverage": coverage,
            "AUROC": np.nan,
            "Brier": np.nan,
            "ECE":   np.nan,
        }
        if ps:
            probs  = np.array(ps, dtype=float)
            labels = np.array(ys, dtype=int)
            rec_dict.update({
                "AUROC": roc_auc_score(labels, probs),
                "Brier": brier_score_loss(labels, probs),
                "ECE":   compute_ece(probs, labels),
            })
        out.append(rec_dict)

    return out

def format_mean_std_tables(agg: pd.DataFrame, metric: str, decimals: int = 3) -> pd.DataFrame:
    """
    Build a table of 'mean ± std' for the given metric across (model x dataset).
    Blank out cells where either mean or std is NaN (e.g., missing results).
    """
    mean_df = agg[f"{metric}_mean"].unstack(level=1)
    std_df  = agg[f"{metric}_std"].unstack(level=1)
    formatted = pd.DataFrame(index=mean_df.index, columns=mean_df.columns, dtype=object)
    for idx in mean_df.index:
        for col in mean_df.columns:
            m = mean_df.loc[idx, col]
            s = std_df.loc[idx, col]
            if pd.notna(m) and pd.notna(s):
                formatted.loc[idx, col] = f"{m:.{decimals}f} ± {s:.{decimals}f}"
            else:
                formatted.loc[idx, col] = ""
    return formatted

def format_percentage_tables(agg: pd.DataFrame, metric: str, decimals: int = 1) -> pd.DataFrame:
    """
    Same as above but metric is a fraction (0..1); show as percent with % sign.
    """
    mean_df = agg[f"{metric}_mean"].unstack(level=1)
    std_df  = agg[f"{metric}_std"].unstack(level=1)
    formatted = pd.DataFrame(index=mean_df.index, columns=mean_df.columns, dtype=object)
    for idx in mean_df.index:
        for col in mean_df.columns:
            m = mean_df.loc[idx, col]
            s = std_df.loc[idx, col]
            if pd.notna(m) and pd.notna(s):
                formatted.loc[idx, col] = f"{100*m:.{decimals}f}% ± {100*s:.{decimals}f}%"
            else:
                formatted.loc[idx, col] = ""
    return formatted

def main():
    rows = []

    # Structured/Reasoning results
    for ds in DATASETS:
        for model in STRUCT_MODELS:
            f = REASONING_ROOT / ds / model / "10_runs.json"
            if not f.exists():
                continue
            metrics = evaluate_file(f)
            for m in metrics:
                rows.append({
                    "dataset": ds,
                    "model":   model,
                    "run_id":  m["run_id"],
                    "AUROC":   m["AUROC"],
                    "Brier":   m["Brier"],
                    "ECE":     m["ECE"],
                    "Coverage": m["Coverage"],
                })

    # Baseline results
    for ds in DATASETS:
        for model in BASELINE_MODELS:
            f = BASELINE_ROOT / ds / model / "10_runs.json"
            if not f.exists():
                continue
            metrics = evaluate_file(f)
            for m in metrics:
                rows.append({
                    "dataset": ds,
                    "model":   model,
                    "run_id":  m["run_id"],
                    "AUROC":   m["AUROC"],
                    "Brier":   m["Brier"],
                    "ECE":     m["ECE"],
                    "Coverage": m["Coverage"],
                })

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)

    # Compute mean and std across runs for each (model,dataset)
    agg = df.groupby(["model", "dataset"]).agg(
        AUROC_mean = ("AUROC", "mean"),
        AUROC_std  = ("AUROC", "std"),
        Brier_mean = ("Brier", "mean"),
        Brier_std  = ("Brier", "std"),
        ECE_mean   = ("ECE", "mean"),
        ECE_std    = ("ECE", "std"),
        Coverage_mean = ("Coverage", "mean"),
        Coverage_std  = ("Coverage", "std"),
    )

    # Metrics tables
    for metric in ["AUROC", "Brier", "ECE"]:
        table = format_mean_std_tables(agg, metric, decimals=3)
        print(f"\n=== {metric} (mean ± std) ===")
        print(table.to_string())

    # Confidence coverage table (in %)
    cov_table = format_percentage_tables(agg, "Coverage", decimals=1)
    print(f"\n=== Confidence Coverage (% with 'Confidence: X/10', excluding reasoning_length>{MAX_REASONING_LEN}) (mean ± std) ===")
    print(cov_table.to_string())

if __name__ == "__main__":
    main()
