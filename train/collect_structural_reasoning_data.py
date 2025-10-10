import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(
        description="Filter correct examples, drop any without a confidence tag, "
                    "rebalance confidence by slicing, print distributions, "
                    "write out a new summary JSON and a local HF dataset."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="structural_reasoning_traces/openr1-math-10k/qwen3-8b/summary_1runs.json",
        help="Path to the input summary JSON"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.9,
        help="Mixing weight between pre‑filter distribution and uniform"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write the adjusted summary JSON "
             "(defaults to input path with '_rebalanced.json' suffix)"
    )
    args = parser.parse_args()

    # Determine output JSON path
    if args.output_path:
        out_path = args.output_path
    else:
        base = args.summary_path
        out_path = base[:-5] + "_rebalanced.json" if base.endswith(".json") else base + "_rebalanced.json"

    # 1) Load original summary
    with open(args.summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 2) Pre-filter histogram & mean
    all_confs = []
    for run in summary["runs"]:
        for rec in run["records"]:
            m = re.search(r"Confidence:\s*(\d{1,2})/10", rec.get("full_response",""))
            if m:
                all_confs.append(int(m.group(1)))
    total_all = len(all_confs)
    pre_counts = Counter(all_confs)
    mean_pre = sum(all_confs) / total_all if total_all else 0
    print("Pre‑filter distribution (0–10):", dict(sorted(pre_counts.items())))
    print(f"Mean confidence before filtering: {mean_pre:.2f}\n")

    # 3) Filter correct examples with confidence tag
    filtered = []
    for run in summary["runs"]:
        rid = run["run_id"]
        for rec in run["records"]:
            if not rec.get("correct", False):
                continue
            m = re.search(r"Confidence:\s*(\d{1,2})/10", rec.get("full_response",""))
            if not m:
                continue
            orig = int(m.group(1))
            rec_copy = rec.copy()
            rec_copy["run_id"]    = rid
            rec_copy["orig_conf"] = orig
            filtered.append(rec_copy)

    Nf = len(filtered)
    post_counts = Counter(r["orig_conf"] for r in filtered)
    mean_post = sum(r["orig_conf"] for r in filtered) / Nf if Nf else 0
    print("Post‑filter distribution (0–10):", dict(sorted(post_counts.items())))
    print(f"Mean confidence after filtering: {mean_post:.2f}\n")

    # 4) Compute target counts by mixing with uniform
    alpha   = args.alpha
    uniform = 1/11
    pre_p = {s: pre_counts.get(s,0)/total_all for s in range(0,11)}
    target_p = {s: alpha*pre_p[s] + (1-alpha)*uniform for s in range(0,11)}
    target_counts = {s: int(round(target_p[s]*Nf)) for s in range(0,11)}
    drift = sum(target_counts.values()) - Nf
    if drift:
        s_max = max(pre_p, key=lambda s: pre_p[s])
        target_counts[s_max] -= drift
    print("Target counts (mixed):", dict(sorted(target_counts.items())))

    # 5) Slice‑assign new_conf
    filtered.sort(key=lambda r: r["orig_conf"])
    idx = 0
    for s in range(0,11):
        for i in range(idx, idx + target_counts[s]):
            filtered[i]["new_conf"] = s
        idx += target_counts[s]
    new_counts = Counter(r["new_conf"] for r in filtered)
    mean_new = sum(r["new_conf"] for r in filtered) / Nf if Nf else 0
    print("Rebalanced distribution (0–10):", dict(sorted(new_counts.items())))
    print(f"Mean confidence after rebalancing: {mean_new:.2f}\n")

    # 6) Rebuild summary JSON, preserving all original fields
    runs_out = defaultdict(list)
    for rec in filtered:
        orig = rec.pop("orig_conf")
        new  = rec.pop("new_conf")
        rid  = rec.pop("run_id")
        rec["full_response"] = re.sub(
            rf"Confidence:\s*{orig}/10",
            f"Confidence: {new}/10",
            rec["full_response"]
        )
        rec["confidence"] = new
        runs_out[rid].append(rec)

    out_summary = {k:v for k,v in summary.items() if k!="runs"}
    out_summary["runs"] = [
        {"run_id": rid, "records": runs_out[rid]}
        for rid in sorted(runs_out)
    ]

    # 7) Write rebalanced summary JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote rebalanced summary to: {out_path}")

    # 8) Save to local Hugging Face dataset
    hf_dir = "openr1-math-10k-structural-reasoning-qwen3-8b"
    os.makedirs(hf_dir, exist_ok=True)
    ds = Dataset.from_list(filtered)
    ds.save_to_disk(hf_dir)
    print(f"Saved Hugging Face dataset to: {hf_dir}")
    print(f"Dataset length: {len(ds)}")

if __name__ == "__main__":
    main()
