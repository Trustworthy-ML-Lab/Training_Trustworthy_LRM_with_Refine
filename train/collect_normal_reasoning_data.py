import argparse
import json
from collections import Counter
from datasets import Dataset
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Filter normal reasoning results by final-answer correctness."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="normal_reasoning_traces/openr1-math-10k/qwen3-8b/summary_1runs.json",
        help="Path to the summary JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="openr1-math-10k-normal-reasoning-qwen3-8b",
        help="Directory to save the filtered Hugging Face dataset."
    )
    args = parser.parse_args()

    # 1) Load the summary JSON file
    with open(args.summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 2) Filter examples: only include records where 'correct' is True
    filtered = []
    reasoning_lengths = []

    for run in summary.get("runs", []):
        run_id = run.get("run_id")
        for rec in run.get("records", []):
            if rec.get("correct", False):
                rec_copy = rec.copy()
                rec_copy["run_id"] = run_id
                filtered.append(rec_copy)
                reasoning_lengths.append(rec.get("reasoning_length", 0))

    # 3) Print basic statistics
    print(f"Total examples with final prediction correct: {len(filtered)}")

    if filtered:
        print("\nReasoning length statistics among filtered data:")
        print(f"  Max:     {np.max(reasoning_lengths)}")
        print(f"  99%ile:  {np.percentile(reasoning_lengths, 99):.1f}")
        print(f"  95%ile:  {np.percentile(reasoning_lengths, 95):.1f}")
        print(f"  90%ile:  {np.percentile(reasoning_lengths, 90):.1f}")
        print(f"  Median:  {np.median(reasoning_lengths):.1f}")
    else:
        print("\nNo examples matched the criteria.")

    # 4) Save as a Hugging Face Dataset
    if filtered:
        ds = Dataset.from_list(filtered)
        os.makedirs(args.output_dir, exist_ok=True)
        ds.save_to_disk(args.output_dir)
        print(f"\nSaved filtered dataset to '{args.output_dir}'")

if __name__ == "__main__":
    main()
