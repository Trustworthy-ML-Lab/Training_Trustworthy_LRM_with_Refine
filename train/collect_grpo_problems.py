#!/usr/bin/env python
import argparse
import json
import os
import random
from datasets import Dataset, load_from_disk

def main():
    parser = argparse.ArgumentParser(
        description="Prepare an RL dataset by mixing incorrect examples "
                    "from the summary JSON with samples from a local HF dataset."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="structural_reasoning_data/openr1-math-10k/qwen3-8b/summary_1runs.json",
        help="Path to the input summary JSON"
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
        default="openr1-math-10k-rl",
        help="Path to the local Hugging Face dataset directory"
    )
    parser.add_argument(
        "--n_incorrect",
        type=int,
        default=1400,
        help="Number of incorrect examples to sample"
    )
    parser.add_argument(
        "--n_hf",
        type=int,
        default=600,
        help="Number of examples to take from the HF dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="openr1-math-2k-rl-hard",
        help="Directory to save the mixed RL dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load summary JSON and extract incorrect records
    with open(args.summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    incorrect_records = []
    for run in summary.get("runs", []):
        for rec in run.get("records", []):
            if not rec.get("correct", False):
                rec_copy = rec.copy()
                incorrect_records.append(rec_copy)

    if len(incorrect_records) < args.n_incorrect:
        raise ValueError(
            f"Only found {len(incorrect_records)} incorrect examples, "
            f"but --n_incorrect={args.n_incorrect}"
        )

    sampled_incorrect = incorrect_records[: args.n_incorrect]
    print(f"Sampled {len(sampled_incorrect)} incorrect examples.")

    # 2) Load local HF dataset and take the first N
    ds_hf = load_from_disk(args.hf_dir)
    if len(ds_hf) < args.n_hf:
        raise ValueError(
            f"Local HF dataset has {len(ds_hf)} examples, "
            f"but --n_hf={args.n_hf}"
        )
    sampled_hf = ds_hf.select(range(args.n_hf))
    print(f"Took first {len(sampled_hf)} examples from local HF dataset.")

    # 3) Build combined list, keeping only "question"→"problem" and "gold"→"answer"
    mixed = []
    for rec in sampled_incorrect:
        mixed.append({
            "problem": rec["question"],
            "answer":  rec["gold"]
        })

    for rec in sampled_hf:
        mixed.append({
            "problem": rec["problem"],
            "answer":  rec["answer"]
        })

    # Optional: shuffle the combined set
    random.shuffle(mixed)

    # 4) Create and save the new HF dataset
    ds_mixed = Dataset.from_list(mixed)
    ds_mixed.save_to_disk(args.output_dir)
    print(f"Saved mixed RL dataset ({len(ds_mixed)} examples) to: {args.output_dir}")

if __name__ == "__main__":
    main()
