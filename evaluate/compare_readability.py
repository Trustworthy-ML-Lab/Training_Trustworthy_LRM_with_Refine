#!/usr/bin/env python3
"""
Single‑file comparative interpretability rating of two models’ CoT reasoning via vLLM.
"""

import os
import json
import argparse
import re
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig

# Default max tokens for the rating response
DEFAULT_BUDGET = 4096
# Maximum prompt length in tokens; skip prompts exceeding this
MAX_PROMPT_TOKENS = 35840
MODEL_ID = "Qwen/QwQ-32B"
BOX_RE = re.compile(r"\\boxed\{([1-5])\}")

# Prompt template for pairwise reasoning comparison (choose one option)
PROMPT_TMPL = """
You are a **readability judge**.
Your single task is to compare the two model reasonings and decide which one is easier to follow.
**Do NOT evaluate correctness of the math**—treat all equations as plain text.

Focus only on readability:
• Which reasoning is more organized and less fragmented?
• Which flows more smoothly from one part to the next?
• Which uses clearer language and structure that makes it easier to track?

Evaluate using these criteria:
1) Orientation & plan: conveys a concrete, problem-specific approach.
2) Local cohesion: sentences follow logically; transitions are explicit when steps change.
3) Focus & economy: minimal redundancy; no meandering; good signal-to-noise.
4) Reference clarity: terms/variables introduced before use and referred to consistently.
5) Organization: reasoning unfolds in a clear progression, regardless of headings or tags.

Below are two model reasonings for the same problem.

### Problem
{question}

### Model 1 Reasoning
{response1}

### Model 2 Reasoning
{response2}

Choose the option that best reflects relative readability:

1 – Model 1 is clearly easier to read than Model 2
2 – Model 1 is slightly easier to read than Model 2
3 – Both are equally readable
4 – Model 2 is slightly easier to read than Model 1
5 – Model 2 is clearly easier to read than Model 1

After comparing, output **ONLY** the final option number as \\boxed{{<integer>}}.
""".strip()


def apply_chat(prompt: str, tokenizer):
    conv = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


def make_params(n: int, budget: int, cfg) -> SamplingParams:
    kw = {"n": n, "max_tokens": budget}
    if getattr(cfg, "temperature", None) is not None:
        kw["temperature"] = cfg.temperature
    if getattr(cfg, "top_k", None) is not None:
        kw["top_k"] = cfg.top_k
    if getattr(cfg, "top_p", None) is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)


def build_prompt(question: str, r1: str, r2: str) -> str:
    return PROMPT_TMPL.format(question=question, response1=r1, response2=r2)


def parse_boxed(text: str) -> int:
    matches = BOX_RE.findall(text)
    return int(matches[-1]) if matches else None


def main(dataset: str,
         model1: str,
         model2: str,
         n_sample: int,
         tp: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tp,
        max_model_len=40960,
        gpu_memory_utilization=0.8,
        dtype="bfloat16"
    )
    cfg = GenerationConfig.from_pretrained(MODEL_ID)
    sampling_params = make_params(n=1, budget=DEFAULT_BUDGET, cfg=cfg)

    # Load both model outputs
    path1 = f"evaluate_results/{dataset}/{model1}/{n_sample}_runs.json"
    path2 = f"evaluate_results/{dataset}/{model2}/{n_sample}_runs.json"
    with open(path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    assert len(data1["runs"]) == len(data2["runs"]), "Mismatch in runs count"

    # Build list of all prompts, skipping too-long ones by token count
    raw_prompts = []
    meta = []
    skip = []
    for run_idx, (run1, run2) in enumerate(zip(data1["runs"], data2["runs"])):
        recs1 = run1.get("records", [])
        recs2 = run2.get("records", [])
        assert len(recs1) == len(recs2), f"Mismatch in records for run {run_idx}"
        for rec_idx, (r1, r2) in enumerate(zip(recs1, recs2)):
            question = r1.get("question") or r2.get("question") or data1.get("question", "")
            prompt = build_prompt(question, r1.get("full_response", ""), r2.get("full_response", ""))
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > MAX_PROMPT_TOKENS:
                skip.append(True)
            else:
                skip.append(False)
                raw_prompts.append(prompt)
            meta.append((run_idx, rec_idx))

    # Prepare inputs for vLLM
    chat_inputs = [apply_chat(p, tokenizer) for p in raw_prompts]
    results = llm.generate(prompts=chat_inputs, sampling_params=sampling_params)

    # Prepare output structure
    output = {"runs": [], "aggregate": {}}
    scores_per_run = [[] for _ in data1["runs"]]
    for run in data1["runs"]:
        output["runs"].append({"run_id": run.get("run_id"), "records": []})

    # Map back results, assign None for skipped
    res_i = 0
    for idx, (run_idx, rec_idx) in enumerate(meta):
        if skip[idx]:
            choice = None
        else:
            gen_text = results[res_i].outputs[0].text.strip()
            choice = parse_boxed(gen_text)
            res_i += 1
        scores_per_run[run_idx].append(choice)
        output["runs"][run_idx]["records"].append({
            "comparison_choice": choice,
            "raw_response": gen_text if not skip[idx] else None
        })

    # Compute aggregates ignoring None
    mean_scores = []
    for scores in scores_per_run:
        valid = [s for s in scores if s is not None]
        mean_scores.append(np.mean(valid) if valid else None)
    all_valid = [m for m in mean_scores if m is not None]
    agg = {
        "mean_choice": float(np.mean(all_valid)) if all_valid else None,
        "std_choice": float(np.std(all_valid, ddof=1)) if len(all_valid) > 1 else 0.0
    }
    output["aggregate"] = agg

    # Save results
    out_dir = f"evaluate_readability_results/{dataset}/{model1}_vs_{model2}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{n_sample}_runs.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nComparison on dataset '{dataset}' ({n_sample} samples each):")
    print(f"Mean choice: {agg['mean_choice']:.2f} ± {agg['std_choice']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two models’ CoT clarity via vLLM"
    )
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument(
        "--structural_model", required=True,
        help="Sub‑folder for the first (structural) model"
    )
    parser.add_argument(
        "--normal_model", required=True,
        help="Sub‑folder for the second (normal) model"
    )
    parser.add_argument(
        "--n_sample", type=int, default=10,
        help="Number of runs per question"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Tensor parallel size for vLLM"
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model1=args.structural_model,
        model2=args.normal_model,
        n_sample=args.n_sample,
        tp=args.tensor_parallel_size
    )
