# Simple batched inference (e.g., on MATH-500) using vLLM.
# - Prompts the model to show step-by-step reasoning
# - Requires the final answer in \boxed{...}
# - Also requires: "Confidence: X/10" (X is an integer 0–10) on a new line
# - Saves per-run accuracies, reasoning lengths, confidences + mean/std to JSON.

import os
import json
import argparse
import numpy as np
import re
import torch
from datasets import load_dataset
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from utils import extract_answer, verify_answer, DATASET_MAP, MODEL_MAP

# Set a default budget for reasoning tokens
DEFAULT_BUDGET = 32768 - 1024

CONFIDENCE_REGEX = re.compile(
    r'Confidence\s*:\s*(\d{1,2})(?:\s*/\s*10)?\b', re.IGNORECASE
)

def build_instruction_prompt(question: str) -> str:
    """
    Simple instruction: final boxed answer, then 'Confidence: X/10' on a new line.
    """
    return (
        f"Problem:\n{question}\n"
        "Solve the above problem. After you output the final answer, on a new line, you **MUST** evaluate the soundness of your reasoning and give a confidence estimation exactly: Confidence: X/10 "
        "(X is an integer 0–10 reflecting your chance of being correct based on your reasoning).\n\n"
    )


def apply_chat(question: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": build_instruction_prompt(question)}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )

def make_params(n: int, budget: int, cfg) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    """
    kw = {"n": n, "max_tokens": budget}
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = cfg.top_k
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)

def extract_confidence(text: str):
    """
    Parse 'Confidence: X/10' from the model output. Returns int in [0,10] or None.
    """
    m = CONFIDENCE_REGEX.search(text)
    if not m:
        return None
    try:
        x = int(m.group(1))
        # clip to [0,10] just in case
        return max(0, min(10, x))
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Inference with boxed final answer + confidence (X/10) and JSON output"
    )
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="structural-qwen2.5-7b")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # 1) Load dataset
    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    ds = load_dataset(dataset_name, split=split)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key = DATASET_MAP[args.dataset]["answer_key"]

    if args.dataset == "AIME2024":
        override_28 = r"""Torus $T$ is the surface produced by revolving a circle with radius $3$ around an axis in the plane of the circle that is a distance $6$ from the center of the circle (so like a donut). Let $S$ be a sphere with a radius $11$. When $T$ rests on the inside of $S$, it is internally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. The difference $r_i-r_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$. 
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)--(2.9,-0.4)); [/asy]"""
        ds = ds.map(
            lambda example, idx: {"problem": override_28} if idx == 28 else example,
            with_indices=True
        )

    # 2) Load model config and tokenizer
    model_id  = MODEL_MAP[args.model]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    if args.model == "deepseek-qwen3-8b":
        cfg = GenerationConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
    else:
        cfg = GenerationConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3) Initialize vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=DEFAULT_BUDGET + 1024,
        dtype=torch.bfloat16
    )

    # 4) Prepare sampling parameters
    sampling_params = make_params(args.n_sample, DEFAULT_BUDGET, cfg)

    # 5) Build prompts (with explicit formatting + confidence requirement)
    prompts = []
    for ex in ds:
        q = ex[question_key]
        prompts.append(apply_chat(q, tokenizer))

    # 6) Generate
    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # 7) Collect runs and compute stats
    runs = {rid: [] for rid in range(args.n_sample)}
    for idx, gen in enumerate(results):
        gold = ds[idx][answer_key]
        if args.dataset == "gpqa":
            gold = extract_answer(gold)
        for rid, out in enumerate(gen.outputs):
            text = out.text.strip()

            # prediction extraction
            pred = extract_answer(text)

            # correctness
            correct = False
            try:
                correct = verify_answer(gold, pred)
            except Exception:
                pass

            # reasoning length (entire response) in tokens
            reasoning_length = len(tokenizer.encode(text, add_special_tokens=False))

            # confidence extraction
            conf = extract_confidence(text)  # int 0..10 or None

            runs[rid].append({
                "question":           ds[idx][question_key],
                "full_response":      text,
                "reasoning_length":   reasoning_length,
                "prediction":         pred,
                "gold":               gold,
                "correct":            correct,
                "confidence":         conf
            })

    summary = {"runs": [], "aggregate": {}}
    accs = []
    lengths = []
    confs = []

    for run_id, recs in runs.items():
        # accuracy per run
        acc = sum(1 for r in recs if r["correct"]) / len(recs) * 100
        accs.append(acc)

        # average reasoning length per run
        avg_len = sum(r["reasoning_length"] for r in recs) / len(recs)
        lengths.append(avg_len)

        # average confidence per run (ignore None)
        run_confs = [r["confidence"] for r in recs if r["confidence"] is not None]
        avg_conf = float(np.mean(run_confs)) if run_confs else None
        confs.append(avg_conf if avg_conf is not None else np.nan)

        summary["runs"].append({
            "run_id":           run_id,
            "accuracy":         acc,
            "avg_length":       avg_len,
            "avg_confidence":   avg_conf,
            "records":          recs
        })

    # Aggregate across runs
    # For confidence, ignore runs where avg_confidence is None (np.nan)
    confs_clean = [c for c in confs if not (c is None or np.isnan(c))]

    summary["aggregate"]["mean_accuracy"]     = float(np.mean(accs))
    summary["aggregate"]["std_accuracy"]      = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    summary["aggregate"]["mean_length"]       = float(np.mean(lengths))
    summary["aggregate"]["std_length"]        = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
    summary["aggregate"]["mean_confidence"]   = float(np.mean(confs_clean)) if confs_clean else None
    summary["aggregate"]["std_confidence"]    = float(np.std(confs_clean, ddof=1)) if len(confs_clean) > 1 else (0.0 if confs_clean else None)

    # 8) Save JSON
    output_path = f"evaluate_plain_model_confidence_results/{args.dataset}/{args.model}/{args.n_sample}_runs.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Per-run accuracies: {accs}")
    print(f"Mean ± std accuracy: {summary['aggregate']['mean_accuracy']:.2f}% ± {summary['aggregate']['std_accuracy']:.2f}%")
    print(f"Mean ± std length: {summary['aggregate']['mean_length']:.1f} ± {summary['aggregate']['std_length']:.1f} tokens")
    if summary["aggregate"]["mean_confidence"] is None:
        print("Mean ± std confidence: (no valid confidences parsed)")
    else:
        print(f"Mean ± std confidence: {summary['aggregate']['mean_confidence']:.2f}/10 ± {summary['aggregate']['std_confidence']:.2f}")

    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
