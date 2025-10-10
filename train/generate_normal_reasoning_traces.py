# inference_vllm_simple.py
#
# Simple batched inference on MATH-500 using vLLM.
# Generates step-by-step reasoning, asks the model to place
# the final answer in a boxed format (e.g., \boxed{answer}),
# and saves per-run accuracies + reasoning lengths + mean/std to JSON.

import os
import json
import argparse
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from math_verify import parse, verify, LatexExtractionConfig
from utils import extract_answer, verify_answer, DATASET_MAP, MODEL_MAP

# # Set a default budget for reasoning tokens
# DEFAULT_BUDGET = 16384


def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
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


def main():
    parser = argparse.ArgumentParser(
        description="Simple inference on MATH-500 with boxed final answer and JSON output"
    )
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="openr1-math-10k")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # 1) Load dataset
    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    ds = load_from_disk(dataset_name)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key   = DATASET_MAP[args.dataset]["answer_key"]

    # 2) Load model config & tokenizer
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
        max_model_len=max_pos,
        dtype="auto"
    )

    # 4) Prepare sampling parameters
    sampling_params = make_params(args.n_sample, max_pos - 1024, cfg)

    # 5) Build prompts
    prompts = []
    for ex in ds:
        prompt = (
            "Solve the following math problem and place your final answer "
            "in a boxed format using LaTeX, e.g., \\boxed{answer}.\n\n"
            f"Problem: {ex[question_key]}"
        )
        prompts.append(apply_chat(prompt, tokenizer))

    # 6) Generate
    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # 7) Collect runs and compute stats
    runs = {rid: [] for rid in range(args.n_sample)}
    for idx, gen in enumerate(results):
        gold = ds[idx][answer_key]
        for rid, out in enumerate(gen.outputs):
            text = out.text.strip()
            # prediction extraction
            pred = extract_answer(text)
            # correctness
            correct = False
            try:
                correct = verify_answer(gold, pred)
            except:
                pass
            # reasoning length (entire response) in tokens
            reasoning_length = len(tokenizer.encode(text, add_special_tokens=False))

            runs[rid].append({
                "question":         ds[idx][question_key],
                "full_response":    text,
                "reasoning_length": reasoning_length,
                "prediction":       pred,
                "gold":             gold,
                "correct":          correct
            })

    summary = {"runs": [], "aggregate": {}}
    accs = []
    lengths = []
    for run_id, recs in runs.items():
        # accuracy per run
        acc = sum(1 for r in recs if r["correct"]) / len(recs) * 100
        accs.append(acc)
        # average reasoning length per run
        avg_len = sum(r["reasoning_length"] for r in recs) / len(recs)
        lengths.append(avg_len)
        summary["runs"].append({
            "run_id":         run_id,
            "accuracy":       acc,
            "avg_length":     avg_len,
            "records":        recs
        })

    summary["aggregate"]["mean_accuracy"]    = float(np.mean(accs))
    summary["aggregate"]["std_accuracy"]     = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    summary["aggregate"]["mean_length"]      = float(np.mean(lengths))
    summary["aggregate"]["std_length"]       = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0

    # 8) Save JSON
    output_path = f"normal_reasoning_traces/{args.dataset}/{args.model}/summary_{args.n_sample}runs.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Per‐run accuracies: {accs}")
    print(f"Mean ± std accuracy: {summary['aggregate']['mean_accuracy']:.2f}% ± {summary['aggregate']['std_accuracy']:.2f}%")
    print(f"Mean ± std length: {summary['aggregate']['mean_length']:.1f} ± {summary['aggregate']['std_length']:.1f} tokens")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
