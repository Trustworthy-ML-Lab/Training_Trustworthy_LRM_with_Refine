import os
import re
import json
import argparse
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from utils import extract_answer, verify_answer, DATASET_MAP, MODEL_MAP
from roles import (
    get_interpreter_prompt,
    get_extractor_prompt,
    get_strategist_prompt,
    get_solver_prompt,
    get_confidence_prompt,
)

# ─────────────────────────────────────────────
# Stage order and budgets
# ─────────────────────────────────────────────
GENERAL_BUDGET        = 1536
SOLVER_BUDGET         = 40960 - 1024 - 1536 * 4

# We'll handle "interpreter" separately (with n_sample > 1), then
# process the remaining stages with n=1.
REMAINING_STAGES = [
    ("extractor",  get_extractor_prompt,  GENERAL_BUDGET),
    ("strategist", get_strategist_prompt, GENERAL_BUDGET),
    ("solver",     get_solver_prompt,     SOLVER_BUDGET),
    ("confidence", get_confidence_prompt, GENERAL_BUDGET),
]

# ─────────────────────────────────────────────
# Helper: apply chat template for vLLM
# ─────────────────────────────────────────────
def apply_chat(prompt: str, tokenizer, enable_thinking: bool = False):
    """
    Wraps a single-user message into the chat template expected by the model.
    If enable_thinking=True, the model will generate inside <think>…</think>.
    """
    conv = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

# ─────────────────────────────────────────────
# Main evaluation (single-pass, n_sample)
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Structured reasoning eval on OpenR1-Math with vLLM (single pass, n_sample)"
    )
    parser.add_argument(
        "--dataset", choices=DATASET_MAP.keys(), default="openr1-math-10k"
    )
    parser.add_argument(
        "--model", choices=MODEL_MAP.keys(), default="qwen3-8b"
    )
    parser.add_argument(
        "--n_sample", type=int, default=1,
        help="Number of samples per example at the interpreter stage"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1
    )
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

    # 3) Prepare vLLM sampling params
    temp  = getattr(cfg, "temperature", None)
    top_k = getattr(cfg, "top_k",      None)
    top_p = getattr(cfg, "top_p",      None)

    def make_params(n, max_tokens):
        kw = {"n": n, "max_tokens": max_tokens}
        if temp  is not None: kw["temperature"] = temp
        if top_k is not None: kw["top_k"] = top_k
        if top_p is not None: kw["top_p"] = top_p
        return SamplingParams(**kw)

    # Interpreter: n_sample >1
    interp_params = make_params(args.n_sample, GENERAL_BUDGET)
    # Remaining stages: n=1
    stage_params = {
        name: make_params(1, budget)
        for name, _, budget in REMAINING_STAGES
    }

    # 4) Initialize vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_pos,
        dtype=torch.bfloat16,
    )

    # 5) Stage 0: Interpreter (batched n_sample)
    print("Stage: interpreter")
    interp_prompts = [
        apply_chat(
            get_interpreter_prompt(ex[question_key].strip(), ""),
            tokenizer,
            enable_thinking=False
        )
        for ex in ds
    ]
    interp_outputs = llm.generate(
        prompts=interp_prompts,
        sampling_params=interp_params
    )

    # 6) Flatten into records: each example → n_sample records
    records = []
    for i, ex in enumerate(ds):
        gold = ex[answer_key].strip()
        q    = ex[question_key].strip()
        for sid, out in enumerate(interp_outputs[i].outputs):
            hist = out.text.strip()
            records.append({
                "example_idx":      i,
                "sample_idx":       sid,
                "question":         q,
                "gold":             gold,
                "history":          hist,
                "prediction":       None,
                "correct":          False,
                "confidence":       None,
                "reasoning_length": None,
            })

    # 7) Remaining stages (extractor, strategist, solver)
    for stage_name, prompt_fn, _ in REMAINING_STAGES:
        print(f"Stage: {stage_name}")
        enable_thinking = (stage_name == "solver")

        prompts = []
        for record in records:
            raw_prompt = prompt_fn(record["question"], record["history"])
            chat_input = apply_chat(raw_prompt, tokenizer, enable_thinking=enable_thinking)
            prompts.append(chat_input)

        outs = llm.generate(prompts=prompts, sampling_params=stage_params[stage_name])

        for idx, out in enumerate(outs):
            text = out.outputs[0].text.strip()

            if stage_name == "solver":
                # Extract <think>…</think> and wrap post-think in <final_answer>
                m = re.search(r"(</think>)(.*)", text, re.DOTALL)
                if m:
                    think_block = text[: m.end(1)].strip()
                    answer_text = m.group(2).strip()
                else:
                    think_block = ""
                    answer_text = text

                final_block = f"<final_answer>\n{answer_text}\n</final_answer>"
                combined = (think_block + "\n\n" + final_block).strip()

                records[idx]["history"] += "\n\n" + combined
                pred = extract_answer(records[idx]["history"])
                records[idx]["prediction"] = pred

            elif stage_name == "confidence":
                # append confidence block to history
                records[idx]["history"] += "\n\n" + text
                # extract numeric score and store
                m = re.search(r"Confidence:\s*(\d{1,2})/10", text)
                if m:
                    records[idx]["confidence"] = int(m.group(1))
                else:
                    records[idx]["confidence"] = None

            else:
                records[idx]["history"] += "\n\n" + text

    # ─────────────────────────────────────────────
    # 8) Save checkpoint before verification
    # ─────────────────────────────────────────────
    output_dir = f"structural_reasoning_traces/{args.dataset}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"pre_verify_checkpoint_{args.n_sample}runs.json")
    with open(checkpoint_path, "w", encoding="utf-8") as cp_file:
        json.dump({"records": records}, cp_file, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved to {checkpoint_path} before verification.")

    # 9) Verify predictions & compute reasoning lengths
    for record in records:
        pred = record["prediction"]
        gold = record["gold"]
        record["correct"] = verify_answer(pred, gold)

        split_marker = "\n\n<final_answer>"
        full_resp = record["history"]
        if split_marker in full_resp:
            reasoning_text = full_resp.split(split_marker, 1)[0]
        else:
            reasoning_text = full_resp
        record["reasoning_length"] = len(
            tokenizer.encode(reasoning_text, add_special_tokens=False)
        )

    # 10) Aggregate results across runs
    runs = {run_id: [] for run_id in range(args.n_sample)}
    for r in records:
        run_id = r["sample_idx"]
        runs[run_id].append({
            "question":         r["question"],
            "full_response":    r["history"],
            "reasoning_length": r["reasoning_length"],
            "prediction":       r["prediction"],
            "gold":             r["gold"],
            "correct":          r["correct"],
            "confidence":       r["confidence"],
        })

    summary = {"runs": [], "aggregate": {}}
    accs, lengths, confs = [], [], []
    for run_id, recs in runs.items():
        acc = sum(r["correct"] for r in recs) / len(recs) * 100
        avg_len = sum(r["reasoning_length"] for r in recs) / len(recs)
        vals    = [r["confidence"] for r in recs if r["confidence"] is not None]
        avg_conf= float(np.mean(vals)) if vals else None
        accs.append(acc)
        lengths.append(avg_len)
        if avg_conf is not None: confs.append(avg_conf)
        summary["runs"].append({
            "run_id":     run_id,
            "accuracy":   acc,
            "avg_length": avg_len,
            "avg_confidence": avg_conf,
            "records":    recs,
        })

    summary["aggregate"]["mean_accuracy"] = float(np.mean(accs))
    summary["aggregate"]["std_accuracy"]  = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    summary["aggregate"]["mean_length"]   = float(np.mean(lengths))
    summary["aggregate"]["std_length"]    = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
    summary["aggregate"]["mean_confidence"] = float(np.mean(confs)) if confs else None
    summary["aggregate"]["std_confidence"]  = float(np.std(confs, ddof=1)) if len(confs) > 1 else 0.0

    # 11) Write final results JSON
    final_output_path = os.path.join(output_dir, f"summary_{args.n_sample}runs.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 12) Print per-run and aggregate stats
    per_run_accs = [run["accuracy"] for run in summary["runs"]]
    print(f"Per-run accuracies: {per_run_accs}")
    print(f"Mean ± std accuracy: {summary['aggregate']['mean_accuracy']:.2f}% ± {summary['aggregate']['std_accuracy']:.2f}%")
    print(f"Mean ± std reasoning length: {summary['aggregate']['mean_length']:.1f} ± {summary['aggregate']['std_length']:.1f} tokens")
    mc = summary['aggregate']['mean_confidence']
    sc = summary['aggregate']['std_confidence']

    mc_str = f"{mc:.1f}" if mc is not None else "None"
    sc_str = f"{sc:.1f}" if sc is not None else "None"

    print(f"Mean ± std confidence: {mc_str} ± {sc_str}")
    print(f"Details written to {final_output_path}")


if __name__ == "__main__":
    main()
