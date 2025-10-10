#!/usr/bin/env python
# train_grpo.py

import re
import os
import argparse
import torch

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)
from trl import GRPOTrainer, GRPOConfig

from utils import verify_answer, extract_answer, DATASET_MAP, MODEL_MAP

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO on datasets/models from your maps with ZeRO-3"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openr1-math-2k-rl-hard",
        help="Dataset key from DATASET_MAP",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ReFIne-qwen3-8b-sft-only",
        help="Model key from MODEL_MAP",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--target_kl",     type=float, default=0.0)
    parser.add_argument("--max_new_tokens",type=int,   default=16384+4096)
    return parser.parse_args()

def reward_fn(
    completions,        # List[List[dict]] of vLLM chat outputs
    **kwargs            # Will include your "gold" column, etc.
):
    """
    Custom GRPO reward function.

    - Extracts the assistant's reply from each chat-style completion.
    - Verifies correctness against the gold answers in kwargs["gold"].
    - Adds bonuses for correct tag structure and <think> mentions.
    """
    # 1) Pull out the flat text responses
    contents = completions

    # 2) Grab gold answers from kwargs (your dataset column must be named "gold")
    golds = kwargs.get("gold", [None] * len(contents))

    rewards = []
    for text, gold in zip(contents, golds):
        # a) Extract boxed final answer (or fallback to full text)
        try:
            pred = extract_answer(text)
        except Exception:
            pred = None

        # b) Correctness reward
        try:
            corr = 1.0 if verify_answer(pred, gold) else 0.0
        except Exception:
            corr = 0.0

        # c) Tag-structure bonus
        expected = [
            "<understanding>", "</understanding>",
            "<facts>",         "</facts>",
            "<plan>",          "</plan>",
            "<think>",         "</think>",
            "<final_answer>",  "</final_answer>",
            "<self_assessment>",    "</self_assessment>",
        ]
        pos, struct_ok = 0, False
        for tag in expected:
            idx = text.find(tag, pos)
            if idx < 0:
                break
            pos = idx + len(tag)
        else:
            struct_ok = True

        # d) <think> block mention bonus
        refer = 0.0
        low = text.lower()
        if "<think>" in low and "</think>" in low:
            body = low.split("<think>", 1)[1].split("</think>", 1)[0]
            inner_tags = ["<understanding>", "<facts>", "<plan>"]
            for t in inner_tags:
                if t in body:
                    refer += 1.0 / 3.0
    
        # e) Confidence-based reward: normalized Brier + high-confidence penalty
        m_conf = re.search(r"Confidence:\s*(\d{1,2})/10", text)
        if m_conf:
            score = float(m_conf.group(1))
            missing_conf_pen = 0.0
        else:
            score = 5.0
            missing_conf_pen = -1.0

        # Map score to probability p in [0,1]
        p = max(0.0, min(score / 10.0, 1.0))

        # Normalized Brier score in [0,1]
        brier_norm = 1.0 - (p - corr)**2

        # Combined confidence reward
        conf_reward = brier_norm + missing_conf_pen

        # f) Combine into final reward
        reward = (corr + struct_ok + refer + conf_reward) / 4.0
        rewards.append(reward)

        # <<< DEBUG PRINT EVERYTHING >>>
        # print("===== reward_fn DEBUG =====")
        # print(f" CONTENT   =\n{text}\n") 
        # print(f" GOLD      = {gold!r}")
        # print(f" PRED      = {pred!r}")
        # print(f" CORRECT?  = {corr}")
        # print(f" STRUCT_OK = {struct_ok}")
        # print(f" REFER = {refer}")
        # print(f" CONF = { conf_reward}")
        # print(f" FINAL REWARD = {reward}")
        # print("============================")

    return rewards


def build_prompts(batch, question_key, answer_key, tokenizer):
    return {
        "prompt": [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for q in batch[question_key]
        ],
        "gold": batch[answer_key]
    }

def main():
    args = parse_args()

    # 1) Load dataset via DATASET_MAP
    ds_info = DATASET_MAP[args.dataset]
    ds_args = ds_info["args"]
    ds_raw = load_from_disk(ds_args[0])
    question_key = ds_info["question_key"]
    answer_key   = ds_info["answer_key"]

    # 2) Load tokenizer
    model_id  = MODEL_MAP[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3) Prepare prompts
    ds = ds_raw.map(
        lambda batch: build_prompts(batch, question_key, answer_key, tokenizer),
        batched=True,
        remove_columns=ds_raw.column_names,
    )

    gen_cfg = GenerationConfig.from_pretrained(model_id)

    # 1) Build your GRPOConfig (no top_k/top_p here)
    cfg = GRPOConfig(
        # ─── data‐parallel sizing ────────────────────────────────────────
        per_device_train_batch_size    = 1,
        gradient_accumulation_steps    = 1,

        # ─── training schedule ─────────────────────────────────────────
        num_train_epochs               = 1,
        learning_rate                  = args.learning_rate,
        logging_steps                  = 1,

        # ─── rollout / generation ──────────────────────────────────────
        num_generations                = 4,
        max_completion_length          = args.max_new_tokens,
        temperature                    = gen_cfg.temperature,
        top_k                          = gen_cfg.top_k,
        top_p                          = gen_cfg.top_p,

        # ─── KL penalty ────────────────────────────────────────────────
        beta                           = args.target_kl,

        # ─── vLLM rollout (optional) ──────────────────────────────────
        use_vllm                       = True,
        vllm_mode                      = "colocate",
        vllm_tensor_parallel_size      = 8,
        vllm_gpu_memory_utilization    = 0.3,

        bf16=True,               # <-- tell HF/TRL to run in bfloat16
        bf16_full_eval=True,     # (optional) eval in bf16 too
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        logging_dir = f"./trainer_logs/{args.model}",

        output_dir         = f"./checkpoints/{args.model}",  # where to dump checkpoints
        save_strategy      = "steps",                        # checkpoint by step
        save_steps         = 100,                            # every 100 updates
        save_total_limit   = 1,
    )

    # 3) Instantiate the trainer telling it to use vLLM
    trainer = GRPOTrainer(
        model         = MODEL_MAP[args.model],   # or your loaded `policy` object
        reward_funcs  = reward_fn,               # your function: List[str] → List[float]
        args          = cfg,                     # the GRPOConfig you just built
        train_dataset = ds,                      # your mapped HF Dataset
    )

    # 8) Train & save
    trainer.train()
    out_dir = model_id.replace('-sft-only', '')
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
