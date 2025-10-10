# train_full_finetune_chat_multi_gpu.py

import os
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch
from utils import MODEL_MAP

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full‐parameter finetune on structural‐reasoning dataset with bf16 and multi‐GPU."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="openr1-math-10k-structural-reasoning-qwen3-8b/",
        help="Path to the HF dataset directory (e.g., filtered_dataset/)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="qwen3-8b",
        help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU core for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="If > 0, overrides num_train_epochs and runs this many steps total"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of full epochs to train for (ignored if max_train_steps > 0)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384+4096,
        help="Hard cutoff: if a combined prompt+response exceeds this, it will be truncated"
    )
    return parser.parse_args()


import torch

def collate_batch_for_causal_lm(batch_examples, pad_token_id):
    """
    batch_examples: a list of dicts, each with keys
       - "input_ids": List[int]
       - "attention_mask": List[int]
       - "labels": List[int]
    pad_token_id: tokenizer.pad_token_id

    Returns a dict of three tensors, each of shape (batch_size, max_len_in_batch):
       - input_ids:    LongTensor
       - attention_mask: LongTensor
       - labels:       LongTensor  (padded positions = -100)
    """

    # 1) Compute the length of each example, and the overall max length
    lengths = [len(ex["input_ids"]) for ex in batch_examples]
    max_len = max(lengths)

    # 2) Prepare empty lists to collect padded sequences
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for ex in batch_examples:
        seq_len = len(ex["input_ids"])
        pad_len = max_len - seq_len

        # 2a) Pad input_ids with pad_token_id on the right
        padded_ids = ex["input_ids"] + [pad_token_id] * pad_len
        batch_input_ids.append(padded_ids)

        # 2b) Pad attention_mask with 0 on the right
        padded_mask = ex["attention_mask"] + [0] * pad_len
        batch_attention_mask.append(padded_mask)

        # 2c) Pad labels with -100 on the right
        lbl = ex["labels"] + [-100] * pad_len
        batch_labels.append(lbl)

    # 3) Convert lists of lists into tensors
    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }



def main():
    args = parse_args()

    if args.base_model not in MODEL_MAP:
        raise ValueError(f"Model '{args.base_model}' not found in MODEL_MAP.")
    base_model = MODEL_MAP[args.base_model]
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Load dataset from disk
    # ────────────────────────────────────────────────────────────────────────────
    ds = load_from_disk(args.dataset_dir)

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Load tokenizer & model (chat format) with BF16
    # ────────────────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    # Make sure pad_token is defined
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Enable gradient checkpointing & disable KV cache
    # ────────────────────────────────────────────────────────────────────────────
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Tokenization function (no fixed padding!)
    # ────────────────────────────────────────────────────────────────────────────
    def tokenize_fn(example):
        # Build the chat‐style prompt string:
        chat_prompt: str = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        # Full text = prompt + “ground‐truth” + EOS
        combined_text = chat_prompt + example["full_response"] + tokenizer.eos_token

        # Tokenize just the prompt to find out how many tokens it used:
        prompt_ids = tokenizer(
            chat_prompt,
            return_attention_mask=False,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # Now tokenize the entire prompt+response. We only truncate if it EXCEEDS max_seq_length.
        # We do NOT pad here; let the collator pad it later.
        tokenized = tokenizer(
            combined_text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Build “labels” so that:
        #   - tokens in [0 : prompt_len] get label = -100 (we don’t predict prompt)
        #   - tokens in [prompt_len : len(input_ids)] get label = actual token ID
        labels = input_ids.copy()
        for i in range(len(labels)):
            if i < prompt_len:
                labels[i] = -100
            # otherwise keep the token ID.  (We’ll pad these labels in the collator.)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized_ds = ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=ds.column_names,
        num_proc=64,
    )
    tokenized_ds.reset_format()
    tokenized_ds.set_format(type="python")

    # ────────────────────────────────────────────────────────────────────────────
    # 5) Data collator
    # ────────────────────────────────────────────────────────────────────────────
    def my_collator(features):
        return collate_batch_for_causal_lm(features, tokenizer.pad_token_id)

    model_tag = args.base_model
    output_dir = f"ReFIne-{model_tag}-sft-only"

    # ────────────────────────────────────────────────────────────────────────────
    # 6) TrainingArguments & Trainer (DDP‐ready)
    # ────────────────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        logging_steps=20,
        optim="adamw_torch",
        max_steps=args.max_train_steps if args.max_train_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,          # keep “input_ids”/“attention_mask”/“labels”
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=my_collator,
    )

    # ────────────────────────────────────────────────────────────────────────────
    # 7) Launch training
    # ────────────────────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
