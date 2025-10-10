import os
import json
import re
import math
import numpy as np
from math_verify import parse, verify, LatexExtractionConfig

DATASET_MAP = {
    "gpqa": {
        "args": ("hendrydong/gpqa_diamond_mc", "test"),
        "question_key": "problem",
        "answer_key": "solution"
    },
    "MATH-500": {
        "args": ("HuggingFaceH4/MATH-500", "test"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "AIME2024": {
        "args": ("HuggingFaceH4/aime_2024", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "gsm8k": {
        "args": ("skrishna/gsm8k_only_answer", "test"),
        "question_key": "text",
        "answer_key": "label"
    },
    "openr1-math": {
        "args": ("open-r1/OpenR1-Math-220k", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "openr1-math-10k": {
        "args": ("openr1-math-10k", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "openr1-math-2k-rl-hard": {
        "args": ("openr1-math-2k-rl-hard", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
}
MODEL_MAP   = {
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",

    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",

    "qwq-32b": "Qwen/QwQ-32B",

    "ReFIne-qwen3-1.7b-sft-only": "../train/ReFIne-qwen3-1.7b-sft-only",
    "ReFIne-qwen3-4b-sft-only": "../train/ReFIne-qwen3-4b-sft-only",
    "ReFIne-qwen3-8b-sft-only": "../train/ReFIne-qwen3-8b-sft-only",

    "ReFIne-qwen3-1.7b": "../train/ReFIne-qwen3-1.7b",
    "ReFIne-qwen3-4b": "../train/ReFIne-qwen3-4b",
    "ReFIne-qwen3-8b": "../train/ReFIne-qwen3-8b",

    "Plain-qwen3-1.7b-sft-only": "../train/Plain-qwen3-1.7b-sft-only",
    "Plain-qwen3-4b-sft-only": "../train/Plain-qwen3-4b-sft-only",
    "Plain-qwen3-8b-sft-only": "../train/Plain-qwen3-8b-sft-only",

    "Plain-qwen3-1.7b": "../train/Plain-qwen3-1.7b",
    "Plain-qwen3-4b": "../train/Plain-qwen3-4b",
    "Plain-qwen3-8b": "../train/Plain-qwen3-8b",
}

def verify_answer(pred: str, ref: str) -> bool:

    # ── patterns & threshold ─────────────────────────────────────────────────
    BASE_N_RE    = re.compile(r"^\(?([0-9A-Za-z]+)\)?_\{(\d+)\}$")
    EXP_RE       = re.compile(r"\^\{(\d+)\}")
    MAX_SAFE_EXP = 10_000

    # ── normalize inputs ─────────────────────────────────────────────────────
    if pred is None or ref is None:
        return False
    p = pred.strip()
    r = ref.strip()

    # ── 1) base-N literal in prediction ─────────────────────────────────────
    m = BASE_N_RE.match(p)
    if m:
        return m.group(1) == r

    # ── 2) base-N literal in reference ──────────────────────────────────────
    m = BASE_N_RE.match(r)
    if m:
        return m.group(1) == p

    # ── 3) huge-exponent guard ───────────────────────────────────────────────
    exps = [int(e) for e in EXP_RE.findall(p)]
    if exps and max(exps) > MAX_SAFE_EXP:
        return p.replace(" ", "") == r.replace(" ", "")

    # ── 4) fallback to math_verify ──────────────────────────────────────────
    wrap = lambda s: f"\\({s}\\)"
    cfg  = LatexExtractionConfig()
    try:
        g_node = parse(wrap(r), extraction_config=[cfg])
        p_node = parse(wrap(p), extraction_config=[cfg])
        return verify(g_node, p_node, float_rounding=2)
    except Exception:
        return False

def extract_answer(text):
    if text is None:
        return None
    # Step 1: Remove everything that is not a number, letter, ".", or "-"
    # text = re.sub(r'[^0-9a-zA-Z{}\\.\-]', '', text)
    # Try extracting from 'boxed' first
    boxed_matches = extract_boxed(text)
    if boxed_matches:
        extracted_answer = boxed_matches[-1][1:-1]
        return extracted_answer

    # Fallback: extract any numbers
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if not numbers:
        return None

    try:
        extracted_number = float(numbers[-1])
        # Guard against infinity
        if math.isinf(extracted_number):
            return None
        
        return numbers[-1]
    except (ValueError, OverflowError):
        return None

def extract_boxed(text):
    pattern = re.compile(r'boxed\{')
    matches = []
    stack = []
    
    i = 0
    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break
        
        start = match.end() - 1  # Position at the first `{`
        stack.append(start)
        i = start + 1
        count = 1  # To track `{}` pairs
        
        while i < len(text) and stack:
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:  # Found a matching closing `}`
                    start = stack.pop()
                    matches.append(text[start:i+1])
                    break
            i += 1
    
    return matches