from datasets import load_dataset
from utils import verify_answer, DATASET_MAP
from latex2sympy2_extended import latex2sympy
import sympy as sp

def passes_verify_answer(gold: str) -> bool:
    try:
        _ = verify_answer("0", gold)
        return True
    except:
        return False

def core_to_float(core: str) -> float:
    expr = latex2sympy(core)
    return float(expr)

def is_valid_answer(a: str) -> bool:
    if not a: return False
    core = a.strip()
    if not (0 < len(core) <= 10): return False
    if not passes_verify_answer(a): return False
    try:
        _ = core_to_float(core)
        return True
    except:
        return False

# 1) Load & filter
dataset_name, split = DATASET_MAP["openr1-math"]["args"]
ds = load_dataset(dataset_name, split=split)
ds = ds.filter(lambda ex: is_valid_answer(ex["answer"]))

# 2) Shuffle once
ds = ds.shuffle(seed=42)

# 3) Take subset
ds_sft = ds.select(range(10_000))
ds_rl = ds.select(range(10_000, 20_000))  

# 4) Keep only desired columns
try:
    ds_sft = ds_sft.select_columns(["problem", "answer"])
    ds_rl = ds_rl.select_columns(["problem", "answer"])
except Exception:
    ds_sft = ds_sft.remove_columns([c for c in ds_sft.column_names if c not in ["problem", "answer"]])
    ds_rl = ds_rl.remove_columns([c for c in ds_rl.column_names if c not in ["problem", "answer"]])

# 5) Save
ds_sft.save_to_disk("openr1-math-10k")
ds_rl.save_to_disk("openr1-math-10k-rl")
