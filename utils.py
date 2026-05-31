import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_boxed_only(text: str) -> Optional[str]:
    """Strict variant: returns the contents of the LAST \\boxed{} or None.

    Intended for AIME-style tasks where small-integer answers often appear in
    intermediate reasoning; falling back to "last number" produces frequent
    false positives.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not boxes:
        return None
    content = boxes[-1]
    number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
    return number.group(0) if number else content.strip()


def score_aime(pred_text: Optional[str], gold: str):
    """Return (ok: bool, pred_norm: Optional[str], error_msg: Optional[str]) for AIME-style tasks.

    Uses extract_boxed_only (no fallback). Compares as integers.
    Centralizes logic previously duplicated across baseline/text_mas/latent_mas.
    """
    pred = normalize_answer(extract_boxed_only(pred_text or ""))
    gold_s = str(gold or "").strip()
    if pred is None:
        return False, None, f"No \\boxed{{}} answer found. Gold: {gold_s}"
    try:
        return (int(pred) == int(gold_s)), pred, None
    except (ValueError, TypeError):
        return False, pred, f"Could not parse as int. Pred: {pred}, Gold: {gold_s}"


def score_gsm8k(pred_text: Optional[str], gold: str):
    """Return (ok, pred_norm, error_msg) for GSM8K-style tasks.

    Uses extract_gsm8k_answer (boxed preferred, falls back to last number).
    Centralized for consistency.
    """
    pred = normalize_answer(extract_gsm8k_answer(pred_text or ""))
    gold_s = normalize_answer(gold)
    ok = (pred == gold_s) if (pred and gold_s) else False
    return ok, pred, None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

