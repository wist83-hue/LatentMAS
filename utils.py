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


def last_boxed_only_string(text: str) -> Optional[str]:
    """Return the LAST '\\boxed{...}' (or '\\fbox{...}') substring, brace-balanced.

    Unlike the regex `\\boxed\\{([^}]*)\\}` used for GSM8K/AIME, this handles
    NESTED braces (e.g. \\boxed{\\frac{1}{2}}) by scanning for the matching close
    brace. Returns the full '\\boxed{...}' string (math-verify parses it), or None.
    """
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i, depth, close = idx, 0, None
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                close = i
                break
        i += 1
    return text[idx:close + 1] if close is not None else None


def score_math(pred_text: Optional[str], gold: str):
    """Return (ok, pred_norm, error_msg) for MATH-500-style LaTeX answers.

    Uses math-verify for equivalence (handles fractions, radicals, intervals,
    sets, etc.). Extracts the model's last \\boxed{} first (brace-balanced), then
    falls back to letting math-verify parse the whole output.
    """
    from math_verify import parse, verify
    gold_s = str(gold or "").strip()
    if not gold_s:
        return False, None, "empty gold"
    try:
        # Gold: parse as LaTeX (wrap in $...$ so bare LaTeX like \frac{1}{2} parses).
        gold_parsed = parse(f"${gold_s}$") or parse(gold_s)
        boxed = last_boxed_only_string(pred_text or "")
        pred_src = boxed if boxed is not None else (pred_text or "")
        pred_parsed = parse(pred_src)
        if not pred_parsed:
            return False, None, f"no parseable answer. Gold: {gold_s}"
        # math-verify's verify(gold, target) is order-sensitive: gold first.
        ok = bool(verify(gold_parsed, pred_parsed))
        return ok, (boxed or pred_src)[:200], None
    except Exception as e:  # never let a scorer exception abort a run
        return False, None, f"math-verify error: {e}"


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

