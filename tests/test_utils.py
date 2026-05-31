"""Tests for utils.py: answer extraction, scoring, seeding, sandboxed exec."""
import pytest

from utils import (
    auto_device,
    extract_boxed_only,
    extract_gold,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
    score_aime,
    score_gsm8k,
    set_seed,
)


class TestExtractGsm8k:
    def test_boxed_integer(self):
        assert extract_gsm8k_answer("answer: \\boxed{42}") == "42"

    def test_boxed_negative(self):
        assert extract_gsm8k_answer("\\boxed{-7}") == "-7"

    def test_boxed_decimal(self):
        assert extract_gsm8k_answer("\\boxed{3.14}") == "3.14"

    def test_multiple_boxed_returns_last(self):
        assert extract_gsm8k_answer("first \\boxed{1} then \\boxed{2}") == "2"

    def test_boxed_with_text_inside_returns_number(self):
        assert extract_gsm8k_answer("\\boxed{the answer is 5}") == "5"

    def test_boxed_with_no_number_returns_text(self):
        # Strict fallback to inner content
        assert extract_gsm8k_answer("\\boxed{abc}") == "abc"

    def test_no_box_falls_back_to_last_number(self):
        assert extract_gsm8k_answer("step 1: 10. step 2: 20. final: 30") == "30"

    def test_completely_empty_returns_none(self):
        assert extract_gsm8k_answer("no numbers here") is None
        assert extract_gsm8k_answer("") is None


class TestExtractBoxedOnly:
    def test_boxed_returns_integer(self):
        assert extract_boxed_only("\\boxed{42}") == "42"

    def test_no_box_returns_none(self):
        # Strict — no fallback to last number
        assert extract_boxed_only("the final answer is 42") is None

    def test_no_box_with_numbers_returns_none(self):
        assert extract_boxed_only("step 1: 10. step 2: 20.") is None

    def test_empty_returns_none(self):
        assert extract_boxed_only("") is None


class TestExtractGold:
    def test_gsm8k_format(self):
        assert extract_gold("Long explanation #### 42") == "42"

    def test_negative(self):
        assert extract_gold("#### -7") == "-7"

    def test_decimal(self):
        assert extract_gold("#### 3.14") == "3.14"

    def test_missing_marker(self):
        assert extract_gold("no marker here") is None


class TestNormalizeAnswer:
    def test_strips_and_lowers(self):
        assert normalize_answer("  Hello  ") == "hello"

    def test_none_passthrough(self):
        assert normalize_answer(None) is None

    def test_already_clean(self):
        assert normalize_answer("42") == "42"


class TestScoreAime:
    def test_correct(self):
        ok, pred, err = score_aime("\\boxed{42}", "42")
        assert ok is True
        assert pred == "42"
        assert err is None

    def test_wrong(self):
        ok, pred, err = score_aime("\\boxed{99}", "42")
        assert ok is False
        assert pred == "99"
        assert err is None

    def test_no_box_no_credit(self):
        # The whole point of the AIME fix — last-number fallback would have
        # picked up the 42 in the explanation, but we want strict \boxed.
        ok, pred, err = score_aime("the answer is 42", "42")
        assert ok is False
        assert pred is None
        assert "No \\boxed{}" in err

    def test_gold_unparseable(self):
        ok, pred, err = score_aime("\\boxed{42}", "abc")
        assert ok is False
        assert err is not None and "parse" in err.lower()

    def test_none_text(self):
        ok, pred, err = score_aime(None, "42")
        assert ok is False
        assert pred is None


class TestScoreGsm8k:
    def test_correct_boxed(self):
        ok, pred, err = score_gsm8k("\\boxed{42}", "42")
        assert ok is True
        assert pred == "42"
        assert err is None

    def test_correct_last_number_fallback(self):
        ok, pred, _ = score_gsm8k("after computing: 42", "42")
        assert ok is True
        assert pred == "42"

    def test_wrong(self):
        ok, _, _ = score_gsm8k("\\boxed{99}", "42")
        assert ok is False

    def test_empty_gold(self):
        ok, _, _ = score_gsm8k("\\boxed{42}", "")
        assert ok is False


class TestExtractPython:
    def test_simple_block(self):
        text = "explanation\n```python\nprint(1)\n```\nmore"
        assert extract_markdown_python_block(text) == "print(1)"

    def test_no_block_returns_none(self):
        assert extract_markdown_python_block("plain text") is None

    def test_multiple_blocks_returns_last(self):
        text = "```python\nx=1\n```\nthen\n```python\nx=2\n```"
        assert extract_markdown_python_block(text) == "x=2"


class TestSetSeed:
    def test_reproducible_random(self):
        import random
        set_seed(123)
        r1 = random.random()
        set_seed(123)
        r2 = random.random()
        assert r1 == r2


class TestAutoDevice:
    def test_explicit_cpu(self):
        import torch
        assert auto_device("cpu") == torch.device("cpu")

    def test_none_falls_back(self):
        import torch
        d = auto_device(None)
        assert d.type in ("cpu", "cuda")


class TestRunWithTimeout:
    def test_success(self):
        ok, err = run_with_timeout("x = 1 + 1", timeout=5)
        assert ok is True
        assert err is None

    def test_exception(self):
        ok, err = run_with_timeout("raise RuntimeError('boom')", timeout=5)
        assert ok is False
        assert "RuntimeError" in err

    def test_timeout(self):
        ok, err = run_with_timeout("while True: pass", timeout=1)
        assert ok is False
        assert "Timeout" in err
