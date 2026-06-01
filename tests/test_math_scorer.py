"""Tests for the MATH-500 scorer (utils.score_math + last_boxed_only_string).

score_math uses math-verify for LaTeX equivalence, so these assert real
equivalence behavior (1/2 == 0.5 == \\frac{1}{2}), brace-balanced boxed
extraction, and that non-equivalent answers are rejected. Pure-CPU, no model.
"""
import pytest

from utils import last_boxed_only_string, score_math


def test_last_boxed_balances_nested_braces():
    assert last_boxed_only_string(r"x \boxed{\frac{1}{2}} y") == r"\boxed{\frac{1}{2}}"
    assert last_boxed_only_string(r"\boxed{\frac{a}{b+\frac{c}{d}}}") == r"\boxed{\frac{a}{b+\frac{c}{d}}}"
    # last box wins
    assert last_boxed_only_string(r"\boxed{1} then \boxed{2}") == r"\boxed{2}"
    assert last_boxed_only_string("no box here") is None
    assert last_boxed_only_string("") is None


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        (r"answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}", True),
        (r"\boxed{0.5}", r"\frac{1}{2}", True),       # decimal == fraction
        (r"\boxed{1/2}", r"\frac{1}{2}", True),       # slash == frac
        (r"\boxed{16}", "16", True),
        (r"\boxed{\sqrt{2}}", r"\sqrt{2}", True),
        (r"\boxed{\frac{\pi}{4}}", r"\frac{\pi}{4}", True),
        (r"nested \boxed{\frac{1}{2}} stuff", "0.5", True),
        (r"\boxed{(1,2)}", "(1, 2)", True),
        (r"\boxed{2}", r"\sqrt{2}", False),           # 2 != sqrt(2)
        (r"\boxed{17}", "16", False),
    ],
)
def test_score_math_equivalence(pred, gold, expected):
    ok, _, err = score_math(pred, gold)
    assert ok is expected, f"pred={pred!r} gold={gold!r} err={err}"


def test_score_math_empty_gold_is_safe():
    ok, _, err = score_math(r"\boxed{1}", "")
    assert ok is False and err is not None
