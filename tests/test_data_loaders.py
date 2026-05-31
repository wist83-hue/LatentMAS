"""Tests for data.py loaders — focus on bounds-check fixes.

We don't actually call HF datasets (would require network + cache); we patch
load_dataset to return controlled fakes and verify loader behavior.
"""
from unittest.mock import patch
import pytest


class _FakeDataset:
    """Minimal iterable-of-dicts stub for HF datasets."""
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


@patch("data.load_dataset")
class TestLoadMbppPlus:
    def test_handles_short_test_list(self, mock_ld):
        from data import load_mbppplus
        mock_ld.return_value = _FakeDataset([
            {"prompt": "do X", "test_list": ["assert f(1)==1"], "test": "assert f(2)==2"},
        ])
        items = list(load_mbppplus())
        assert len(items) == 1
        assert "assert f(1)==1" in items[0]["question"]

    def test_handles_empty_test_list(self, mock_ld):
        from data import load_mbppplus
        mock_ld.return_value = _FakeDataset([
            {"prompt": "do Y", "test_list": [], "test": "assert f()"},
        ])
        items = list(load_mbppplus())
        assert len(items) == 1
        assert "no example tests provided" in items[0]["question"]


@patch("data.load_dataset")
class TestLoadMedqa:
    def test_unmatched_answer_handled(self, mock_ld):
        from data import load_medqa
        # raw_answer doesn't match any option -> answer stays "" (not NameError)
        mock_ld.return_value = _FakeDataset([
            {"query": "Q?", "answer": "Nonexistent", "options": ["A1", "B1", "C1", "D1"]},
        ])
        items = list(load_medqa())
        assert len(items) == 1
        assert items[0]["solution"] == ""

    def test_matched_answer_returns_letter(self, mock_ld):
        from data import load_medqa
        mock_ld.return_value = _FakeDataset([
            {"query": "Q?", "answer": "match-here", "options": ["A", "B match-here", "C", "D"]},
        ])
        items = list(load_medqa())
        assert items[0]["solution"] == "b"

    def test_env_path_override(self, mock_ld, monkeypatch):
        from data import load_medqa
        monkeypatch.setenv("MEDQA_PATH", "/custom/path/q.json")
        mock_ld.return_value = _FakeDataset([])
        list(load_medqa())
        # Verify load_dataset was called with the env-provided path
        call_kwargs = mock_ld.call_args.kwargs
        assert call_kwargs.get("data_files") == "/custom/path/q.json"


@patch("data.load_dataset")
class TestLoadGsm8k:
    def test_basic(self, mock_ld):
        from data import load_gsm8k
        mock_ld.return_value = _FakeDataset([
            {"question": "  Q  ", "answer": "Reasoning #### 42"},
        ])
        items = list(load_gsm8k())
        assert items[0]["question"] == "Q"
        assert items[0]["gold"] == "42"


@patch("data.load_dataset")
class TestLoadAime:
    def test_2024(self, mock_ld):
        from data import load_aime2024
        mock_ld.return_value = _FakeDataset([{"problem": " P ", "answer": 42}])
        items = list(load_aime2024())
        assert items[0]["question"] == "P"
        assert items[0]["gold"] == "42"

    def test_2025(self, mock_ld):
        from data import load_aime2025
        mock_ld.return_value = _FakeDataset([{"problem": " P ", "answer": "  7 "}])
        items = list(load_aime2025())
        assert items[0]["question"] == "P"
        assert items[0]["gold"] == "7"


@patch("data.load_dataset")
class TestLoadArc:
    def test_easy_label_mapping(self, mock_ld):
        from data import load_arc_easy
        mock_ld.return_value = _FakeDataset([{
            "question": "Q",
            "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]},
            "answerKey": "2",
        }])
        items = list(load_arc_easy())
        # Labels 1..4 -> a..d
        assert "a: a" in items[0]["question"]
        assert items[0]["gold"] == "b"

    def test_challenge_letter_labels(self, mock_ld):
        from data import load_arc_challenge
        mock_ld.return_value = _FakeDataset([{
            "question": "Q",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
            "answerKey": "C",
        }])
        items = list(load_arc_challenge())
        assert items[0]["gold"] == "c"
