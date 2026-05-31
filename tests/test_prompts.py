"""Tests for prompts.py: prompt builders produce well-formed messages per role/task."""
import argparse
import pytest

from prompts import (
    build_agent_message_sequential_latent_mas,
    build_agent_messages_sequential_text_mas,
)


def _args(task="gsm8k"):
    return argparse.Namespace(
        model_name="Qwen/Qwen3-4B", task=task, think=False,
        text_mas_context_length=-1,
    )


class TestLatentMasSequentialPrompts:
    def test_planner(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="planner", question="What is 2+2?", method="latent_mas", args=_args(),
        )
        assert isinstance(msgs, list)
        assert any("Planner" in m["content"] for m in msgs)
        assert any("2+2" in m["content"] for m in msgs)

    def test_critic(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="critic", question="Q", method="latent_mas", args=_args(),
        )
        assert any("Critic" in m["content"] for m in msgs)

    def test_refiner(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="refiner", question="Q", method="latent_mas", args=_args(),
        )
        assert any("Refiner" in m["content"] for m in msgs)

    def test_judger_gsm8k_has_boxed_instruction(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="judger", question="Q", method="latent_mas", args=_args(task="gsm8k"),
        )
        assert any("\\boxed" in m["content"] for m in msgs)

    def test_judger_gpqa_includes_letter_choices(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="judger", question="Q", method="latent_mas", args=_args(task="gpqa"),
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "A,B,C,D" in joined

    def test_wrong_method_assertion(self):
        with pytest.raises(AssertionError, match="latent_mas"):
            build_agent_message_sequential_latent_mas(
                role="planner", question="Q", method="text_mas", args=_args(),
            )

    def test_non_qwen_assertion(self):
        bad = argparse.Namespace(model_name="meta-llama/Llama-3-8B", task="gsm8k", think=False)
        with pytest.raises(AssertionError, match="qwen"):
            build_agent_message_sequential_latent_mas(
                role="planner", question="Q", method="latent_mas", args=bad,
            )


class TestConciseSuffix:
    def test_appended_when_flag_set_for_nonjudger(self):
        args = argparse.Namespace(
            model_name="Qwen/Qwen3-4B", task="gsm8k", think=False,
            text_mas_context_length=-1,
            concise_nonjudger_prompt=True,
        )
        msgs = build_agent_message_sequential_latent_mas(
            role="planner", question="Q", method="latent_mas", args=args,
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "single short sentence" in joined

    def test_not_appended_for_judger(self):
        args = argparse.Namespace(
            model_name="Qwen/Qwen3-4B", task="gsm8k", think=False,
            text_mas_context_length=-1,
            concise_nonjudger_prompt=True,
        )
        msgs = build_agent_message_sequential_latent_mas(
            role="judger", question="Q", method="latent_mas", args=args,
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "single short sentence" not in joined

    def test_not_appended_when_flag_off(self):
        args = argparse.Namespace(
            model_name="Qwen/Qwen3-4B", task="gsm8k", think=False,
            text_mas_context_length=-1,
            concise_nonjudger_prompt=False,
        )
        msgs = build_agent_message_sequential_latent_mas(
            role="planner", question="Q", method="latent_mas", args=args,
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "single short sentence" not in joined


class TestTextMasSequentialPrompts:
    def test_returns_messages(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="planner", question="Q", context="", method="text_mas", args=_args(),
        )
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
