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


class TestMinimalPersonaPrompts:
    def _args(self, minimal=False, task="gsm8k"):
        return argparse.Namespace(
            model_name="Qwen/Qwen3-4B", task=task, think=False,
            text_mas_context_length=-1,
            concise_nonjudger_prompt=False,
            minimal_persona_prompts=minimal,
        )

    def test_minimal_replaces_planner_prompt(self):
        args = self._args(minimal=True)
        msgs = build_agent_message_sequential_latent_mas(
            role="planner", question="What is 2+2?", method="latent_mas", args=args,
        )
        content = msgs[-1]["content"]
        assert "Solve this problem step by step" in content
        assert "What is 2+2?" in content
        # Verbose persona framing should be gone
        assert "Planner Agent" not in content

    def test_minimal_replaces_critic_prompt(self):
        args = self._args(minimal=True)
        msgs = build_agent_message_sequential_latent_mas(
            role="critic", question="Q", method="latent_mas", args=args,
        )
        content = msgs[-1]["content"]
        assert "Solve this problem step by step" in content
        assert "Critic Agent" not in content

    def test_minimal_replaces_refiner_prompt(self):
        args = self._args(minimal=True)
        msgs = build_agent_message_sequential_latent_mas(
            role="refiner", question="Q", method="latent_mas", args=args,
        )
        content = msgs[-1]["content"]
        assert "Solve this problem step by step" in content
        assert "Refiner Agent" not in content

    def test_minimal_does_not_touch_judger(self):
        args = self._args(minimal=True)
        msgs = build_agent_message_sequential_latent_mas(
            role="judger", question="Q", method="latent_mas", args=args,
        )
        content = msgs[-1]["content"]
        # Judger prompt is unchanged — should still have the verbose framing
        assert "Solve this problem step by step" not in content
        assert "\\boxed" in content

    def test_minimal_off_keeps_persona_prompts(self):
        args = self._args(minimal=False)
        msgs = build_agent_message_sequential_latent_mas(
            role="planner", question="Q", method="latent_mas", args=args,
        )
        content = msgs[-1]["content"]
        # Original verbose framing should be present
        assert "Planner Agent" in content


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


class TestSoftConcisePipelineSuffix:
    def _args(self, **kw):
        base = dict(model_name="Qwen/Qwen2.5-Math-7B-Instruct", task="math500", think=False,
                    text_mas_context_length=-1, concise_nonjudger_prompt=False,
                    concise_pipeline_prompt=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_soft_suffix_on_compute_when_set(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="compute", question="Q", method="latent_mas",
            args=self._args(concise_pipeline_prompt=True))
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" in joined and "single short sentence" not in joined

    def test_soft_suffix_not_on_verify(self):
        # verify is the text-producer — must keep its full prompt
        msgs = build_agent_message_sequential_latent_mas(
            role="verify", question="Q", method="latent_mas",
            args=self._args(concise_pipeline_prompt=True))
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" not in joined

    def test_soft_wins_when_both_flags_set(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="strategize", question="Q", method="latent_mas",
            args=self._args(concise_pipeline_prompt=True, concise_nonjudger_prompt=True))
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" in joined and "single short sentence" not in joined

    def test_off_by_default(self):
        msgs = build_agent_message_sequential_latent_mas(
            role="compute", question="Q", method="latent_mas", args=self._args())
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" not in joined


class TestComputeAsProducer:
    """compute as the final agent (2-persona strategize->compute DAG) must box the
    answer and be exempt from concision; as a middle agent it does neither."""
    def _args(self):
        # large context_length so ctx isn't truncated (default -1 drops the last char)
        return argparse.Namespace(
            model_name="Qwen/Qwen2.5-Math-7B-Instruct", task="math500", think=False,
            text_mas_context_length=100000, concise_pipeline_prompt=True,
            concise_nonjudger_prompt=False, minimal_persona_prompts=False)

    def test_compute_boxes_when_producer(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="compute", question="Q", context="strat", method="text_mas",
            args=self._args(), is_producer=True)
        joined = " ".join(m["content"] for m in msgs)
        assert "YOUR_FINAL_ANSWER" in joined  # boxing instruction added

    def test_compute_no_box_when_middle(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="compute", question="Q", context="strat", method="text_mas",
            args=self._args(), is_producer=False)
        joined = " ".join(m["content"] for m in msgs)
        assert "YOUR_FINAL_ANSWER" not in joined

    def test_producer_exempt_from_concision(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="compute", question="Q", context="strat", method="text_mas",
            args=self._args(), is_producer=True)
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" not in joined  # producer keeps full prompt

    def test_nonproducer_strategize_still_concised(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="strategize", question="Q", context="", method="text_mas",
            args=self._args(), is_producer=False)
        joined = " ".join(m["content"] for m in msgs)
        assert "essential steps" in joined

    def test_compute_standalone_when_no_prior_context(self):
        # compute->verify DAG: compute is first, no strategy precedes it
        msgs = build_agent_messages_sequential_text_mas(
            role="compute", question="Q", context="", method="text_mas",
            args=self._args(), is_producer=False)
        joined = " ".join(m["content"] for m in msgs)
        assert "Strategy from the previous agent" not in joined  # no dangling empty strategy

    def test_compute_executes_strategy_when_context_present(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="compute", question="Q", context="[Strategist]: do X", method="text_mas",
            args=self._args(), is_producer=False)
        joined = " ".join(m["content"] for m in msgs)
        assert "Strategy from the previous agent" in joined and "do X" in joined


class TestTextMasSequentialPrompts:
    def test_returns_messages(self):
        msgs = build_agent_messages_sequential_text_mas(
            role="planner", question="Q", context="", method="text_mas", args=_args(),
        )
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
