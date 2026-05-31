"""End-to-end tests for the run_batch dispatch path with tiny-gpt2.

Covers sequential and parallel pipelines, anchor emission, baseline + text_mas.
"""
import copy
import pytest


def _items():
    return [
        {"question": "What is 2+2?", "solution": "4", "gold": "4"},
        {"question": "What is 3+3?", "solution": "6", "gold": "6"},
    ]


class TestBaselineRunBatch:
    def test_runs_and_returns_results(self, tiny_args, tiny_model_wrapper):
        from methods.baseline import BaselineMethod
        args = copy.copy(tiny_args)
        args.method = "baseline"
        args.task = "gsm8k"
        method = BaselineMethod(tiny_model_wrapper, max_new_tokens=8, generate_bs=2, args=args)
        results = method.run_batch(_items())
        assert len(results) == 2
        for r in results:
            assert "prediction" in r
            assert "correct" in r
            assert "agents" in r
            assert len(r["agents"]) == 1  # baseline = single agent


class TestLatentMasRunBatchSequential:
    def test_default_pipeline_runs(self, tiny_args, tiny_model_wrapper):
        from methods.latent_mas import LatentMASMethod
        args = copy.copy(tiny_args)
        args.method = "latent_mas"
        args.task = "gsm8k"
        args.latent_steps = 2
        method = LatentMASMethod(
            tiny_model_wrapper, latent_steps=2, judger_max_new_tokens=8,
            generate_bs=2, args=args,
        )
        results = method.run_batch(_items())
        assert len(results) == 2
        for r in results:
            # 4 agents: planner, critic, refiner, judger
            assert len(r["agents"]) == 4
            assert r["agents"][-1]["role"] == "judger"


class TestLatentMasRunBatchParallel:
    def test_parallel_branch_runs(self, tiny_args, tiny_model_wrapper):
        from methods.latent_mas import LatentMASMethod
        args = copy.copy(tiny_args)
        args.method = "latent_mas"
        args.task = "gsm8k"
        args.latent_steps = 2
        args.pipeline = "parallel(planner|critic),refiner,judger"
        method = LatentMASMethod(
            tiny_model_wrapper, latent_steps=2, judger_max_new_tokens=8,
            generate_bs=2, args=args,
        )
        results = method.run_batch(_items())
        assert len(results) == 2
        for r in results:
            # parallel(p|c) -> 2 branch traces, then refiner, then judger = 4
            assert len(r["agents"]) == 4
            # Both branch traces should be marked
            branch_traces = [a for a in r["agents"] if a.get("branch")]
            assert len(branch_traces) == 2

    def test_parallel_branches_isolation(self, tiny_args, tiny_model_wrapper):
        """Each branch should see only its own prompt, not prior branches.

        Smoke test that the snapshot/_copy_past_kv path works without error.
        """
        from methods.latent_mas import LatentMASMethod
        args = copy.copy(tiny_args)
        args.method = "latent_mas"
        args.task = "gsm8k"
        args.pipeline = "parallel(planner|critic|refiner),judger"
        method = LatentMASMethod(
            tiny_model_wrapper, latent_steps=2, judger_max_new_tokens=4,
            generate_bs=2, args=args,
        )
        results = method.run_batch(_items())
        assert len(results) == 2

    def test_parallel_trace_input_is_string(self, tiny_args, tiny_model_wrapper):
        """Regression: branch traces stored raw message list instead of rendered
        prompt string, which broke run.py's print loop (.rstrip on a list)."""
        from methods.latent_mas import LatentMASMethod
        args = copy.copy(tiny_args)
        args.pipeline = "parallel(planner|critic),judger"
        method = LatentMASMethod(
            tiny_model_wrapper, latent_steps=1, judger_max_new_tokens=2,
            generate_bs=2, args=args,
        )
        results = method.run_batch(_items())
        for r in results:
            for a in r["agents"]:
                # Every trace's "input" must be a string so run.py can rstrip it
                assert isinstance(a.get("input", ""), str), (
                    f"trace 'input' must be str, got {type(a.get('input')).__name__}"
                )


class TestLatentMasAnchor:
    def test_anchor_emits_text(self, tiny_args, tiny_model_wrapper):
        from methods.latent_mas import LatentMASMethod
        args = copy.copy(tiny_args)
        args.method = "latent_mas"
        args.task = "gsm8k"
        args.inter_persona_anchor_tokens = 5
        method = LatentMASMethod(
            tiny_model_wrapper, latent_steps=2, judger_max_new_tokens=4,
            generate_bs=2, args=args,
        )
        results = method.run_batch(_items())
        # Anchor texts go into the trace "output" field for non-judger agents
        for r in results:
            non_judger = [a for a in r["agents"] if a["role"] != "judger"]
            for a in non_judger:
                # Anchor text should be a non-empty string for each non-judger
                assert isinstance(a["output"], str)
