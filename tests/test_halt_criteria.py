"""Tests for the four latent-loop halt criteria: velocity, entropy, argmax-stable, KL.

Uses tiny-gpt2 + small K to keep runtime reasonable.
"""
import copy
import pytest
import torch

from models import _past_length


def _ids_mask(mw, prompt="hello"):
    p = [[{"role": "user", "content": prompt}]] * 2
    _, ids, mask, _ = mw.prepare_chat_batch(p)
    return ids, mask


class TestVelocityHalt:
    def test_loose_threshold_halts_early(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        # A very loose threshold should halt at min_steps (3).
        args.latent_halt_threshold = 100.0  # larger than any plausible relative-squared delta
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=20)
        # Should run exactly halt_min_steps then halt
        assert _past_length(past) == ids.shape[1] + 3

    def test_zero_threshold_disables(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_halt_threshold = 0.0
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=5)
        # Runs all 5 steps
        assert _past_length(past) == ids.shape[1] + 5


class TestEntropyHalt:
    def test_loose_threshold_halts_early(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        # Max entropy for tiny-gpt2 vocab is log(50257) ≈ 10.8 nats. A
        # threshold above that always fires.
        args.latent_halt_entropy_nats = 100.0
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=20)
        assert _past_length(past) == ids.shape[1] + 3

    def test_zero_disables(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_halt_entropy_nats = 0.0
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=4)
        assert _past_length(past) == ids.shape[1] + 4


class TestArgmaxStableHalt:
    def test_requires_consecutive_run(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        # Demand N=10 stable steps in a 5-step loop -> never fires
        args.latent_halt_argmax_steps = 10
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=5)
        # Runs to cap
        assert _past_length(past) == ids.shape[1] + 5


class TestKLHalt:
    def test_loose_threshold_halts_early(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        # Very large KL threshold should fire as soon as prev_log_probs exists
        # (step 4+, since prev only set after step >= halt_min_steps)
        args.latent_halt_kl_nats = 100.0
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=20)
        # KL needs prev_log_probs from the previous step's logits compute. The
        # very first time we hit the halt block we don't have it yet; the
        # second time we do. So we expect 4 steps total at min_steps=3:
        # step 3: prev_log_probs set, no halt; step 4: KL fires.
        assert _past_length(past) == ids.shape[1] + 4


class TestEosHalt:
    def test_eos_halt_when_argmax_is_eos(self, tiny_model_wrapper, tiny_args):
        """If we force lm_head to always argmax to EOS, the loop should halt at min_steps.

        tiny-gpt2 has tied embeddings so we can't monkeypatch the weight
        without corrupting the input embedding. We patch lm_head.forward
        instead — that only affects the output projection.
        """
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_halt_on_eos = True
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        eos = mw.tokenizer.eos_token_id
        lm_head = mw.model.get_output_embeddings()
        vocab = lm_head.out_features
        orig_forward = lm_head.forward

        def fake_forward(x):
            # Logits all -inf-ish except EOS row which is large -> argmax = EOS
            B = x.shape[0]
            out = torch.full((B, vocab), -1e4, dtype=x.dtype, device=x.device)
            out[:, eos] = 1.0
            return out

        lm_head.forward = fake_forward
        try:
            past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=20)
            assert _past_length(past) == ids.shape[1] + 3
        finally:
            lm_head.forward = orig_forward

    def test_eos_halt_off_runs_to_cap(self, tiny_model_wrapper, tiny_args):
        """Without --latent_halt_on_eos, EOS being argmax does not halt."""
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_halt_on_eos = False
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=4)
        assert _past_length(past) == ids.shape[1] + 4


class TestOrCombining:
    def test_velocity_and_entropy_both_loose_fires_at_min(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_halt_threshold = 100.0
        args.latent_halt_entropy_nats = 100.0
        args.latent_halt_min_steps = 3
        mw.args = args
        ids, mask = _ids_mask(mw)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=20)
        assert _past_length(past) == ids.shape[1] + 3
