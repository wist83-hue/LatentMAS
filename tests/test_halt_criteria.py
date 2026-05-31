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
