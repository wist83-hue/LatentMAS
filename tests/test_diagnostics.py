"""Tests for diagnostic flags: --latent_ablation, --latent_decode_debug."""
import copy
import io
import sys

import pytest
import torch

from models import _past_length


def _ids_mask(mw, prompts=None):
    """Build a batch with distinct prompts so per-example latent vectors differ.

    tiny-gpt2 has hidden_size=2 (tiny!), so identical-prompt batches produce
    identical latent vectors and shuffle ablation becomes a no-op.
    """
    if prompts is None:
        prompts = ["alpha", "beta gamma", "delta epsilon zeta", "eta theta iota kappa"]
    p = [[{"role": "user", "content": s}] for s in prompts]
    _, ids, mask, _ = mw.prepare_chat_batch(p)
    return ids, mask


class TestLatentAblationZero:
    def test_zero_produces_zero_vectors(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_ablation = "zero"
        mw.args = args
        ids, mask = _ids_mask(mw)
        _, vecs = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=3, return_latent_vecs=True,
        )
        # vecs shape [B, K, D]; all should be zero
        assert torch.allclose(vecs, torch.zeros_like(vecs))


class TestLatentAblationShuffle:
    def test_shuffle_changes_assignment(self, tiny_model_wrapper, tiny_args):
        # Use preserve mode so per-row magnitudes (not just directions) differ
        # between rows. With scalar_mean clamp + tiny-gpt2's D=2, the rows can
        # collapse to indistinguishable vectors and shuffle becomes a no-op.
        mw = tiny_model_wrapper
        torch.manual_seed(0)
        args = copy.copy(tiny_args)
        args.latent_ablation = "none"
        args.latent_norm_mode = "preserve"
        mw.args = args
        ids, mask = _ids_mask(mw)
        _, base = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=2, return_latent_vecs=True,
        )
        torch.manual_seed(123)
        args = copy.copy(tiny_args)
        args.latent_ablation = "shuffle"
        args.latent_norm_mode = "preserve"
        mw.args = args
        _, shuf = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=2, return_latent_vecs=True,
        )
        assert vecs_not_identical(base, shuf), (
            "shuffle ablation produced identical vectors to base run"
        )


class TestLatentAblationGaussian:
    def test_gaussian_matches_magnitude(self, tiny_model_wrapper, tiny_args):
        # tiny-gpt2 has hidden_size=2 so any randomness-based direction test
        # is too coincidence-prone. We test the contractual property:
        # per-row magnitudes should match within numerical tolerance.
        mw = tiny_model_wrapper
        torch.manual_seed(0)
        args = copy.copy(tiny_args)
        args.latent_ablation = "none"
        mw.args = args
        ids, mask = _ids_mask(mw)
        _, base = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=2, return_latent_vecs=True,
        )
        args = copy.copy(tiny_args)
        args.latent_ablation = "gaussian"
        mw.args = args
        torch.manual_seed(42)
        _, g = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=2, return_latent_vecs=True,
        )
        # Shape matches
        assert g.shape == base.shape
        # Only step 1 has the same realigned input as the base run (later
        # steps diverge because the gaussian replacement is fed back). So we
        # check the magnitude-matching contract on step 1 specifically.
        base_step1 = base[:, 0, :].to(torch.float32).norm(dim=-1)
        g_step1 = g[:, 0, :].to(torch.float32).norm(dim=-1)
        assert torch.allclose(g_step1, base_step1, rtol=1e-3, atol=1e-4)


class TestDecodeDebug:
    def test_decode_debug_prints_per_step(self, tiny_model_wrapper, tiny_args, capsys):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_decode_debug = True
        mw.args = args
        ids, mask = _ids_mask(mw)
        mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=3)
        captured = capsys.readouterr().out
        # One line per step
        assert captured.count("[latent-decode]") == 3
        assert "step=1" in captured
        assert "step=3" in captured


def vecs_not_identical(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> bool:
    """Helper: return True if a and b differ noticeably anywhere."""
    return not torch.allclose(a, b, atol=atol)
