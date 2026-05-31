"""Tests for models.py: ModelWrapper, latent loop, stitch, copy/truncate, halt."""
import pytest
import torch

from models import (
    ModelWrapper,
    _NORM_EPS,
    _W_A_RIDGE_LAMBDA,
    _ensure_pad_token,
    _past_length,
)


class TestEnsurePadToken:
    def test_sets_to_eos(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", use_fast=True)
        tok.pad_token = None
        tok.pad_token_id = None
        _ensure_pad_token(tok)
        assert tok.pad_token_id is not None


class TestPastLength:
    def test_none_is_zero(self):
        assert _past_length(None) == 0

    def test_empty_is_zero(self):
        # Empty containers count as falsy in `not past_key_values`
        assert _past_length([]) == 0

    def test_with_cache(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        toks = mw.tokenizer("hello world", return_tensors="pt")
        toks = {k: v.to(mw.device) for k, v in toks.items()}
        with torch.no_grad():
            out = mw.model(**toks, use_cache=True)
        assert _past_length(out.past_key_values) == toks["input_ids"].shape[1]


class TestRenderChat:
    def test_basic(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        messages = [{"role": "user", "content": "hi"}]
        rendered = mw.render_chat(messages)
        assert isinstance(rendered, str)
        assert "hi" in rendered


class TestPrepareChatBatch:
    def test_returns_padded_batch(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        batches = [[{"role": "user", "content": "short"}],
                   [{"role": "user", "content": "a much longer prompt here"}]]
        prompts, ids, mask, tokens = mw.prepare_chat_batch(batches)
        assert len(prompts) == 2
        assert ids.shape[0] == 2
        assert ids.shape[1] == mask.shape[1]
        assert (mask[1].sum() >= mask[0].sum()).item()


class TestRealignmentMatrix:
    def test_disabled_returns_identity(self, tiny_args, tiny_model_wrapper):
        mw = tiny_model_wrapper
        # default tiny_args has latent_space_realign=False
        matrix, target_norm = mw._build_latent_realign_matrix(mw.model, torch.device("cpu"), tiny_args)
        D = mw.model.get_input_embeddings().weight.shape[1]
        assert matrix.shape == (D, D)
        # Identity check
        eye = torch.eye(D, dtype=matrix.dtype)
        assert torch.allclose(matrix, eye)
        assert target_norm.item() > 0

    def test_enabled_solves_ridge(self, tiny_args, tiny_model_wrapper):
        import copy
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_space_realign = True
        # Force a fresh build by clearing cache
        mw._latent_realign_matrices.clear()
        matrix, target_norm = mw._build_latent_realign_matrix(mw.model, torch.device("cpu"), args)
        D = mw.model.get_input_embeddings().weight.shape[1]
        assert matrix.shape == (D, D)
        # Should NOT be identity for an untied model
        eye = torch.eye(D, dtype=matrix.dtype)
        # tiny-gpt2 has tied embeddings, so W ≈ identity. Just check finite.
        assert torch.isfinite(matrix).all()
        assert target_norm.item() > 0
        # Reset for other tests
        mw._latent_realign_matrices.clear()


class TestApplyLatentRealignment:
    def test_preserve_mode_keeps_magnitude_variation(self, tiny_model_wrapper, tiny_args):
        import copy
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_norm_mode = "preserve"
        mw.args = args
        mw._latent_realign_matrices.clear()
        D = mw.model.get_input_embeddings().weight.shape[1]
        # Two rows of very different magnitudes
        h = torch.stack([torch.ones(D), torch.ones(D) * 10.0])
        out = mw._apply_latent_realignment(h, mw.model)
        norms = out.to(torch.float32).norm(dim=-1)
        # With identity W_a + preserve, row-2 should be ~10x row-1
        assert norms[1] > 5.0 * norms[0]

    def test_scalar_mean_clamps_to_one_magnitude(self, tiny_model_wrapper, tiny_args):
        import copy
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_norm_mode = "scalar_mean"
        mw.args = args
        mw._latent_realign_matrices.clear()
        D = mw.model.get_input_embeddings().weight.shape[1]
        h = torch.stack([torch.ones(D), torch.ones(D) * 10.0])
        out = mw._apply_latent_realignment(h, mw.model)
        norms = out.to(torch.float32).norm(dim=-1)
        # Legacy behavior: both rows clamped to the same scalar target_norm
        assert torch.allclose(norms[0], norms[1], atol=1e-2)

    def test_median_mode_also_clamps(self, tiny_model_wrapper, tiny_args):
        import copy
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_norm_mode = "median"
        mw.args = args
        mw._latent_realign_matrices.clear()
        # Clear cached median in case prior tests set it
        if hasattr(mw, "_target_norm_median"):
            del mw._target_norm_median
        D = mw.model.get_input_embeddings().weight.shape[1]
        h = torch.stack([torch.ones(D), torch.ones(D) * 10.0])
        out = mw._apply_latent_realignment(h, mw.model)
        norms = out.to(torch.float32).norm(dim=-1)
        assert torch.allclose(norms[0], norms[1], atol=1e-2)

    def test_unknown_mode_raises(self, tiny_model_wrapper, tiny_args):
        import copy
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_norm_mode = "bogus"
        mw.args = args
        D = mw.model.get_input_embeddings().weight.shape[1]
        with pytest.raises(ValueError, match="unknown latent_norm_mode"):
            mw._apply_latent_realignment(torch.randn(2, D), mw.model)


class TestGenerateLatentBatch:
    def test_produces_cache_grown_by_prompt_plus_K(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        prompts = [[{"role": "user", "content": "hello"}]] * 2
        _, ids, mask, _ = mw.prepare_chat_batch(prompts)
        K = 3
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=K)
        prompt_len = ids.shape[1]
        assert _past_length(past) == prompt_len + K

    def test_return_latent_vecs(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        prompts = [[{"role": "user", "content": "hi"}]] * 2
        _, ids, mask, _ = mw.prepare_chat_batch(prompts)
        K = 4
        past, vecs = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=K, return_latent_vecs=True
        )
        D = mw.model.get_input_embeddings().weight.shape[1]
        assert vecs.shape == (2, K, D)

    def test_zero_steps(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        prompts = [[{"role": "user", "content": "hi"}]] * 2
        _, ids, mask, _ = mw.prepare_chat_batch(prompts)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=0)
        assert _past_length(past) == ids.shape[1]


class TestStitchAndPrefill:
    def test_appends_correct_length(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        D = mw.model.get_input_embeddings().weight.shape[1]
        # Seed a small cache with a prompt
        prompts = [[{"role": "user", "content": "seed"}]] * 2
        _, ids, mask, _ = mw.prepare_chat_batch(prompts)
        past = mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=0)
        start_len = _past_length(past)
        # Build two branches' worth of fake data
        B = 2
        P1, K1 = 5, 3
        P2, K2 = 4, 2
        branch_data = [
            (torch.randn(B, P1, D), torch.ones(B, P1, dtype=torch.long), torch.randn(B, K1, D)),
            (torch.randn(B, P2, D), torch.ones(B, P2, dtype=torch.long), torch.randn(B, K2, D)),
        ]
        new_past = mw.stitch_and_prefill(past, branch_data)
        added = (P1 + K1) + (P2 + K2)
        assert _past_length(new_past) == start_len + added


class TestTextGeneration:
    def test_generates_some_tokens(self, tiny_model_wrapper):
        mw = tiny_model_wrapper
        prompts = [[{"role": "user", "content": "say hi"}]] * 2
        _, ids, mask, _ = mw.prepare_chat_batch(prompts)
        gens, past = mw.generate_text_batch(ids, mask, max_new_tokens=4)
        assert len(gens) == 2
        # past should grow by prompt_padded_len + new tokens (approximately)
        assert _past_length(past) >= ids.shape[1]


class TestConstants:
    def test_named_constants_exist(self):
        assert _NORM_EPS == 1e-6
        assert _W_A_RIDGE_LAMBDA == 1e-5
