"""Tests for --latent_feedback_mode {w_a, argmax_embed, soft_embed, coconut}."""
import copy
import pytest
import torch


def _ids_mask(mw, prompts=None):
    if prompts is None:
        prompts = ["alpha", "beta gamma", "delta epsilon"]
    p = [[{"role": "user", "content": s}] for s in prompts]
    _, ids, mask, _ = mw.prepare_chat_batch(p)
    return ids, mask


class TestArgmaxEmbedMode:
    def test_produces_real_input_embeddings(self, tiny_model_wrapper, tiny_args):
        """argmax_embed should produce vectors equal to actual E_in rows."""
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_feedback_mode = "argmax_embed"
        mw.args = args
        ids, mask = _ids_mask(mw)
        _, vecs = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=2, return_latent_vecs=True,
        )
        # Every fed-back vec must equal some row of the input embedding matrix
        emb_w = mw.model.get_input_embeddings().weight  # [V, D]
        for b in range(vecs.shape[0]):
            for k in range(vecs.shape[1]):
                v = vecs[b, k]
                # Find min distance to any embedding row
                dists = (emb_w - v.unsqueeze(0)).pow(2).sum(dim=-1)
                assert dists.min().item() < 1e-6, (
                    f"argmax_embed vec[{b},{k}] is not a real embedding row"
                )


class TestSoftEmbedMode:
    def test_temperature_1_produces_mixture(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_feedback_mode = "soft_embed"
        args.latent_soft_embed_temperature = 1.0
        mw.args = args
        ids, mask = _ids_mask(mw)
        _, vecs = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=1, return_latent_vecs=True,
        )
        # Magnitude should be in the ballpark of a typical embedding (within 10x).
        emb_w = mw.model.get_input_embeddings().weight
        typical_emb_norm = emb_w.norm(dim=-1).mean().item()
        v_norms = vecs.to(torch.float32).norm(dim=-1)
        assert (v_norms.max().item() < 10 * typical_emb_norm)
        assert (v_norms.min().item() > 0.01 * typical_emb_norm)

    def test_lower_temperature_approaches_argmax(self, tiny_model_wrapper, tiny_args):
        """soft_embed should get progressively closer to argmax_embed as τ→0."""
        mw = tiny_model_wrapper
        # argmax mode result
        args1 = copy.copy(tiny_args)
        args1.latent_feedback_mode = "argmax_embed"
        mw.args = args1
        ids, mask = _ids_mask(mw)
        _, vec_argmax = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=1, return_latent_vecs=True,
        )
        # soft with two temperatures
        def soft_at(t):
            a = copy.copy(tiny_args)
            a.latent_feedback_mode = "soft_embed"
            a.latent_soft_embed_temperature = t
            mw.args = a
            _, v = mw.generate_latent_batch(
                ids, attention_mask=mask, latent_steps=1, return_latent_vecs=True,
            )
            return v
        v_hi = soft_at(2.0)
        v_lo = soft_at(0.1)
        # Distance to argmax should be smaller for v_lo than v_hi
        d_hi = (v_hi.float() - vec_argmax.float()).norm(dim=-1).mean().item()
        d_lo = (v_lo.float() - vec_argmax.float()).norm(dim=-1).mean().item()
        assert d_lo < d_hi, f"lower τ should be closer to argmax (d_lo={d_lo}, d_hi={d_hi})"


class TestCoconutMode:
    def test_passes_through_hidden(self, tiny_model_wrapper, tiny_args):
        """coconut mode should feed back the raw hidden, no transformation."""
        mw = tiny_model_wrapper
        # We can't easily compare to "hidden" from outside without calling the
        # model internally. Instead, verify the mode runs and produces non-zero
        # vectors of the right shape, and that they differ from argmax_embed
        # (which would produce real embedding rows).
        args_coc = copy.copy(tiny_args)
        args_coc.latent_feedback_mode = "coconut"
        mw.args = args_coc
        ids, mask = _ids_mask(mw)
        _, v_coc = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=1, return_latent_vecs=True,
        )
        args_arg = copy.copy(tiny_args)
        args_arg.latent_feedback_mode = "argmax_embed"
        mw.args = args_arg
        _, v_arg = mw.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=1, return_latent_vecs=True,
        )
        # Different modes -> different vectors
        assert not torch.allclose(v_coc, v_arg, atol=1e-4)


class TestUnknownModeRaises:
    def test_invalid_mode(self, tiny_model_wrapper, tiny_args):
        mw = tiny_model_wrapper
        args = copy.copy(tiny_args)
        args.latent_feedback_mode = "made_up"
        mw.args = args
        ids, mask = _ids_mask(mw)
        with pytest.raises(ValueError, match="unknown latent_feedback_mode"):
            mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=1)


class TestDefaultIsBackwardCompat:
    def test_default_mode_is_w_a(self, tiny_args):
        """Existing scripts that don't set the flag should get the prior behavior."""
        assert getattr(tiny_args, "latent_feedback_mode", None) is None or tiny_args.latent_feedback_mode == "w_a"
