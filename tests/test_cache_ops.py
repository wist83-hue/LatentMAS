"""Tests for _copy_past_kv and _truncate_past in LatentMASMethod."""
import argparse
import pytest
import torch

from models import _past_length


@pytest.fixture
def latent_method(tiny_args):
    from methods.latent_mas import LatentMASMethod
    # We need a method instance for _copy_past_kv / _truncate_past; don't need the model
    class _FakeModel:
        device = torch.device("cpu")
    # Constructing the real method requires args.device, args.device2, args.task
    return LatentMASMethod(_FakeModel(), args=tiny_args)


def _make_cache(mw, prompt="hello world"):
    p = [[{"role": "user", "content": prompt}]] * 2
    _, ids, mask, _ = mw.prepare_chat_batch(p)
    return mw.generate_latent_batch(ids, attention_mask=mask, latent_steps=3)


class TestCopyPastKv:
    def test_none_returns_none(self, latent_method):
        assert latent_method._copy_past_kv(None) is None

    def test_unknown_type_raises(self, latent_method):
        with pytest.raises(TypeError, match="unsupported past_key_values type"):
            latent_method._copy_past_kv("not a cache")

    def test_copies_isolate(self, latent_method, tiny_model_wrapper):
        past = _make_cache(tiny_model_wrapper)
        original_len = _past_length(past)
        copied = latent_method._copy_past_kv(past)
        assert _past_length(copied) == original_len
        # Mutate the copy and verify original is unaffected
        for layer in copied.layers:
            if layer.is_initialized:
                layer.keys.zero_()
                layer.values.zero_()
        # Original should still hold non-zero values
        assert any(
            layer.is_initialized and layer.keys.abs().sum().item() > 0
            for layer in past.layers
        )


class TestTruncatePast:
    def test_none_returns_none(self, latent_method):
        assert latent_method._truncate_past(None, 5) is None

    def test_zero_keep_returns_none(self, latent_method, tiny_model_wrapper):
        past = _make_cache(tiny_model_wrapper)
        assert latent_method._truncate_past(past, 0) is None

    def test_truncates_to_requested_length(self, latent_method, tiny_model_wrapper):
        past = _make_cache(tiny_model_wrapper)
        orig = _past_length(past)
        keep = max(1, orig - 2)
        truncated = latent_method._truncate_past(past, keep)
        assert _past_length(truncated) == keep
