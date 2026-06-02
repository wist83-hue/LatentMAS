"""Comprehensive RoPE-alignment tests for the latent_only KV re-basing.

Mismatched RoPE is a SILENT corruptor: the model still runs and emits plausible
text, but attention is garbage. So these tests pin down the re-basing from several
independent angles:

  1. FORMULA exactness (float64): re-basing R(start+i)-rotated keys by -start equals
     R(i)-rotated keys, to ~1e-12. Proves the algebra independent of fp precision.
  2. IMPLEMENTATION (_rerope_keys_shift, float32 as the model uses): same property at
     model precision.
  3. CONVENTION vs transformers: our forward-RoPE convention matches the library's
     apply_rotary_pos_emb — so re-basing inverts the SAME rotation the model applied.
  4. ATTENTION invariance: <R(p)q, rebased_key_i> == <R(p)q, R(i)k> — the dot product
     that actually drives attention is preserved.
  5. _truncate_past END-TO-END on a real DynamicCache: the last-K keys come out rotated
     for positions [0,K), values are the matching slice, and seq_length == K.

The ultimate check (the actual model recovering accuracy with --latent_only) is an
E2E run, not a unit test — but these gate the silent-corruption failure mode.
"""
import pytest
import torch

from methods.latent_mas import _rotate_half, _rope_inv_freq, _rerope_keys_shift

THETA = 10000.0


def _forward_rope(x, positions, head_dim, theta=THETA):
    """Reference forward RoPE (Llama/Qwen2 convention) at the given absolute positions.
    x: [b, h, K, head_dim]; positions: 1-D length-K tensor."""
    inv_freq = _rope_inv_freq(head_dim, theta, x.device).to(x.dtype)
    freqs = positions.to(x.dtype)[:, None] * inv_freq[None, :]   # [K, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)                      # [K, d]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


@pytest.mark.parametrize("start", [1, 5, 137, 491, 903])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_formula_exact_float64(start, head_dim):
    # Pure-algebra check in float64: rebasing high-position keys == low-position keys.
    torch.manual_seed(0)
    b, h, K = 2, 4, 6
    k = torch.randn(b, h, K, head_dim, dtype=torch.float64)
    pos_hi = torch.arange(start, start + K, dtype=torch.float64)
    pos_lo = torch.arange(0, K, dtype=torch.float64)
    k_hi = _forward_rope(k, pos_hi, head_dim)
    k_lo = _forward_rope(k, pos_lo, head_dim)
    # local float64 rebase mirroring _rerope_keys_shift's formula
    inv_freq = _rope_inv_freq(head_dim, THETA, k.device).to(torch.float64)
    ang = float(start) * inv_freq
    emb = torch.cat([ang, ang], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    k_rebased = k_hi * cos - _rotate_half(k_hi) * sin
    assert torch.allclose(k_rebased, k_lo, atol=1e-10), (k_rebased - k_lo).abs().max().item()


@pytest.mark.parametrize("start", [1, 5, 137, 491, 903])
def test_rerope_keys_shift_matches_low_positions_f32(start):
    # The ACTUAL function at the model's float32 RoPE precision.
    torch.manual_seed(1)
    b, h, K, d = 2, 4, 6, 128
    k = torch.randn(b, h, K, d, dtype=torch.float32)
    k_hi = _forward_rope(k, torch.arange(start, start + K, dtype=torch.float32), d)
    k_lo = _forward_rope(k, torch.arange(0, K, dtype=torch.float32), d)
    k_rebased = _rerope_keys_shift(k_hi, start, d, THETA)
    # float32 cos/sin of large angles (start up to ~900) caps precision near ~1e-3
    assert torch.allclose(k_rebased, k_lo, atol=2e-3), (k_rebased - k_lo).abs().max().item()


def test_rerope_shift_zero_is_noop():
    k = torch.randn(1, 2, 4, 64)
    assert torch.equal(_rerope_keys_shift(k, 0, 64, THETA), k)


def test_attention_dot_product_preserved():
    # What actually matters: <R(p)q, rebased key at i> == <R(p)q, R(i)k>.
    torch.manual_seed(2)
    b, h, K, d = 1, 2, 5, 128
    start, p = 200, 7  # producer query at position p (>=K), latent keys rebased to [0,K)
    k = torch.randn(b, h, K, d)
    q = torch.randn(b, h, 1, d)
    q_p = _forward_rope(q, torch.tensor([float(p)]), d)
    k_hi = _forward_rope(k, torch.arange(start, start + K, dtype=torch.float32), d)
    k_rebased = _rerope_keys_shift(k_hi, start, d, THETA)
    k_true_low = _forward_rope(k, torch.arange(0, K, dtype=torch.float32), d)
    score_rebased = (q_p * k_rebased).sum(-1)   # [b,h,K]
    score_true = (q_p * k_true_low).sum(-1)
    assert torch.allclose(score_rebased, score_true, atol=2e-3), (score_rebased - score_true).abs().max().item()


def test_convention_matches_transformers():
    # Our forward-RoPE convention must equal the library's, so the inverse re-base lines
    # up with the rotation the actual model applied. Skip if the API shape differs.
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception as e:  # pragma: no cover
        pytest.skip(f"qwen2 rope import unavailable: {e}")
    torch.manual_seed(3)
    b, h, K, d = 2, 4, 6, 128
    k = torch.randn(b, h, K, d)
    positions = torch.arange(0, K)
    inv_freq = _rope_inv_freq(d, THETA, k.device)
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)          # [K, d]
    cos = emb.cos()[None]                            # [1, K, d]  (batch dim)
    sin = emb.sin()[None]
    # apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1) -> heads dim inserted at 1
    _, k_tf = apply_rotary_pos_emb(k, k, cos, sin)
    k_ours = _forward_rope(k, positions, d)
    assert torch.allclose(k_tf, k_ours, atol=1e-5), (k_tf - k_ours).abs().max().item()


def test_truncate_past_rebases_cache_end_to_end():
    # Build a DynamicCache whose last K keys are rotated at [prompt_len, prompt_len+K);
    # _truncate_past should keep them re-based to [0,K), keep matching values, len==K.
    from transformers import DynamicCache
    from methods.latent_mas import LatentMASMethod

    class _Cfg:  # head_dim = 512/8 = 64
        hidden_size = 512
        num_attention_heads = 8
        rope_theta = THETA
    class _HF:  pass
    class _MW:  pass
    m = object.__new__(LatentMASMethod)
    mw = _MW(); mw.model = _HF(); mw.model.config = _Cfg()
    m.model = mw

    torch.manual_seed(4)
    b, n_kv, d = 2, 2, 64
    prompt_len, K = 130, 5
    k_prompt = torch.randn(b, n_kv, prompt_len, d)
    k_lat = torch.randn(b, n_kv, K, d)                       # the "latent step" key vectors
    v_prompt = torch.randn(b, n_kv, prompt_len, d)
    v_lat = torch.randn(b, n_kv, K, d)
    # full cache keys: prompt roped at [0,prompt_len), latent roped at [prompt_len, +K)
    full_k = torch.cat([
        _forward_rope(k_prompt, torch.arange(0, prompt_len, dtype=torch.float32), d),
        _forward_rope(k_lat, torch.arange(prompt_len, prompt_len + K, dtype=torch.float32), d),
    ], dim=2)
    full_v = torch.cat([v_prompt, v_lat], dim=2)

    cache = DynamicCache()
    n_layers = 3
    for li in range(n_layers):
        cache.update(full_k.clone(), full_v.clone(), li)
    assert cache.get_seq_length() == prompt_len + K

    out = m._truncate_past(cache, K)
    assert out.get_seq_length() == K
    expected_k = _forward_rope(k_lat, torch.arange(0, K, dtype=torch.float32), d)
    for li in range(n_layers):
        assert torch.allclose(out.layers[li].keys, expected_k, atol=2e-3), \
            (out.layers[li].keys - expected_k).abs().max().item()
        assert torch.allclose(out.layers[li].values, v_lat, atol=0), "values must be the kept slice, unrotated"
    # original cache untouched (copy semantics)
    assert cache.get_seq_length() == prompt_len + K
