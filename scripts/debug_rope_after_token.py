"""Verify the user's RoPE claim around the latent-token insertion:

  The 'after token' (first token of the producer/solve prompt) is the SAME token in
  K=0 and K=1, so its pre-RoPE key is identical. In K=0 it prefills onto an EMPTY cache
  (position 0 -> rotation R(0)=identity); in K=1 it prefills onto a length-1 latent cache
  (position 1 -> R(1)). Therefore:

      R(+1) . (K=0 after-key)  ==  (K=1 after-key)        [token after the latent: +1 step]

  and the cache slot BEFORE the after-token differs (latent present in K=1), while
  everything from the after-token onward is the same token content shifted by one RoPE step.

We use a length-1 cache for K=1 (positionally identical to the real --latent_only latent
cache, which is exactly 1 token at position 0); the re-base unit tests separately confirm
the latent key itself sits at position 0.
"""
import argparse, torch
from models import ModelWrapper
from methods.latent_mas import _rerope_keys_shift

MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
args = argparse.Namespace(model_name=MODEL, device="cuda", method="latent_mas",
                          use_second_HF_model=False, latent_space_realign=False,
                          gpu_memory_utilization=0.9, tensor_parallel_size=1)
dev = torch.device("cuda")
mw = ModelWrapper(MODEL, dev, use_vllm=False, args=args)
model, tok = mw.model, mw.tokenizer
hd = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)
theta = float(getattr(model.config, "rope_theta", 10000.0))

# the producer (solve) prompt = the floor single-agent prompt, chat-templated
user = ("\nTarget Question: What is the integer value of $x$ in the arithmetic sequence "
        "$3^2, x, 3^4$?\n\nYou are a helpful assistant.\n\nYou must reason step-by-step to "
        "solve the **provided Target Question** without outputting other irrelevant "
        "information.\n\nNow, reason step by step and output the final answer inside "
        "\\boxed{YOUR_FINAL_ANSWER}.\n")
msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
print(f"producer prompt: {ids.shape[1]} tokens; first ('after') token id={ids[0,0].item()} = {tok.decode([ids[0,0].item()])!r}")

with torch.no_grad():
    # K=0: prefill onto EMPTY cache -> after-token at position 0
    c0 = model(ids, use_cache=True).past_key_values
    # for each K: a length-K cache (K latent slots, positions 0..K-1), then prefill the
    # prompt -> after-token at position K. Verify R(+K) . (K=0 after-key) == (K-th after-key).
    caches = {}
    for K in (1, 2, 5, 10):
        cK = model(ids[:, :K], use_cache=True).past_key_values   # length-K filler cache
        caches[K] = model(ids, past_key_values=cK, use_cache=True).past_key_values

print(f"\ncache lengths: K=0 -> {c0.get_seq_length()} (=prompt len); " +
      ", ".join(f"K={K} -> {caches[K].get_seq_length()}" for K in (1,2,5,10)) + " (=K latent + prompt)")
def stats(L, K):
    k0 = c0.layers[L].keys[..., 0:1, :].float()
    kK = caches[K].layers[L].keys[..., K:K+1, :].float()
    v0 = c0.layers[L].values[..., 0:1, :].float()
    vK = caches[K].layers[L].values[..., K:K+1, :].float()
    krot = _rerope_keys_shift(k0, -K, hd, theta).float()   # R(+K) . k0
    return (krot - kK).abs().max().item(), (v0 - vK).abs().max().item()

print("\n*** LAYER 0 (input = raw token embedding; the clean RoPE-position check) ***")
print(f"  {'K':>3}  {'max|R(+K)*k0 - kK|':>20}  {'max|v0 - vK|':>14}   verdict")
for K in (1, 2, 5, 10):
    km, vm = stats(0, K)
    ok = "PASS — rotate K=0 after-token +%d steps == K=%d after-token" % (K, K) if (km < 0.05 and vm < 0.05) else "FAIL"
    print(f"  {K:>3}  {km:>20.5f}  {vm:>14.5f}   {ok}")

print("\n*** higher layers — after-token has ALREADY attended to the K latent tokens, so")
print("    its key AND value legitimately diverge with K (this is the latent influence, NOT a RoPE bug) ***")
for L in (1, 14, 27):
    row = "  layer %2d: " % L
    for K in (1, 10):
        km, vm = stats(L, K)
        row += f"K={K}: |Rk-k|={km:.2f} |v-v|={vm:.2f}   "
    print(row)

# DECISIVE: is +K the BEST rotation? (rules out an off-by-one position bug — the residual
# is bf16, not a misplacement). Scan neighbouring shifts at layer 0; the min must be at +K.
print("\n*** best-shift scan at layer 0 (min should sit exactly at shift=+K => position is exactly K) ***")
for K in (2, 5, 10):
    k0 = c0.layers[0].keys[..., 0:1, :].float()
    kK = caches[K].layers[0].keys[..., K:K+1, :].float()
    diffs = {s: (_rerope_keys_shift(k0, -s, hd, theta).float() - kK).abs().max().item()
             for s in range(K-2, K+3)}
    best = min(diffs, key=diffs.get)
    print(f"  K={K}: " + "  ".join(f"+{s}:{d:.2f}" for s, d in diffs.items()) + f"   -> best shift = +{best} ({'== K, correct' if best==K else 'MISPLACED!'})")
# also confirm rotation preserves norm (pure rotation, no magnitude corruption)
import torch as _t
n0 = c0.layers[0].keys[...,0,:].float().norm().item()
print("\nkey-norm preserved (rotation can't change magnitude):")
for K in (1,10):
    nK = caches[K].layers[0].keys[...,K,:].float().norm().item()
    print(f"  |k0|={n0:.4f}  |k{K}|={nK:.4f}  (equal => pure rotation)")
