"""What RoPE rotation does the latent vector (W_a output) receive?

The latent step feeds W_a's output as an input embedding at the position right AFTER the
strategize prompt (length S). So the latent token's key in the cache is
    R(S) . W_k . RMSNorm(latent_vec)        [latent_vec = W_a's realigned output]
i.e. S rotation steps applied to "what comes out of W_a". (--latent_only then re-bases
it back to position 0.)

NON-circular check: take W_a's output, compute its key at position 0 independently
(forward it alone, no past), rotate by +S, and confirm it equals the latent key the real
latent step left in the cache at position S. Best-shift scan localizes the rotation to
exactly S.  For train_index=0, S (strategize prompt) = 110  (NOT 105 = the solve prompt).
"""
import argparse, torch
from models import ModelWrapper
from methods.latent_mas import _rerope_keys_shift
from prompts import build_agent_message_sequential_latent_mas as LAT

MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
args = argparse.Namespace(
    model_name=MODEL, device="cuda", method="latent_mas",
    latent_space_realign=True, latent_feedback_mode="w_a", latent_norm_mode="scalar_mean",
    latent_soft_embed_temperature=2.0, latent_ablation="none",
    latent_decode_debug=True, latent_ood_debug=False,
    latent_halt_threshold=0.0, latent_halt_entropy_nats=0.0, latent_halt_argmax_steps=0,
    latent_halt_kl_nats=0.0, latent_halt_min_steps=3, latent_halt_on_eos=False,
    use_second_HF_model=False, gpu_memory_utilization=0.9, tensor_parallel_size=1,
)
dev = torch.device("cuda")
mw = ModelWrapper(MODEL, dev, use_vllm=False, args=args)
model, tok = mw.model, mw.tokenizer
hd = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)
theta = float(getattr(model.config, "rope_theta", 10000.0))

q = "What is the integer value of $x$ in the arithmetic sequence $3^2, x, 3^4$?"
pa = argparse.Namespace(model_name=MODEL, task="math500", method="latent_mas", think=False,
                        text_mas_context_length=100000, concise_pipeline_prompt=False,
                        concise_nonjudger_prompt=False, minimal_persona_prompts=False)
msgs = LAT("strategize", q, "", "latent_mas", pa, is_producer=False, is_first=True)
txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
S = ids.shape[1]
print(f"strategize prompt length S = {S}  (latent step lands at position {S})")

with torch.no_grad():
    # real latent step (K=1): prefill strategize prompt + 1 latent -> cache len S+1
    cache, latent_vecs = mw.generate_latent_batch(ids, latent_steps=1, return_latent_vecs=True)
    latent_vec = latent_vecs[0]                       # [B, D] = W_a's realigned output
    cached_latent_key = cache.layers[0].keys[..., S:S+1, :].float()   # post-RoPE @ position S

    # independently: forward W_a's output alone -> its key at position 0 (un-rotated)
    key0 = model(inputs_embeds=latent_vec.unsqueeze(1).to(model.dtype), use_cache=True
                 ).past_key_values.layers[0].keys[..., 0:1, :].float()

print(f"\nW_a output (latent_vec): norm={latent_vec.float().norm().item():.3f}, dim={latent_vec.shape}")
print(f"cache length after 1 latent step = {cache.get_seq_length()} (= S+1 = {S+1})")

print(f"\nDoes rotating W_a's key (pos 0) by +S reproduce the cached latent token (pos {S})?")
match = (_rerope_keys_shift(key0, -S, hd, theta).float() - cached_latent_key).abs().max().item()
raw   = (key0 - cached_latent_key).abs().max().item()
print(f"  max | R(+{S})*key0 - cached_latent_key | = {match:.5f}   (≈0 => exactly {S} rotation steps)")
print(f"  max | key0 - cached_latent_key | (unrot)  = {raw:.5f}   (large => genuinely rotated)")

print(f"\nbest-shift scan (min should sit exactly at +S={S}):")
diffs = {s: (_rerope_keys_shift(key0, -s, hd, theta).float() - cached_latent_key).abs().max().item()
         for s in range(S-2, S+3)}
best = min(diffs, key=diffs.get)
print("  " + "  ".join(f"+{s}:{d:.2f}" for s, d in diffs.items()) + f"   -> best = +{best} ({'== S, correct' if best==S else 'MISMATCH'})")
# norm preserved?
print(f"\nkey-norm: |key0(pos0)|={key0.norm().item():.3f}  |cached(pos{S})|={cached_latent_key.norm().item():.3f}  (equal => pure rotation)")
