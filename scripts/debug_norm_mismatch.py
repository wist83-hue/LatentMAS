"""#B: Is the latent token's key/value norm anomalous vs real prompt tokens?
A high-norm key would win attention scores (q.k) -> explains the elevated attention in #1.
Compare the latent token (position 0, post-rebase) against the producer prompt tokens.
"""
import argparse, torch
from models import ModelWrapper
from methods.latent_mas import LatentMASMethod
from prompts import build_agent_message_sequential_latent_mas as LAT

MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
args = argparse.Namespace(
    model_name=MODEL, device="cuda", method="latent_mas",
    latent_space_realign=True, latent_feedback_mode="w_a", latent_norm_mode="scalar_mean",
    latent_soft_embed_temperature=2.0, latent_ablation="none",
    latent_decode_debug=False, latent_ood_debug=False,
    latent_halt_threshold=0.0, latent_halt_entropy_nats=0.0, latent_halt_argmax_steps=0,
    latent_halt_kl_nats=0.0, latent_halt_min_steps=3, latent_halt_on_eos=False,
    use_second_HF_model=False, gpu_memory_utilization=0.9, tensor_parallel_size=1,
)
dev = torch.device("cuda")
mw = ModelWrapper(MODEL, dev, use_vllm=False, args=args)
model, tok = mw.model, mw.tokenizer
trunc = object.__new__(LatentMASMethod); trunc.model = mw

q = "What is the integer value of $x$ in the arithmetic sequence $3^2, x, 3^4$?"
pa = argparse.Namespace(model_name=MODEL, task="math500", method="latent_mas", think=False,
                        text_mas_context_length=100000, concise_pipeline_prompt=False,
                        concise_nonjudger_prompt=False, minimal_persona_prompts=False)
strat_ids = tok(tok.apply_chat_template(LAT("strategize", q, "", "latent_mas", pa, False, True),
                tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
solve_ids = tok(tok.apply_chat_template(LAT("solve", q, "", "latent_mas", pa, True, False),
                tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

cache, lv = mw.generate_latent_batch(strat_ids, latent_steps=1, return_latent_vecs=True)
latent_vec = lv[0]
rebased = trunc._truncate_past(cache, 1)
with torch.no_grad():
    prod = model(solve_ids, past_key_values=rebased, use_cache=True).past_key_values

print("\nKey/Value NORMS per position (norm over heads x head_dim), latent(pos0) vs prompt(pos1..):")
print(f"  {'layer':>5}  {'latent_K':>9}  {'prompt_K mean/med/max':>24}  {'latent_V':>9}  {'prompt_V mean/med/max':>24}  pctile(latent_K)")
for L in (0, 1, 14, 27):
    kk = prod.layers[L].keys[0].float()     # [n_kv, seq, hd]
    vv = prod.layers[L].values[0].float()
    kn = kk.norm(dim=(0, 2))                 # [seq]
    vn = vv.norm(dim=(0, 2))
    lk, lv_ = kn[0].item(), vn[0].item()
    pk, pv = kn[1:], vn[1:]
    pct = (pk < lk).float().mean().item() * 100   # what % of prompt keys are smaller than latent
    print(f"  {L:>5}  {lk:>9.1f}  {pk.mean():>7.1f}/{pk.median():.1f}/{pk.max():.1f}{'':>6}  {lv_:>9.1f}  {pv.mean():>7.1f}/{pv.median():.1f}/{pv.max():.1f}{'':>6}  {pct:.0f}%")

emb = model.get_input_embeddings().weight.float()
print(f"\nW_a output (latent_vec) norm = {latent_vec.float().norm().item():.3f}")
print(f"input-embedding norms: mean={emb.norm(dim=1).mean().item():.3f} median={emb.norm(dim=1).median().item():.3f} "
      f"(scalar_mean should match latent_vec to mean)")
# per-head max key norm at the generation point's layers (a single hot head can dominate attention)
print("\nper-HEAD key norm (layer 0): latent vs prompt-token max")
kk0 = prod.layers[0].keys[0].float()  # [n_kv, seq, hd]
lat_ph = kk0[:, 0, :].norm(dim=-1)               # [n_kv]
prm_ph = kk0[:, 1:, :].norm(dim=-1)              # [n_kv, seq-1]
for h in range(kk0.shape[0]):
    print(f"  head {h}: latent={lat_ph[h].item():.1f}  prompt mean={prm_ph[h].mean().item():.1f} max={prm_ph[h].max().item():.1f}")
