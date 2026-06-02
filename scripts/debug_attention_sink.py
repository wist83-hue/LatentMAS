"""#1: Is the latent token at position 0 an over-attended 'sink' that injects garbage?

Position 0 in decoder LMs is the attention sink (content-independent attention magnet).
With --latent_only the latent garbage vector sits at position 0. So we measure: how much
of the producer's attention lands on position 0 when it holds (a) the normal first token
(<|im_start|>, K=0) vs (b) the real W_a latent garbage (K=1). If the latent token gets
sink-level attention, its OOD value is being force-blended into the producer, not ignored.
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
# need attention weights -> eager
try:
    model.set_attn_implementation("eager")
except Exception:
    model.config._attn_implementation = "eager"
trunc = object.__new__(LatentMASMethod); trunc.model = mw   # for _truncate_past

q = "What is the integer value of $x$ in the arithmetic sequence $3^2, x, 3^4$?"
pa = argparse.Namespace(model_name=MODEL, task="math500", method="latent_mas", think=False,
                        text_mas_context_length=100000, concise_pipeline_prompt=False,
                        concise_nonjudger_prompt=False, minimal_persona_prompts=False)
strat = tok.apply_chat_template(LAT("strategize", q, "", "latent_mas", pa, is_producer=False, is_first=True),
                                tokenize=False, add_generation_prompt=True)
solve = tok.apply_chat_template(LAT("solve", q, "", "latent_mas", pa, is_producer=True, is_first=False),
                                tokenize=False, add_generation_prompt=True)
strat_ids = tok(strat, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
solve_ids = tok(solve, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

def attn_to_pos0(past, label):
    with torch.no_grad():
        out = model(solve_ids, past_key_values=past, use_cache=True, output_attentions=True, return_dict=True)
    atts = out.attentions  # tuple[L] of [B, H, q_len, kv_len]
    if atts is None or atts[0] is None:
        print(f"  [{label}] output_attentions unavailable (attn not eager)"); return
    L = len(atts); P = solve_ids.shape[1]
    # mean over heads+layers of attention weight on kv-position 0, per query position
    per_layer_last = []   # weight on pos0 from the LAST query token (the generation point)
    per_layer_mean = []   # weight on pos0 averaged over all query tokens
    for a in atts:
        w0 = a[0, :, :, 0]            # [H, q_len] = attention to kv-pos 0
        per_layer_last.append(w0[:, -1].mean().item())
        per_layer_mean.append(w0.mean().item())
    uniform = 1.0 / (P + (past.get_seq_length() if past is not None else 0))
    import statistics as st
    print(f"  [{label}] kv_len={atts[0].shape[-1]}, uniform share={uniform:.4f}")
    print(f"     attn->pos0 (last query tok): mean over layers={st.mean(per_layer_last):.4f}, max layer={max(per_layer_last):.4f}")
    print(f"     attn->pos0 (all query toks): mean over layers={st.mean(per_layer_mean):.4f}, max layer={max(per_layer_mean):.4f}")
    return per_layer_mean

print("\n=== K=0: position 0 = normal first token (<|im_start|>), no latent ===")
attn_to_pos0(None, "K=0")

print("\n=== K=1: position 0 = REAL W_a latent garbage ===")
cache, _ = mw.generate_latent_batch(strat_ids, latent_steps=1, return_latent_vecs=True)
rebased = trunc._truncate_past(cache, 1)   # latent_only: keep+rebase to position 0
print(f"  (latent cache len={rebased.get_seq_length()} at position 0)")
attn_to_pos0(rebased, "K=1")
