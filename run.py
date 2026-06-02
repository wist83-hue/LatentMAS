import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_math500,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model id to load (e.g. 'Qwen/Qwen3-4B', 'casperhansen/deepseek-r1-distill-qwen-14b-awq').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "math500", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    # MATH-500 train/test split (replicable; tune on 'train', report final on held-out 'test').
    parser.add_argument("--data_subset", choices=["all", "train", "test"], default="all",
                        help="MATH-500 only: seeded disjoint split. 'all' = full set in original order.")
    parser.add_argument("--data_seed", type=int, default=42, help="Seed for the MATH-500 train/test shuffle.")
    parser.add_argument("--train_size", type=int, default=131, help="MATH-500 train subset size (131 = half of the 262 level-4+5 problems).")
    parser.add_argument("--test_size", type=int, default=131, help="MATH-500 test subset size.")
    parser.add_argument("--data_levels", type=str, default="4,5",
                        help="MATH-500 only: comma-separated difficulty levels (1-5) to include, or 'all'. Default '4,5' (the 262 hardest, for a wider floor-ceiling gap).")
    parser.add_argument("--data_index", type=int, default=-1,
                        help="MATH-500 only: if >=0, run ONLY the i-th problem of the (filtered) subset. For single-problem debugging.")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy (deterministic argmax) decoding instead of sampling — for reproducible debugging / no-op verification.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method (uniform across personas, unless --latent_steps_map is set)")
    parser.add_argument("--latent_only", action="store_true", help="Pass ONLY the K latent vectors between agents (truncate each non-producer's prompt prefill from the KV, keeping just its latent steps). Makes the inter-agent transfer rounding-error-small so the producer keeps a full generation budget within the context window — for apples-to-apples comparison against the single-agent baseline.")
    parser.add_argument("--latent_in_producer_turn", action="store_true", help="(#3 structural fix) Place the K latent vectors INSIDE the producer's assistant turn (after <|im_start|>assistant\\n), re-forwarded in producer context, instead of prepending them as KV before the opening <|im_start|>. Isolates latent PLACEMENT from prompt-retention. HF sequential path only; carries the last non-producer agent's latent vectors.")
    parser.add_argument("--latent_steps_map", type=str, default=None, help='Per-persona latent steps, e.g. "planner:40,critic:25,refiner:10". Overrides --latent_steps for listed roles; unlisted roles fall back to --latent_steps.')
    parser.add_argument("--pipeline", type=str, default=None, help='Pipeline spec, e.g. "planner,(critic+refiner)*2,judger". Default: planner,critic,refiner,judger.')
    parser.add_argument("--latent_halt_threshold", type=float, default=0.0, help="Relative-squared-velocity threshold for adaptive latent-loop halting. 0 disables (uses fixed latent_steps).")
    parser.add_argument("--latent_halt_min_steps", type=int, default=3, help="Minimum latent steps before halting may trigger.")
    parser.add_argument("--latent_halt_entropy_nats", type=float, default=0.0, help="Halt latent loop when next-token entropy (nats) of all batch elements drops below this. 0 disables. Combined with --latent_halt_threshold by OR.")
    parser.add_argument("--latent_halt_argmax_steps", type=int, default=0, help="Halt latent loop when argmax(logits) is the same token for N consecutive steps for all batch elements. 0 disables.")
    parser.add_argument("--latent_halt_kl_nats", type=float, default=0.0, help="Halt latent loop when KL(p_N || p_{N-1}) drops below this (nats) for all batch elements. 0 disables.")
    parser.add_argument(
        "--latent_halt_on_eos",
        action="store_true",
        help=(
            "Halt latent loop when argmax(lm_head(hidden)) equals the tokenizer's EOS id "
            "for all batch elements. Mirrors how text generation naturally stops; closes the "
            "argmax_embed vs text_mas_short equivalence gap. Combined with other halts by OR."
        ),
    )
    parser.add_argument("--inter_persona_anchor_tokens", type=int, default=0, help="After each non-judger persona's latent loop, emit this many text tokens (greedy) as a manifold anchor between iterations. 0 disables.")
    parser.add_argument(
        "--latent_ablation",
        choices=["none", "zero", "shuffle", "gaussian"],
        default="none",
        help=(
            "Diagnostic: replace each latent vector before injection. "
            "'none' (default): no change. "
            "'zero': inject zeros (tests 'is any signal needed?'). "
            "'shuffle': permute latent vectors across the batch (tests "
            "'is the per-example signal needed, or any latent-shaped vector?'). "
            "'gaussian': random vectors with matching per-row magnitude (tests "
            "'is the specific direction needed?'). If accuracy is unchanged "
            "under all three, the latent path is noise the judger routes around."
        ),
    )
    parser.add_argument(
        "--latent_decode_debug",
        action="store_true",
        help="Per latent step, log the top-5 argmax tokens of lm_head(latent_vec) to stdout. Makes drift visible.",
    )
    parser.add_argument(
        "--latent_ood_debug",
        action="store_true",
        help=(
            "Per latent step, log two OOD metrics for the fed-back vector: "
            "(1) nearest-neighbor L2 / cosine distance to any input embedding row, "
            "compared to a reference distance computed once at startup from real "
            "E_in rows; (2) distance to E_in[argmax(lm_head(pre-W_a hidden))] — "
            "the input embedding of the token the model would emit next."
        ),
    )
    parser.add_argument(
        "--latent_feedback_mode",
        choices=["auto", "w_a", "argmax_embed", "soft_embed", "coconut"],
        default="auto",
        help=(
            "How to convert the per-step hidden state into the next inputs_embed. "
            "'auto' (default): pick w_a for untied embeddings, soft_embed (τ=2) for "
            "tied — captures the empirical finding that w_a degenerates to identity "
            "on tied embedding models (Qwen3-4B, smaller Qwen3 variants) where "
            "soft_embed performs significantly better. "
            "'w_a' (paper-faithful): hidden @ W_a, then rescale per --latent_norm_mode. "
            "'argmax_embed': E_in[argmax(lm_head(hidden))] — hard-discretize to the "
            "model's own predicted next token and re-embed it. "
            "'soft_embed': softmax(lm_head(hidden)/tau) @ E_in — expected E_in under "
            "the predicted next-token distribution. "
            "'coconut': raw hidden, no transformation. Original Coconut formulation."
        ),
    )
    parser.add_argument(
        "--latent_soft_embed_temperature",
        type=float,
        default=2.0,
        help=(
            "Temperature for soft_embed (or 'auto' when it resolves to soft_embed). "
            "Lower = more peaky (closer to argmax_embed). Default 2.0; empirically "
            "best on tied Qwen3-4B."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument(
        "--text_mas_nonjudger_max_tokens",
        type=int,
        default=0,
        help=(
            "Cap each non-judger agent's textual output at this many tokens "
            "(judger keeps --max_new_tokens). 0 disables (default = use --max_new_tokens "
            "for all agents). Generation is greedy when this is set; EOS still stops early. "
            "Used to compare against latent_mas argmax_embed: short greedy text reasoning "
            "and argmax_embed-K=N produce identical KV cache state."
        ),
    )
    parser.add_argument(
        "--concise_nonjudger_prompt",
        action="store_true",
        help=(
            "Append a 'reply with a single short sentence' instruction to non-judger "
            "agent prompts. Helps short-budget text_mas pack content into its cap, and "
            "should be used with matching latent_mas runs for an apples-to-apples comparison."
        ),
    )
    parser.add_argument(
        "--concise_pipeline_prompt",
        action="store_true",
        help=(
            "Append a SOFT 'be concise — essential steps only' instruction to non-producer "
            "pipeline agents (e.g. strategize/compute), leaving the producer (verify/judger) "
            "untouched. Frees 4096-context budget so the producer doesn't truncate. Softer "
            "than --concise_nonjudger_prompt (which forces a single sentence)."
        ),
    )
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help=(
            "For Qwen3+ models: pass enable_thinking=False to the chat template, "
            "making the model skip its internal <think></think> reasoning phase and "
            "go straight to the final answer. Lets us produce a baseline that's "
            "comparable to papers that disabled thinking mode."
        ),
    )
    parser.add_argument(
        "--minimal_persona_prompts",
        action="store_true",
        help=(
            "Replace planner/critic/refiner persona prompts with a minimal "
            "'Solve this problem step by step' template. Designed for reasoning-"
            "distilled models (R1-Distill etc.) whose training distribution is "
            "incompatible with the verbose 'You are a Planner Agent' framing. "
            "Judger prompt is unchanged."
        ),
    )
    parser.add_argument(
        "--latent_thinking_brackets",
        action="store_true",
        help=(
            "Wrap each non-judger persona's latent loop with explicit <think> and "
            "</think> tokens. Designed for reasoning-distilled models (R1-Distill) "
            "whose training had complete <think>...</think> blocks. Opens by "
            "appending '<think>' to the prompt suffix (like --think); closes by "
            "injecting '</think>\\n\\n' tokens into the KV cache after the latent "
            "loop. Judger is unaffected."
        ),
    )
    parser.add_argument(
        "--latent_thinking_brackets_global",
        action="store_true",
        help=(
            "Wrap the ENTIRE multi-agent latent sequence in a SINGLE <think>...</think> "
            "block (rather than one per persona). Opens '<think>' once at the start "
            "of the first non-judger persona, closes '</think>\\n\\n' once before "
            "the judger's prompt. Matches R1's training distribution of one "
            "thinking block per response. Mutually exclusive with "
            "--latent_thinking_brackets (per-persona)."
        ),
    )
    # The paper's central method depends on the ridge-regressed W_a matrix
    # mapping output hidden space -> input embedding space. Default ON so
    # `--method latent_mas` is paper-faithful out of the box; opt out with
    # --no_latent_space_realign for ablation.
    parser.add_argument("--latent_space_realign", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--latent_norm_mode",
        choices=["preserve", "scalar_mean", "median", "none"],
        default="scalar_mean",
        help=(
            "How to rescale post-W_a latent vectors before feeding back. "
            "'scalar_mean' (default, paper-faithful): clamp every row to the vocab-mean "
            "input-embedding norm. Necessary because hidden states have ~100x the magnitude "
            "of input embeddings; without rescaling the fed-back vector is wildly OOD. "
            "'preserve': keep the W_a-output magnitude per row (only correct if W_a actually "
            "produces embedding-scale vectors, which holds for some untied models but breaks "
            "for tied embeddings like Qwen3-4B). "
            "'median': use vocab median norm instead of mean (outlier-robust). "
            "'none': alias for 'preserve'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()

    if args.method == "latent_mas" and args.use_vllm:
        if not getattr(args, "use_second_HF_model", False):
            print("[run.py] latent_mas + vLLM requires a second HF model for the latent pass; "
                  "setting --use_second_HF_model=True")
            args.use_second_HF_model = True
        if not getattr(args, "enable_prefix_caching", False):
            print("[run.py] latent_mas + vLLM benefits from prefix caching; "
                  "setting --enable_prefix_caching=True")
            args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "math500":
        _levels = None if args.data_levels.strip().lower() == "all" else [int(x) for x in args.data_levels.split(",") if x.strip()]
        dataset_iter = load_math500(subset=args.data_subset, seed=args.data_seed,
                                    train_n=args.train_size, test_n=args.test_size, levels=_levels,
                                    pick_index=args.data_index)
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    
    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
