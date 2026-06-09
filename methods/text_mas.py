import gc
from typing import Dict, List

import torch

from . import accumulate_call_metrics, default_agents, parse_pipeline, TEXT_PRODUCER_ROLES
from models import ModelWrapper
# from prompts import build_agent_messages, build_agent_messages_v6, build_agent_messages_v6_text_mas
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from utils import score_gsm8k, score_aime, score_math, extract_markdown_python_block, run_with_timeout
import argparse

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        pipeline_spec = getattr(args, "pipeline", None) if args else None
        self.agents = parse_pipeline(pipeline_spec) if pipeline_spec else default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task
        
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        # Per-example list of per-agent-call metric dicts, accumulated below.
        per_example_calls: List[List[Dict]] = [[] for _ in range(batch_size)]

        for agent_idx, agent in enumerate(self.agents):

            # The LAST agent in the pipeline is the text-producer: it emits the final
            # answer instead of feeding context (so a pipeline can end in any role,
            # e.g. compute in a 2-persona strategize->compute DAG, not just judger/verify).
            is_producer = (agent_idx == len(self.agents) - 1)

            if self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                        is_producer=is_producer,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                        is_producer=is_producer,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            # Optional per-agent token cap for iso-total budget comparisons.
            # When text_mas_nonjudger_max_tokens > 0 ALL agents (including the
            # judger/producer) are capped to that value so the four-agent total
            # equals 4 × cap — a strict ISO-total vs the single-agent baseline.
            # Previously only non-judger agents were capped (judger kept full
            # max_new_tokens_each = 4096), giving a 7168-token total instead of
            # the intended 4096.  Greedy to match argmax_embed's argmax behavior.
            short_cap = int(getattr(self.args, "text_mas_nonjudger_max_tokens", 0) or 0)
            if short_cap > 0:
                cap = short_cap
                use_greedy = True
            else:
                cap = self.max_new_tokens_each
                use_greedy = False

            if self.model.use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=cap,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                agent_metrics: List[Dict] = []
                # Discard past_key_values explicitly (not just `_`) so the
                # KV-cache tensor tuple is released before the next agent's
                # prepare_chat_batch allocates new GPU buffers.  This is one
                # of the two root causes of the per-example VRAM leak: without
                # the explicit del the Python refcount stays at 1 until the
                # name is rebound at the top of the next agent iteration,
                # keeping up to (n_agents - 1) full KV caches live at once.
                generated_texts, _past_kv = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=cap,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=not use_greedy,
                    metrics_out=agent_metrics,
                )
                del _past_kv  # release KV-cache GPU tensors immediately
                # Accumulate this agent's per-example metrics (cache-honest).
                for idx in range(batch_size):
                    if idx < len(agent_metrics):
                        per_example_calls[idx].append(agent_metrics[idx])

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if not is_producer:

                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                # NOTE: trimmed_ids is already a CPU Python list; mask is a
                # temporary bool tensor that goes out of scope here — fine.
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )

            # ----------------------------------------------------------------
            # Per-agent GPU memory cleanup.
            #
            # Root cause of the n=1565 OOM at example 32: PyTorch's CUDA
            # caching allocator retains freed GPU blocks in its pool rather
            # than returning them to CUDA immediately.  With 4 agents × N
            # examples the pool's high-water mark grows monotonically because:
            #   (a) input_ids / attention_mask from prepare_chat_batch are GPU
            #       tensors that stay live until the name is rebound at the top
            #       of the next agent iteration — the last agent's tensors
            #       persist until run_batch() returns;
            #   (b) past_key_values from model.generate() (discarded above as
            #       _past_kv) is a large KV-cache tuple; even after del, the
            #       allocator holds the backing CUDA memory in its free-pool;
            #   (c) no call to torch.cuda.empty_cache() between agents/examples
            #       to flush the pool back to the CUDA memory manager.
            #
            # Fix: delete the GPU input tensors immediately after all idx
            # post-processing is done, then flush the allocator's free-pool.
            # This keeps peak VRAM flat (one example's working set at a time).
            # ----------------------------------------------------------------
            del input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        per_example_metrics = accumulate_call_metrics(per_example_calls)

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
    
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')

            elif self.task in ["aime2024", "aime2025"]:
                gold = str(item.get("gold", "")).strip()
                ok, pred, error_msg = score_aime(final_text, gold)

            elif self.task == "math500":
                gold = str(item.get("gold", "")).strip()
                ok, pred, error_msg = score_math(final_text, gold)

            else:
                gold = item.get("gold", "")
                ok, pred, error_msg = score_gsm8k(final_text, gold)

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "metrics": per_example_metrics[idx],
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
