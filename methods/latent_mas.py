from typing import Dict, List, Optional, Tuple

from . import default_agents, parse_pipeline, Agent, Parallel
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import score_gsm8k, score_aime, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from vllm import SamplingParams

from transformers.cache_utils import Cache

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.latent_steps_map: Dict[str, int] = {}
        raw_map = getattr(args, "latent_steps_map", None) if args else None
        if raw_map:
            for pair in raw_map.split(","):
                role, k = pair.split(":")
                self.latent_steps_map[role.strip().lower()] = int(k)
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        pipeline_spec = getattr(args, "pipeline", None) if args else None
        self.agents = parse_pipeline(pipeline_spec) if pipeline_spec else default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

    def _copy_past_kv(self, past_kv: Optional[Cache]) -> Optional[Cache]:
        """Deep-copy a DynamicCache so an isolated branch run can't mutate the parent state.

        Branch isolation depends on this. If we silently fail to copy, the
        branch's latent loop would write into the parent cache and the
        independence guarantee breaks. So fail loudly on unknown types.

        transformers 5.x removed `to_legacy_cache()`; we copy via the per-layer
        keys/values tensors instead.
        """
        if past_kv is None:
            return None
        if not isinstance(past_kv, Cache):
            raise TypeError(
                f"_copy_past_kv: unsupported past_key_values type {type(past_kv).__name__}; "
                "expected a transformers Cache subclass (DynamicCache, etc.)"
            )
        new_cache = past_kv.__class__()
        # Make sure the new cache has as many layer slots as the source
        while len(new_cache.layers) < len(past_kv.layers):
            new_cache.layers.append(type(past_kv.layers[0])())
        for i, src in enumerate(past_kv.layers):
            dst = new_cache.layers[i]
            if src.is_initialized:
                # Initialize with empty tensors of the right dtype/device, then
                # overwrite with cloned data. Direct assignment after init keeps
                # the cache's bookkeeping consistent.
                dst.lazy_initialization(src.keys, src.values)
                dst.keys = src.keys.clone()
                dst.values = src.values.clone()
        return new_cache

    def _truncate_past(self, past_kv: Optional[Cache], tokens_to_keep: int) -> Optional[Cache]:
        """Truncate `past_kv` to the last `tokens_to_keep` positions, returning a copy.

        We copy first so the original cache is preserved for the caller.
        """
        if past_kv is None or tokens_to_keep <= 0:
            return None
        copied = self._copy_past_kv(past_kv)
        cur = copied.get_seq_length()
        if cur <= tokens_to_keep:
            return copied
        # Keep the LAST `tokens_to_keep` positions
        start = cur - tokens_to_keep
        for layer in copied.layers:
            if layer.is_initialized:
                layer.keys = layer.keys[..., start:, :].contiguous()
                layer.values = layer.values[..., start:, :].contiguous()
        return copied

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        # Track non-judger persona count for --latent_thinking_brackets_global:
        # open <think> only on the first, close </think> only before the judger.
        _nonjudger_seen = 0

        for op in self.agents:

            if isinstance(op, Parallel):
                snapshot = self._copy_past_kv(past_kv)
                branch_data = []
                embed_layer = self.model.model.get_input_embeddings()
                for branch in op.branches:
                    if len(branch) != 1:
                        raise NotImplementedError("parallel branches with >1 agent not yet supported")
                    ba = branch[0]
                    if self.args.prompt == "sequential":
                        bmsgs = [build_agent_message_sequential_latent_mas(role=ba.role, question=it["question"], context="", method=self.method_name, args=self.args) for it in items]
                    else:
                        bmsgs = [build_agent_message_hierarchical_latent_mas(role=ba.role, question=it["question"], context="", method=self.method_name, args=self.args) for it in items]
                    b_prompts, b_ids, b_mask, _ = self.model.prepare_chat_batch(bmsgs, add_generation_prompt=True)
                    snap_copy = self._copy_past_kv(snapshot)
                    k_branch = self.latent_steps_map.get(ba.role, self.latent_steps)
                    _, latent_vecs = self.model.generate_latent_batch(
                        b_ids, attention_mask=b_mask,
                        latent_steps=k_branch, past_key_values=snap_copy,
                        return_latent_vecs=True,
                    )
                    prompt_embeds = embed_layer(b_ids)
                    branch_data.append((prompt_embeds, b_mask, latent_vecs))
                    for idx in range(batch_size):
                        agent_traces[idx].append({
                            "name": ba.name, "role": ba.role, "branch": True,
                            "input": b_prompts[idx], "latent_steps": k_branch, "output": "",
                        })
                past_kv = self.model.stitch_and_prefill(snapshot, branch_data)
                continue

            agent = op
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # --latent_thinking_brackets opens per-persona; --think also does;
                # --latent_thinking_brackets_global opens once on the first non-judger.
                _is_first_nonjudger = (_nonjudger_seen == 0)
                _nonjudger_seen += 1
                _open_per = self.args.think or getattr(self.args, "latent_thinking_brackets", False)
                _open_global = bool(getattr(self.args, "latent_thinking_brackets_global", False)) and _is_first_nonjudger
                _open_think = _open_per or _open_global
                if _open_think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                k_for_agent = self.latent_steps_map.get(agent.role, self.latent_steps)
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=k_for_agent,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = k_for_agent if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                # Close the explicit <think> bracket if --latent_thinking_brackets is set.
                # Injects '</think>\n\n' tokens into the KV cache so the cached
                # thinking span is properly terminated for R1-Distill-style models.
                if getattr(self.args, "latent_thinking_brackets", False) and past_kv is not None:
                    past_kv = self.model.append_tokens_to_cache(
                        "</think>\n\n", past_kv, batch_size,
                    )

                anchor_tokens = int(getattr(self.args, "inter_persona_anchor_tokens", 0) or 0)
                anchor_texts = ["" for _ in range(batch_size)]
                if anchor_tokens > 0 and past_kv is not None:
                    seed_ids = wrapped_ids[:, -1:]
                    seed_mask = torch.ones_like(seed_ids)
                    anchor_texts, past_kv = self.model.generate_text_batch(
                        seed_ids,
                        attention_mask=seed_mask,
                        max_new_tokens=anchor_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_kv,
                    )

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": k_for_agent,
                            "output": anchor_texts[idx],
                        }
                    )
            else:

                # Close the single global <think> block (if --latent_thinking_brackets_global)
                # right before the judger's prompt prefill.
                if (getattr(self.args, "latent_thinking_brackets_global", False)
                        and past_kv is not None and _nonjudger_seen > 0):
                    past_kv = self.model.append_tokens_to_cache(
                        "</think>\n\n", past_kv, batch_size,
                    )

                any_latent = self.latent_steps > 0 or any(v > 0 for v in self.latent_steps_map.values())
                past_for_decoding = past_kv if any_latent else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

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

            else:
                gold = item.get("gold", "")
                ok, pred, error_msg = score_gsm8k(final_text, gold)
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                k_for_agent = self.latent_steps_map.get(agent.role, self.latent_steps)
                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=k_for_agent,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = k_for_agent if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if k_for_agent > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -k_for_agent:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": k_for_agent,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # Find a "user turn start" marker in the prompt so we can splice
                # the latent embeddings into the user-content region. The marker
                # depends on the model's chat template; we try a handful of
                # known patterns. Add to this list when supporting new families.
                _user_turn_markers = [
                    "<|im_start|>user\n",        # Qwen2/2.5/3, R1-distill-Qwen
                    "<|start_header_id|>user<|end_header_id|>\n\n",  # Llama 3
                    "[INST]",                    # Mistral / Llama 2
                ]
                marker = None
                for m in _user_turn_markers:
                    if all(m in p for p in judger_prompts):
                        marker = m
                        break
                if marker is None:
                    raise RuntimeError(
                        "vLLM judger path: could not locate a known user-turn marker in "
                        "judger_prompts. Add the chat template's user-turn marker to "
                        "_user_turn_markers in methods/latent_mas.py."
                    )

                len_of_left = []
                for p in judger_prompts:
                    idx = p.find(marker)
                    left = p[: idx + len(marker)]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item["gold"]
            ok, pred, _ = score_gsm8k(final_text, gold)
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
