import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


# Tikhonov regularizer for the W_a ridge regression (Eq. solving
# (EᵀE + λI) W = Eᵀ E_in). Small enough to barely affect the solution
# while keeping (EᵀE + λI) invertible for any vocab size.
_W_A_RIDGE_LAMBDA = 1e-5

# Floor for ||x|| denominators in renormalization to avoid div-by-zero
# when a hidden state collapses to (near-)zero magnitude.
_NORM_EPS = 1e-6


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    return past_key_values.get_seq_length()


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        if self.use_vllm:
            self._init_vllm(model_name, args)
        else:
            self._init_hf(model_name, device, args)

    def _init_vllm(self, model_name: str, args) -> None:
        tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
        gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
        print(f"[vLLM] Using vLLM backend for model {model_name}")
        if args.enable_prefix_caching and args.method == "latent_mas":
            self.vllm_engine = LLM(
                model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util,
                enable_prefix_caching=True, enable_prompt_embeds=True,
            )
        else:
            self.vllm_engine = LLM(
                model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        use_second_hf = bool(getattr(args, "use_second_HF_model", False))
        if use_second_hf:
            self.HF_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            ).to(args.device2).eval()
            self.embedding_layer = self.HF_model.get_input_embeddings()
            self.HF_device = args.device2
            self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
        elif self.latent_space_realign:
            raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
        _ensure_pad_token(self.tokenizer)

    def _init_hf(self, model_name: str, device: torch.device, args) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
        _ensure_pad_token(self.tokenizer)
        # Choose dtype based on the device we'll actually place the model on,
        # not just whether CUDA is available. bf16 on CPU breaks layer_norm.
        load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=load_dtype)
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            template_kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
            # Qwen3+ supports an enable_thinking template flag. Default ON; setting
            # the CLI flag --disable_thinking turns it off (model emits an empty
            # <think></think> block and goes straight to the final answer).
            # Models whose chat templates don't reference enable_thinking will
            # silently ignore this kwarg.
            if self.args and getattr(self.args, "disable_thinking", False):
                template_kwargs["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        # Fast path: realign disabled -> only need target_norm (a scalar from
        # input embedding magnitudes). Compute it cheaply without ever
        # materializing both full embedding matrices in fp32.
        if not self.args.latent_space_realign:
            with torch.no_grad():
                target_norm = input_embeds.weight.detach().to(dtype=torch.float32).norm(dim=1).mean().to(device)
            D = input_embeds.weight.shape[1]
            realign_matrix = torch.eye(D, device=device, dtype=torch.float32)
            return realign_matrix, target_norm

        # Paper path: solve ridge regression W_a = (EᵀE + λI)⁻¹ Eᵀ E_in.
        # This needs both full matrices in fp32 (~2 × V × D × 4 bytes).
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = _W_A_RIDGE_LAMBDA * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()
        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _maybe_compute_ood_reference(self, model) -> None:
        """Lazy-init the reference NN distance between real E_in rows.

        Used by --latent_ood_debug to give a baseline against which to compare
        the fed-back latent vector's NN distance.
        """
        if getattr(self, "_ood_ref_l2", None) is not None:
            return
        emb_raw = model.get_input_embeddings().weight.detach()
        V = emb_raw.shape[0]
        # Memory-fused per-row norm: avoids materializing [V, D] fp32 squared
        # intermediate (~2.5GB for Qwen3-8B). torch.norm is kernel-fused.
        self._ood_emb_sq_norms = emb_raw.norm(dim=-1).to(torch.float32) ** 2  # [V]
        # For the sample-vs-all matmul we need fp32 emb. Cast in-place by
        # chunks to keep peak memory manageable on the model's GPU.
        n = min(64, V)
        idx = torch.randperm(V, generator=torch.Generator().manual_seed(0))[:n].to(emb_raw.device)
        # We need emb in fp32 only for the matmul below; do it once.
        emb_fp32 = emb_raw.to(torch.float32)
        sample = emb_fp32[idx]  # [n, D]
        s_sq = (sample.norm(dim=-1) ** 2).unsqueeze(1)  # [n, 1] — fused
        inner = sample @ emb_fp32.T  # [n, V]
        d_sq = (s_sq + self._ood_emb_sq_norms.unsqueeze(0) - 2 * inner).clamp_min(0)
        for k in range(n):
            d_sq[k, idx[k]] = float("inf")
        nn_l2 = d_sq.sqrt().min(dim=-1).values
        del emb_fp32, inner, d_sq
        import torch as _torch  # avoid name shadowing in some paths
        if hasattr(_torch.cuda, "empty_cache"):
            _torch.cuda.empty_cache()
        self._ood_ref_l2 = nn_l2.mean().item()
        print(
            f"[ood] reference: mean nearest-neighbor L2 distance between real E_in rows "
            f"= {self._ood_ref_l2:.3f} (sampled n={n} of V={V})",
            flush=True,
        )

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)
        mode = getattr(self.args, "latent_norm_mode", "preserve") if self.args else "preserve"
        if mode == "preserve" or mode == "none":
            # Keep the magnitude that came out of W_a. With a well-conditioned
            # W_a this is already in the right scale; with W_a == identity (no
            # realignment), this is the model's native hidden magnitude.
            pass
        elif mode == "scalar_mean":
            # Legacy behavior: rescale every row to the vocab-mean input
            # embedding norm. Kills per-row magnitude variation; can produce
            # the random-walk dynamics we observed in the velocity-halt debug.
            aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
            aligned = aligned * (target_norm / aligned_norm)
        elif mode == "median":
            # Vocab-median input embedding norm. More robust than 'scalar_mean'
            # against outlier embeddings; still a single scalar so it still
            # destroys per-row variation.
            aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
            # `target_norm` was computed as mean by _build_latent_realign_matrix;
            # for median we compute from input embeddings on demand and cache.
            if not hasattr(self, "_target_norm_median"):
                input_embeds = model.get_input_embeddings()
                w = input_embeds.weight.detach().to(dtype=torch.float32)
                self._target_norm_median = w.norm(dim=1).median().detach()
            t = self._target_norm_median.to(device=aligned.device, dtype=aligned.dtype)
            aligned = aligned * (t / aligned_norm)
        else:
            raise ValueError(f"unknown latent_norm_mode: {mode!r}")
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        do_sample: bool = True,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
        )
        # Only pass sampling kwargs when sampling; HF warns otherwise.
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        outputs = self.model.generate(**gen_kwargs)
        sequences = outputs.sequences
        prompt_padded_len = input_ids.shape[1]
        generations: List[str] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, prompt_padded_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        return_latent_vecs: bool = False,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # output_hidden_states=True is needed only to extract the final layer's
        # hidden at the last position. For the prompt prefill (seq=prompt_len)
        # this allocates [num_layers, B, prompt_len, D] briefly; transformers
        # drops intermediate layers after this call returns. For per-step passes
        # below (seq=1), the cost is negligible.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        latent_vecs_all: List[torch.Tensor] = []

        halt_threshold = float(getattr(self.args, "latent_halt_threshold", 0.0) or 0.0) if self.args else 0.0
        halt_entropy = float(getattr(self.args, "latent_halt_entropy_nats", 0.0) or 0.0) if self.args else 0.0
        halt_argmax_steps = int(getattr(self.args, "latent_halt_argmax_steps", 0) or 0) if self.args else 0
        halt_kl = float(getattr(self.args, "latent_halt_kl_nats", 0.0) or 0.0) if self.args else 0.0
        halt_min_steps = int(getattr(self.args, "latent_halt_min_steps", 3) or 3) if self.args else 3
        prev_h1 = None  # hidden at step N-1
        prev_h2 = None  # hidden at step N-2
        prev_argmax = None
        argmax_run_len = 0
        prev_log_probs = None
        ablation = getattr(self.args, "latent_ablation", "none") if self.args else "none"
        decode_debug = bool(getattr(self.args, "latent_decode_debug", False)) if self.args else False
        ood_debug = bool(getattr(self.args, "latent_ood_debug", False)) if self.args else False
        feedback_mode = getattr(self.args, "latent_feedback_mode", "w_a") if self.args else "w_a"
        soft_temp = float(getattr(self.args, "latent_soft_embed_temperature", 1.0) or 1.0) if self.args else 1.0
        halt_on_eos = bool(getattr(self.args, "latent_halt_on_eos", False)) if self.args else False
        eos_id = getattr(self.tokenizer, "eos_token_id", None) if halt_on_eos else None

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            if feedback_mode == "w_a":
                latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            elif feedback_mode == "coconut":
                latent_vec = last_hidden
            elif feedback_mode == "argmax_embed":
                lm_head = source_model.get_output_embeddings()
                emb = source_model.get_input_embeddings()
                logits = lm_head(last_hidden)
                next_id = logits.argmax(dim=-1)  # [B]
                latent_vec = emb(next_id)        # [B, D] - real input embedding
            elif feedback_mode == "soft_embed":
                lm_head = source_model.get_output_embeddings()
                emb_w = source_model.get_input_embeddings().weight  # [V, D]
                logits = lm_head(last_hidden)
                probs = (logits / soft_temp).softmax(dim=-1)         # [B, V]
                latent_vec = probs.to(emb_w.dtype) @ emb_w           # [B, D]
            else:
                raise ValueError(f"unknown latent_feedback_mode: {feedback_mode!r}")

            if ablation == "zero":
                latent_vec = torch.zeros_like(latent_vec)
            elif ablation == "shuffle":
                # Permute across batch dim: each example gets a different
                # example's latent vector. Preserves the magnitude/direction
                # distribution but breaks the per-example signal.
                perm = torch.randperm(latent_vec.shape[0], device=latent_vec.device)
                latent_vec = latent_vec[perm]
            elif ablation == "gaussian":
                # Random direction, per-row matched magnitude.
                target = latent_vec.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
                g = torch.randn_like(latent_vec)
                latent_vec = g * (target / g.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS))

            if decode_debug:
                # Project this latent vector through lm_head and report the top-5
                # tokens. Makes drift visible: domain words early, noise later.
                lm_head = source_model.get_output_embeddings()
                with torch.no_grad():
                    logits = lm_head(latent_vec)
                    topk = logits.topk(5, dim=-1).indices  # [B, 5]
                tok = getattr(self, "tokenizer", None)
                if tok is not None:
                    rows = [tok.convert_ids_to_tokens(topk[i].tolist()) for i in range(topk.shape[0])]
                    print(f"[latent-decode] step={step+1} top5={rows}", flush=True)
                else:
                    print(f"[latent-decode] step={step+1} topk_ids={topk.tolist()}", flush=True)

            if ood_debug:
                self._maybe_compute_ood_reference(source_model)
                lm_head = source_model.get_output_embeddings()
                emb = source_model.get_input_embeddings()
                with torch.no_grad():
                    # Work in the model's native dtype to avoid materializing
                    # a [V, D] fp32 copy of the embedding matrix every step.
                    # Cast small per-batch tensors to fp32 at the end for
                    # readable distance values.
                    emb_w = emb.weight.detach()  # [V, D] in model dtype
                    v = latent_vec.detach().to(emb_w.dtype)
                    # NN L2: ||v - r||² = ||v||² + ||r||² - 2 v·r
                    v_sq = (v.norm(dim=-1) ** 2).unsqueeze(-1)  # [B, 1]
                    inner = v @ emb_w.T  # [B, V]
                    d_sq = (v_sq.float() + self._ood_emb_sq_norms.to(v.device) - 2 * inner.float()).clamp_min(0)
                    nn_l2 = d_sq.sqrt().min(dim=-1).values
                    # NN cosine
                    v_norms = v.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
                    emb_norms = emb_w.norm(dim=-1).clamp_min(_NORM_EPS)  # [V]
                    inner_norm = inner / v_norms.float()  # [B, V] sort of normalized
                    cos = (inner.float() / (v_norms.float() * emb_norms.float().unsqueeze(0)))
                    nn_cos = 1.0 - cos.max(dim=-1).values
                    # Decode-and-check
                    pre_logits = lm_head(last_hidden)
                    pred_id = pre_logits.argmax(dim=-1)
                    target = emb_w[pred_id]  # [B, D] in model dtype
                    dec_l2 = (v.float() - target.float()).norm(dim=-1)
                    target_norm = target.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
                    dec_cos = 1.0 - ((v.float() * target.float()).sum(dim=-1) / (v_norms.squeeze(-1).float() * target_norm.squeeze(-1).float()))
                tok_obj = getattr(self, "tokenizer", None)
                pred_str = tok_obj.convert_ids_to_tokens([pred_id[0].item()])[0] if tok_obj is not None else str(pred_id[0].item())
                print(
                    f"[ood] step={step+1} nn_l2={nn_l2.mean().item():.3f} "
                    f"(ref={self._ood_ref_l2:.3f}, ratio={nn_l2.mean().item() / max(self._ood_ref_l2, 1e-6):.2f}x) "
                    f"nn_cos={nn_cos.mean().item():.4f}",
                    flush=True,
                )
                print(
                    f"[ood] step={step+1} pred='{pred_str}' decode_l2={dec_l2.mean().item():.3f} "
                    f"decode_cos={dec_cos.mean().item():.4f}",
                    flush=True,
                )

            latent_vecs_all.append(latent_vec.detach().clone())
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            if (step + 1) >= halt_min_steps:
                vel_halt = False
                if halt_threshold > 0 and prev_h1 is not None and prev_h2 is not None:
                    # Squared-magnitude denominator for relative-velocity halt check;
                    # tiny floor prevents NaN on collapsed states.
                    denom = (last_hidden ** 2).sum(dim=-1).clamp_min(1e-8)
                    d1 = ((last_hidden - prev_h1) ** 2).sum(dim=-1) / denom
                    d2 = ((last_hidden - prev_h2) ** 2).sum(dim=-1) / denom
                    vel_halt = bool(torch.all((d1 < halt_threshold) & (d2 < halt_threshold)))
                ent_halt = False
                arg_halt = False
                kl_halt = False
                eos_halt = False
                need_logits = (halt_entropy > 0 or halt_argmax_steps > 0
                               or halt_kl > 0 or (halt_on_eos and eos_id is not None))
                if need_logits:
                    lm_head = source_model.get_output_embeddings()
                    logits = lm_head(last_hidden)
                    log_probs = logits.log_softmax(dim=-1)
                    cur_argmax = log_probs.argmax(dim=-1) if (halt_argmax_steps > 0 or halt_on_eos) else None
                    if halt_entropy > 0:
                        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                        ent_halt = bool(torch.all(entropy < halt_entropy))
                    if halt_argmax_steps > 0:
                        if prev_argmax is not None and bool(torch.all(cur_argmax == prev_argmax)):
                            argmax_run_len += 1
                        else:
                            argmax_run_len = 1
                        prev_argmax = cur_argmax
                        arg_halt = argmax_run_len >= halt_argmax_steps
                    if halt_kl > 0 and prev_log_probs is not None:
                        kl = (log_probs.exp() * (log_probs - prev_log_probs)).sum(dim=-1)
                        kl_halt = bool(torch.all(kl < halt_kl))
                    if halt_kl > 0:
                        prev_log_probs = log_probs
                    if halt_on_eos and eos_id is not None:
                        # Tokenizer may return a single int or a list; normalize.
                        eos_ids = eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]
                        eos_mask = torch.zeros_like(cur_argmax, dtype=torch.bool)
                        for e in eos_ids:
                            eos_mask |= (cur_argmax == e)
                        eos_halt = bool(torch.all(eos_mask))
                if vel_halt or ent_halt or arg_halt or kl_halt or eos_halt:
                    break
            prev_h2 = prev_h1
            prev_h1 = last_hidden

        if return_latent_vecs:
            if latent_vecs_all:
                latent_vecs_tensor = torch.stack(latent_vecs_all, dim=1)
            else:
                latent_vecs_tensor = torch.zeros(
                    (input_ids.shape[0], 0, last_hidden.shape[-1]),
                    dtype=last_hidden.dtype, device=last_hidden.device,
                )
            return past, latent_vecs_tensor
        return past

    @torch.no_grad()
    def stitch_and_prefill(
        self,
        past_key_values: Optional[Tuple],
        branch_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple:
        """Concatenate [prompt_embeds_i, latent_vecs_i] for each branch and prefill via one forward pass.

        branch_data: list of (prompt_embeds [B,P,D], prompt_mask [B,P], latent_vecs [B,K,D]).
        Returns extended past_key_values with stitched content at correct RoPE positions.
        """
        B = branch_data[0][0].shape[0]
        device = branch_data[0][0].device
        chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        for prompt_embeds, prompt_mask, latent_vecs in branch_data:
            chunks.append(prompt_embeds)
            chunks.append(latent_vecs)
            K = latent_vecs.shape[1]
            mask_chunks.append(prompt_mask.to(device=device, dtype=torch.long))
            mask_chunks.append(torch.ones((B, K), dtype=torch.long, device=device))
        full_embeds = torch.cat(chunks, dim=1)
        full_new_mask = torch.cat(mask_chunks, dim=1)
        past_len = _past_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones((B, past_len), dtype=torch.long, device=device)
            full_mask = torch.cat([past_mask, full_new_mask], dim=-1)
        else:
            full_mask = full_new_mask
        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.past_key_values

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

