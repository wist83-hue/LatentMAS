# Cache-to-Cache (C2C): replication and dissection

A controlled study of **Cache-to-Cache (C2C)** — "Direct Semantic Communication Between Large
Language Models" (Fu, Min et al., arXiv 2510.03215, ICLR'26; code `thu-nics/C2C`). C2C lets a
**receiver** LLM borrow a **sharer** LLM's prefill understanding by **projecting and gating the
sharer's KV-cache into the receiver's**, via a small **trained** per-layer fuser (projector + gate +
weight). Only the fuser trains; both LLMs are frozen.

We undertook this after concluding our LatentMAS critique. C2C is the honest, *trained* cousin of
LatentMAS's training-free latent KV passing, and — unlike LatentMAS — its make-or-break artifact (the
trained fuser) is publicly released, so a faithful replication path exists.

**Headline:** C2C replicates faithfully. But two controlled experiments show that **~73% of its gain
on our benchmark is a trained latent *self-refinement adapter* (reproducible with the model fusing
into a copy of itself, zero cross-model information), and only ~27% (≈+3pp) is genuine cross-model
transfer — and only from a sharer that is both larger *and* instruction-aligned.** The paper's most
eye-catching cells ("C2C beats the standalone big model") are an artifact of using **base models** as
the standalone baselines, which fail the answer format and score sub-random.

---

## Setup

- **Hardware:** single RTX 4090 (24 GB). Paper used 8× A100-80G.
- **Env:** fresh conda env `rosetta`, **torch 2.6.0 / transformers 4.52.4** (the repo's pins; the
  authors confirmed transformers 4.57.x in their issues — 4.52.4 is in-range). HF backend only.
- **Models:** receiver **Qwen3-0.6B** (instruct) fixed throughout; sharers Qwen2.5-0.5B-Instruct,
  Qwen3-4B (instruct), Qwen3-4B-Base. All released fusers from `nics-efc/C2C_Fuser`.
- **Benchmark:** MMLU-redux (`edinburgh-dawg/mmlu-redux-2.0`), greedy, `max_new_tokens=64`,
  `use_template`, `enable_thinking=False` (the evaluator's default for the prompt formatting).
- **Eval gotchas (important for reproduction):**
  1. The evaluator's final `Overall accuracy` line and the summary JSON `overall_accuracy` are
     **bugged (always 0.00)**. Compute the real micro-accuracy from the per-subject lines:
     `<subject> accuracy: P% (evaluated on N samples, skipped M)` → `sum(round(P/100*N))/sum(N)`.
  2. The evaluator's **subject discovery is flaky** — it intermittently truncates to a handful of
     subjects and declares "complete" (we saw 2, 8, 52, 57 across runs, even solo). **Fix: pin the
     subject list** via `eval.subjects: [...]` (we pin the same 57 MMLU subjects everywhere; see
     `recipes/eval/repl_identical_pinned.yaml`).
  3. Do not run two evaluators concurrently for the same dataset (GPU contention re-triggers (2)).
- **All numbers below are micro-accuracy over the pinned 57-subject MMLU set**, computed from
  per-subject results.

---

## Result 1 — C2C replicates faithfully

Qwen3-0.6B receiver + Qwen2.5-0.5B sharer (the released `qwen3_0.6b+qwen2.5_0.5b_Fuser`):

| | ours | paper (Table 3) |
|---|---|---|
| Single (receiver-only) | **35.07** | 35.53 |
| C2C | **42.70** | 42.92 |
| **gain** | **+7.63** | +7.39 |

The baseline reproduces (no inflation), the method number reproduces, **and the claimed gain
reproduces** — in sharp contrast to LatentMAS, where the baseline inflated and the gain evaporated.

---

## Result 2 — the "Identical" (self-fusion) experiment: the gain is mostly a trained adapter

**Design.** Make the sharer a *copy of the receiver* (Qwen3-0.6B → Qwen3-0.6B) and train a fuser for
that self-pair. Because **prefill is deterministic** (no sampling/temperature; Qwen3 has zero
dropout), the receiver's clean cache and the sharer's cache are **bit-for-bit identical**. So the
fusion reduces to `fused = (1−w)·R + gate·w·projector(R)` — the only non-`R` content is a learned
transform of `R`'s *own* cache. **Any** gain therefore isolates the projector acting as a trained
**self-refinement of the prefill**, with the cross-model channel set to *exactly* zero by
construction.

**We trained our own** 0.6B+0.6B fuser (no released identical fuser exists). To make it tractable on
one GPU we trained on a reduced sample budget (full epoch ≈ 2 days on a 4090) and continued via a
warm-start (`SFT_train.warmstart.patch`). The eval **trajectory vs training budget**:

| effective samples | self-fusion accuracy |
|---|---|
| Single (no fusion) | 35.07 |
| 50k | 37.25 |
| 88k | 38.17 |
| 127k | **43.25** |
| 165k | 42.84 |

**Self-fusion climbs and plateaus at ~43 — matching C2C-with-a-real-0.5B-sharer (42.70).** A model
fusing with a deterministically identical copy of itself recovers ~all of that C2C cell's gain.

- The earlier modest +3pp (at 50k) was **undertraining**, not "identity is near-optimal."
- The plateau (127k→165k flat) means the asymptote is ~43, not still climbing.

**Mechanism probe (the gates).** Gates are a *static* learned scalar per layer (hard at inference:
`gate = gate_logit > 0`), separate for keys and values; the *weight* is the dynamic, per-head,
input-dependent part. Our trained self-fuser **opens value-gates 100%, key-gates ~57%** — so it is
*not* the identity; it learns a real per-layer injection pattern of its own cache.

**Conclusion:** on this task/pair, the gain attributed to "communication between *different* LLMs" is
reproduced by a model communicating with *itself*. **The value is a trained latent adapter
(self-refinement), not cross-model complementarity.**

---

## Result 3 — big→little ladder: sharer *size* barely matters; *alignment* does, modestly

Holding the receiver fixed at Qwen3-0.6B and varying only the sharer:

| sharer | accuracy | over Single | over self-fusion (~43) |
|---|---|---|---|
| — (Single) | 35.07 | — | — |
| Qwen2.5-**0.5B** | 42.70 | +7.6 | −0.5 |
| **self (Identical, plateau)** | **~43** | +8 | *(reference)* |
| Qwen3-**4B-Base** | 43.66 | +8.6 | +0.4 |
| Qwen3-**4B (instruct)** | **46.08** | +11.0 | **+2.8** |

- **0.5B, 4B-Base, and self-fusion are all clustered within ~1pp.** For these, *which* model you bolt
  on barely matters — you're mostly measuring the fuser.
- **Only the 4B-*instruct* sharer (46.08) clearly beats self-fusion** — a real, persistent **+~3pp**
  (the self-fusion plateaued, so this is not a training-budget artifact).
- **Alignment > size:** a *bigger* but *unaligned* sharer (4B-Base) adds ~nothing (≈ self); the gain
  needs a sharer that is both bigger *and* instruction-aligned.

Decomposition of the 4B-instruct → 0.6B headline (+11.0 over the bare 0.6B): **~+8 is the adapter**
(reproducible by self-fusion, no big model needed), **~+3 is genuine 4B-instruct transfer.**

---

## Result 4 — standalone references: what we *lost*, and why the paper's base-model cells flatter

Standalones on the same protocol (greedy, max_new 64, pinned 57 subjects):

| model, run directly | accuracy |
|---|---|
| Qwen3-0.6B (receiver) | 35.07 |
| **Qwen3-4B (instruct)** | **71.45** |
| **Qwen3-4B-Base** | **1.27**  *(sub-random = answer-format failure)* |

Two capstone findings:

1. **What we lost by demoting the 4B to a sharer: ~25 points.** Run the 4B directly → **71.45**.
   Demote it to a sharer for a 0.6B receiver → **46.08**. The 0.6B's generation is the bottleneck; it
   surfaces only ~46 of the 4B's 71. So **big→little recovers only ~30% of the little→big gap**
   (35→46 of a possible 35→71), **~73% of which is the adapter.** big→little is a **quality–latency
   tradeoff** (give up 25pp for 0.6B-generation speed), *not* near-big-dog quality.

2. **The paper's "C2C beats the big model" cells are a base-model artifact.** Qwen3-4B-Base standalone
   scores **1.27** (it has knowledge but cannot follow the MC format), yet C2C 4B-Base → 0.6B scores
   **43.66 ≈ self-fusion (43)**. So the base model's *knowledge contributes ~nothing*; the lift is the
   format-capable receiver + the trained adapter. When the standalone baseline is a base model
   crippled by format failure, *any* pipeline with a format-capable generator looks like it "beats the
   big model," while real cross-model transfer is ~nil. This is exactly the mechanism behind the
   paper's striking base-receiver cell (their 2.2 → 53.2 on OpenBook).

---

## Synthesis

C2C is a **real and honest method** — it reproduces, the baseline doesn't cheat, and the trained fuser
genuinely lifts a small model. But the controlled decomposition shows the celebrated framing —
*direct semantic communication between different LLMs* — overstates what's happening on our benchmark:

- **~73% of the gain is a trained latent self-refinement adapter** (a model fusing into a copy of
  itself), with **zero** cross-model information.
- **~27% (≈+3pp) is genuine cross-model transfer**, and only from a sharer that is **bigger *and*
  instruction-aligned**. Size alone (4B-Base) and small/weak sharers (0.5B) add nothing over
  self-fusion.
- **big→little as an efficiency play is real but modest:** you lift 35→46 at 0.6B-generation speed,
  but most of that needs no big model, and you remain 25pp below simply running the 4B.
- **The paper's most impressive ("beats the big model") results are produced by base-model standalones
  that fail the answer format** — a flattering baseline, not extraordinary transfer.

## Caveats / scope

- Single benchmark (MMLU-redux) and single receiver (Qwen3-0.6B). The self-fusion ≈ C2C result should
  be confirmed on a second benchmark (ARC-C / OpenBook) to rule out MMLU-specificity.
- Our self-fusion fuser is trained on a reduced budget (≤165k samples) vs the released fusers
  (~500k); it **plateaued at ~43**, so the +3pp 4B-instruct gap is not a training-budget artifact, but
  a fully-trained self-fuser could close a touch more.
- Our receiver is the **instruct** 0.6B (standalone 35), so we deliberately do **not** reproduce the
  paper's base-receiver cells (which is the point of Result 4).
- Warm-start continuation re-anneals the gate temperature (a schedule discontinuity); a clean
  trajectory would need true optimizer/scheduler resume.

## Reproduce

Clone `thu-nics/C2C`, build the `rosetta` env, then:

- **Eval recipes** (`recipes/eval/`): pinned-57-subject MMLU configs for Single, C2C (0.5B/4B/4B-Base),
  Identical (final + trajectory ckpts), and standalones. Run:
  `python script/evaluation/unified_evaluator.py --config <recipe>.yaml`, then compute micro-accuracy
  from per-subject lines (ignore the bugged `overall_accuracy`).
- **Train recipes** (`recipes/train/`): the Identical 0.6+0.6 fuser (`identical_0.6+0.6.json`) and the
  warm-start continuation (`identical_cont250k.json`). Single-GPU-faithful: bs2/seq1024,
  grad_accum 128 to hold the paper's global batch 256; ~530M trainable projector params (Adam states
  dominate memory — bs4/seq2048 OOMs 24 GB).
- **`SFT_train.warmstart.patch`**: minimal addition to `thu-nics/C2C`'s `script/train/SFT_train.py` —
  a `model.init_projectors_from` config field that loads prior projector weights before training
  (the repo has no native resume). Default behavior unchanged.
- **`scripts/`**: the eval/queue runners used for the chains above.
