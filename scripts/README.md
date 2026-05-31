# Sweep scripts

Reproducibility harnesses for the experiments we've been running. Each script
is self-contained, sources `_common.sh` for shared setup, and is parameterized
through env vars (no CLI flags — keeps invocation short).

## Setup

All scripts assume:

- A conda env (default `latentmas`) with the project's `requirements.txt`
  installed plus `pytest` (for tests) and `gptqmodel` (for AWQ models).
- The repo checked out and `cwd` doesn't matter — scripts `cd` to the repo via
  `$(dirname "$0")/..` resolution.
- An HF cache. Default `$HF_HOME=<repo>/huggingface` so models can be shared
  across runs without re-downloading.

## Common env vars

| Var | Default | Meaning |
|---|---|---|
| `CONDA_BASE` | `$HOME/anaconda3` | Where to find `conda` |
| `CONDA_ENV` | `latentmas` | Conda env to activate |
| `HF_HOME` | `<repo>/huggingface` | HuggingFace cache dir |
| `MODEL` | `Qwen/Qwen3-4B` | HF model id |
| `TASK` | `gsm8k` | Dataset (gsm8k / aime2024 / gpqa / etc.) |
| `N` | `64` | `--max_samples`; use `-1` for full dataset |
| `K` | `20` | `--latent_steps` where applicable |
| `BS` | `8` | `--generate_bs` |
| `MAX_NEW` | `2048` | `--max_new_tokens` |

Override any of these at invocation, e.g.:

```bash
MODEL=Qwen/Qwen3-8B N=128 K=40 ./scripts/run_headlines.sh
```

Results land in `/tmp/<sweep>_<MODEL>_<TASK>.csv` with one row per cell.

## Scripts

| Script | Purpose |
|---|---|
| `run_headlines.sh` | 3-way method comparison: baseline / text_mas / latent_mas |
| `sweep_latent_K.sh` | Vary `--latent_steps` over `K_VALUES` (default `5 10 20 40 80`) |
| `sweep_halt_modes.sh` | Compare the 4 halt criteria (velocity / entropy / argmax-stable / KL) |
| `sweep_feedback_modes.sh` | Compare `--latent_feedback_mode {w_a, coconut, argmax_embed, soft_embed}` |
| `sweep_ablations.sh` | Replace latent vectors with zero/shuffled/gaussian — is latent signal real? |
| `compare_short_text_vs_latent.sh` | 4-cell test of whether latent_mas argmax_embed is just hidden greedy text |

## Tips

- For text_mas, the `TEXT_MAS_BS` env var (default 4) lets you cap the text_mas
  batch size separately from `BS`. text_mas's KV cache balloons because each
  agent emits a full `MAX_NEW` budget — drop to `TEXT_MAS_BS=2` for ~14B
  models.
- The first run on a new model downloads weights from HF (5–30 GB depending on
  size). Use `HF_HOME=<persistent-dir>` to share across sweeps.
- These scripts run on the foreground GPU. They do not chain (no `wait`
  loops) — to chain multiple sweeps, just run them sequentially in one shell,
  or wrap them in a wrapper script.
