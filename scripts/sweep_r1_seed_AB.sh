#!/bin/bash
# Seed sweep of R1 intervention cells A and B, to test whether the A-vs-B gap
# (single-sweep: A=0.844, B=0.891) is a real effect or run-to-run noise.
#
# Motivation: with latent_steps=0, latent_mas discards the upstream personas'
# KV cache (latent_mas.py: `past_for_decoding = past_kv if any_latent else None`,
# and any_latent is False at K=0). So cell B is NOT multi-agent collaboration --
# it's the judger PROMPT run as a standalone single agent. A vs B is therefore
# only a prompt-template difference, and +4.7pp = 3/64 questions under stochastic
# decoding (temp 0.6) on one seed -- plausibly noise. This sweep checks.
#
#   A  --method baseline                         (single-agent baseline prompt)
#   B  --method latent_mas --latent_steps 0 ...  (judger prompt, no collaboration)
#
# Seeds include 42 (reproduces the main-sweep A=0.844 / B=0.891 data point).
#
# Set WAIT_PID to defer the start until that process (the AIME tail-chain) exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] seed sweep waiting for PID $WAIT_PID (AIME chain) to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting A/B seed sweep"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SEEDS="${SEEDS:-42 1 2 3 4}"
OUT=/home/bill/src/LatentMAS/results/r1_seed_sweep
mkdir -p "$OUT"
CSV="$OUT/r1_seed_AB_results.csv"
echo "cell,seed,elapsed_sec,accuracy" > "$CSV"

run_cell() {
    local cell="$1" seed="$2"; shift 2
    LOG="$OUT/r1_seed_${cell}_s${seed}.log"
    echo "[$(date +%T)] cell $cell seed $seed START"
    python run.py --task gsm8k --max_samples 64 --max_new_tokens 2048 --generate_bs 4 \
        --model_name "$MODEL" --seed "$seed" "$@" > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] cell $cell seed $seed DONE: acc=$ACC sec=$SEC"
    echo "$cell,$seed,$SEC,$ACC" >> "$CSV"
}

for seed in $SEEDS; do
    run_cell A "$seed" --method baseline
    run_cell B "$seed" --method latent_mas --latent_steps 0 --latent_space_realign --latent_norm_mode scalar_mean
done

echo ""
echo "=== R1 A/B SEED SWEEP (n=64, seeds: $SEEDS) ==="
column -t -s, "$CSV"
echo ""
python - "$CSV" <<'PY'
import sys, csv, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
by={}
for r in rows:
    try: by.setdefault(r["cell"],[]).append(float(r["accuracy"]))
    except: pass
for c in sorted(by):
    v=by[c]
    m=st.mean(v); sd=st.pstdev(v) if len(v)>1 else 0.0
    print(f"cell {c}: mean={m:.4f}  sd={sd:.4f}  n={len(v)}  vals={['%.3f'%x for x in v]}")
if "A" in by and "B" in by:
    print(f"\nB - A (mean delta) = {st.mean(by['B'])-st.mean(by['A']):+.4f}")
    print("If |delta| is within ~1 sd of either cell, the single-sweep +0.047 gap was noise.")
PY
echo "=== R1 A/B SEED SWEEP COMPLETE $(date +%T) ==="
