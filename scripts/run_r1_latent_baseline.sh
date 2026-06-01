#!/bin/bash
# Cell K -- the latent baseline for the R1 intervention sweep.
#
# This is the matched control for cells C-J: full pipeline, full (non-minimal)
# persona prompts, argmax_embed K=20, NO <think> brackets. It's the plain
# latent_mas config that C-J each perturb one dimension of, so every
# intervention's effect is measured relative to THIS number (not relative to
# cell A, which is the no-latent single-agent baseline). Prior measurement of
# this exact config: ~0.766.
#
# Appends one row to the existing A-J results CSV. Honors WAIT_PID to defer
# start until another process (the A-J sweep) exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] cell K waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; running cell K"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUT=/home/bill/src/LatentMAS/results/r1_interventions
mkdir -p "$OUT"
CSV="$OUT/r1_interventions_results.csv"
# Header is created by sweep_r1_interventions.sh; create it if running standalone.
[ -f "$CSV" ] || echo "label,description,elapsed_sec,accuracy" > "$CSV"

label="K_latent_baseline"
LOG="$OUT/r1_intervention_${label}.log"
echo "[$(date +%T)] $label START"
python run.py --task gsm8k --max_samples 64 --max_new_tokens 2048 --generate_bs 4 \
    --model_name "$MODEL" --seed 42 \
    --method latent_mas --latent_steps 20 --latent_feedback_mode argmax_embed \
    --latent_space_realign --latent_norm_mode scalar_mean > "$LOG" 2>&1
JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] $label DONE: acc=$ACC sec=$SEC"
echo "$label,\"latent baseline: full pipeline + full prompts + argmax_embed K=20, no brackets\",$SEC,$ACC" >> "$CSV"
