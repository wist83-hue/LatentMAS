#!/bin/bash
# MATH-500 anchors (n=128): the FLOOR and CEILING for the latent-reasoning experiment.
#
# Goal: test whether adding LatentMAS latent reasoning to Qwen2.5-Math-7B-Instruct
# (the base lineage of R1-Distill) can climb from the Instruct floor toward the
# R1-Distill ceiling. These two baselines define the range a later K-sweep aims to close.
#
#   FLOOR   = Qwen2.5-Math-7B-Instruct, baseline (no latent, no agents)
#   CEILING = DeepSeek-R1-Distill-Qwen-7B, baseline (reasoning-distilled from the same base)
#
# Budget note (deliberately per-model, documented):
#   - Instruct has a 4,096-token context window, so max_new=2048 (room for problem+answer).
#   - R1-Distill has a 131K window and produces long CoT, so max_new=8096 (the paper's
#     reasoning-intensive budget). Its ceiling may be slightly underestimated if some
#     solutions truncate at 8096 — bump R1_MAX_NEW to raise it.
#
# Scored with math-verify (utils.score_math): LaTeX equivalence, not string match.
#
# Set WAIT_PID to defer start until that process exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] anchors waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

N="${N:-128}"
# SUBSET picks the seeded MATH-500 split: tune on 'train', report final on 'test'.
SUBSET="${SUBSET:-train}"
INSTRUCT_MAX_NEW="${INSTRUCT_MAX_NEW:-2048}"
R1_MAX_NEW="${R1_MAX_NEW:-8096}"
OUT=/home/bill/src/LatentMAS/results/math500_anchors
mkdir -p "$OUT"
CSV="$OUT/math500_anchors_${SUBSET}_results.csv"
echo "anchor,subset,model,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

run_anchor() {
    local anchor="$1" model="$2" max_new="$3" bs="$4"
    LOG="$OUT/math500_${anchor}_${SUBSET}.log"
    echo "[$(date +%T)] $anchor ($model, subset=$SUBSET, max_new=$max_new, bs=$bs) START"
    python run.py --task math500 --data_subset "$SUBSET" --max_samples "$N" --max_new_tokens "$max_new" --generate_bs "$bs" \
        --model_name "$model" --method baseline --seed 42 > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] $anchor DONE: acc=$ACC sec=$SEC"
    echo "$anchor,$SUBSET,$model,$max_new,$bs,$SEC,$ACC,$COR" >> "$CSV"
}

run_anchor floor_instruct   "Qwen/Qwen2.5-Math-7B-Instruct"          "$INSTRUCT_MAX_NEW" 4
run_anchor ceiling_r1distill "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "$R1_MAX_NEW"       2

echo ""
echo "=== MATH-500 ANCHORS (subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "=== MATH-500 ANCHORS COMPLETE $(date +%T) ==="
