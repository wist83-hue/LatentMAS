#!/bin/bash
# MATH-500 anchors (levels 4+5, n=131): the FLOOR and CEILING for the latent experiment.
#
# Goal: test whether adding LatentMAS latent reasoning to Qwen2.5-Math-7B-Instruct
# (the base lineage of R1-Distill) can climb from the Instruct floor toward the
# R1-Distill ceiling. These two baselines define the range a later K-sweep aims to close.
# We use the level-4+5 hard tail (262 problems -> 131/131 train/test) because the
# math-specialized Instruct is near-ceiling on the easy levels (tiny gap on full MATH-500).
#
#   FLOOR   = Qwen2.5-Math-7B-Instruct, baseline (no latent, no agents)
#   CEILING = DeepSeek-R1-Distill-Qwen-7B, baseline (reasoning-distilled from the same base)
#
# Budget note (deliberately ASYMMETRIC and documented — the asymmetry is intrinsic,
# not an unfair knob: the ceiling model's whole advantage is reasoning at length):
#   - Instruct has a HARD 4,096-token context window (max_position_embeddings=4096), so
#     it CANNOT match R1's budget. max_new=3072 is its fair max (train prompts <=779 tok).
#   - R1-Distill has a 131K window and long CoT, so max_new=16384. An earlier 8096 run
#     TRUNCATED 8 of its hardest problems (still mid-<think>, no \boxed) -> underestimated
#     the ceiling. 16384 fixes that. Re-check truncation after the run regardless.
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

N="${N:-131}"
# SUBSET picks the seeded MATH-500 split: tune on 'train', report final on 'test'.
SUBSET="${SUBSET:-train}"
LEVELS="${LEVELS:-4,5}"          # difficulty levels to include (262 problems at 4+5)
TRAIN_SIZE="${TRAIN_SIZE:-131}"  # 131/131 split of the 262 level-4+5 problems
TEST_SIZE="${TEST_SIZE:-131}"
# Per-model budgets (deliberately asymmetric — see header):
#   Instruct has a HARD 4096 context window, so 3072 is its fair max (prompts <=779 tok).
#   R1-Distill (131K window) gets 16384 so its long CoT doesn't truncate (the 8096 run
#   truncated 8 of its hardest problems — see results/math500_anchors history).
INSTRUCT_MAX_NEW="${INSTRUCT_MAX_NEW:-3072}"
R1_MAX_NEW="${R1_MAX_NEW:-16384}"
OUT=/home/bill/src/LatentMAS/results/math500_anchors
mkdir -p "$OUT"
CSV="$OUT/math500_anchors_${SUBSET}_results.csv"
echo "anchor,subset,levels,model,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

run_anchor() {
    local anchor="$1" model="$2" max_new="$3" bs="$4"
    LOG="$OUT/math500_${anchor}_${SUBSET}.log"
    echo "[$(date +%T)] $anchor ($model, subset=$SUBSET, levels=$LEVELS, max_new=$max_new, bs=$bs) START"
    python run.py --task math500 --data_subset "$SUBSET" --data_levels "$LEVELS" \
        --train_size "$TRAIN_SIZE" --test_size "$TEST_SIZE" \
        --max_samples "$N" --max_new_tokens "$max_new" --generate_bs "$bs" \
        --model_name "$model" --method baseline --seed 42 > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] $anchor DONE: acc=$ACC sec=$SEC"
    echo "$anchor,$SUBSET,\"$LEVELS\",$model,$max_new,$bs,$SEC,$ACC,$COR" >> "$CSV"
}

run_anchor floor_instruct   "Qwen/Qwen2.5-Math-7B-Instruct"          "$INSTRUCT_MAX_NEW" 4
run_anchor ceiling_r1distill "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "$R1_MAX_NEW"       2

echo ""
echo "=== MATH-500 ANCHORS (levels=$LEVELS, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "=== MATH-500 ANCHORS COMPLETE $(date +%T) ==="
