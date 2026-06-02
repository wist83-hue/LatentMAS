#!/bin/bash
# MATH-500 LATENT K-sweep: a 2-persona LATENT-MAS DAG on Qwen2.5-Math-7B-Instruct.
# Tests the token-efficiency hypothesis behind the whole project: latent KV passing
# (K tokens) vs text handoffs (hundreds). The non-producer agent contributes K latent
# steps to the KV cache; the producer (LAST agent, position-based) decodes the final
# \boxed answer attending to that latent context.
#
#   PIPELINE / NAME select the DAG:
#     "strategize,compute" -> latent_sc  (strategize loops latently, compute produces)
#     "compute,verify"     -> latent_cv  (compute loops latently, verify produces)
#   K_VALUES sweeps the non-producer's latent steps.
#
# Budget: --latent_only passes ONLY the K(<=40) latent vectors between agents (the
# prior agent's prompt prefill — which RE-includes the question, ~hundreds of tokens —
# is truncated from the KV). So the producer's context = its own prompt + K(<=40) + gen,
# i.e. EXACTLY the single-agent floor (floor_instruct ran max_new=3072 and never
# truncated) plus rounding-error latent tokens. max_new=3072 to match floor exactly.
# Worst case: K(40) + producer_prompt(<=~1000) + gen(3072) ~= 4.0K < 4096.
# Feedback mode auto -> w_a (Qwen2.5-Math is UNTIED). Paper-faithful defaults:
# --latent_space_realign, --latent_norm_mode scalar_mean.
#
# Compare vs floor_instruct 0.756 / pair_sc 0.756 / pair_cv 0.740 / ceiling 0.893.
# Set WAIT_PID to defer start until that process exits.

set -uo pipefail

PIPELINE="${PIPELINE:-strategize,compute}"
NAME="${NAME:-latent_sc}"

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] $NAME waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting $NAME"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
N="${N:-131}"
SUBSET="${SUBSET:-train}"
LEVELS="${LEVELS:-4,5}"
TRAIN_SIZE="${TRAIN_SIZE:-131}"
TEST_SIZE="${TEST_SIZE:-131}"
MAX_NEW="${MAX_NEW:-3072}"      # = floor_instruct's budget; --latent_only keeps it (see header)
BS="${BS:-4}"      # match floor_instruct's generate_bs=4 (same as the baseline run)
K_VALUES="${K_VALUES:-0 5 10 20 40 80 160}"   # K=0 = no-op control (--latent_only: no past → reduces to single-agent baseline; for 'solve' == floor_instruct)
OUT=/home/bill/src/LatentMAS/results/math500_${NAME}
mkdir -p "$OUT"
CSV="$OUT/math500_${NAME}_${SUBSET}_results.csv"
echo "cell,K,subset,levels,pipeline,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

for k in $K_VALUES; do
    LOG="$OUT/math500_${NAME}_K${k}_${SUBSET}.log"
    echo "[$(date +%T)] $NAME K=$k ($PIPELINE, subset=$SUBSET, levels=$LEVELS) START"
    python run.py --task math500 --data_subset "$SUBSET" --data_levels "$LEVELS" \
        --train_size "$TRAIN_SIZE" --test_size "$TEST_SIZE" \
        --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
        --model_name "$MODEL" --method latent_mas --seed 42 \
        --prompt sequential --pipeline "$PIPELINE" --latent_steps "$k" --latent_only \
        --latent_feedback_mode auto --latent_space_realign --latent_norm_mode scalar_mean \
        > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] $NAME K=$k DONE: acc=$ACC sec=$SEC"
    echo "$NAME,$k,$SUBSET,\"$LEVELS\",\"$PIPELINE\",$MAX_NEW,$BS,$SEC,$ACC,$COR" >> "$CSV"
done

echo ""
echo "=== MATH-500 $NAME LATENT K-sweep ($PIPELINE, levels=$LEVELS, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "Compare: floor_instruct 0.756 / pair_sc 0.756 / pair_cv 0.740 / ceiling 0.893"
echo "=== MATH-500 $NAME COMPLETE $(date +%T) ==="
