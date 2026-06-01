#!/bin/bash
# MATH-500 latent K-sweep on Qwen2.5-Math-7B-Instruct with a single
# strategize -> compute -> verify persona set.
#
# Question: can adding LatentMAS latent reasoning to the Instruct model climb
# from the floor (Instruct baseline) toward the ceiling (R1-Distill, same base
# lineage)? See results/math500_anchors for the floor/ceiling.
#
# Pipeline: strategize (latent) -> compute (latent) -> verify (emits \boxed answer).
#   - strategize/compute contribute K latent steps each into the KV cache.
#   - verify is the text-producer (TEXT_PRODUCER_ROLES); it reads that latent
#     context and generates the final verified answer.
# Feedback mode = auto -> w_a (Qwen2.5-Math-7B-Instruct is untied).
# Defaults: --prompt sequential, --latent_space_realign, --latent_norm_mode scalar_mean.
#
# Budget: max_new=2048 (== floor anchor, for comparability). Pipeline prompts
# (~450 tok) + 2*K latent + 2048 gen fits the model's 4,096 context for K<=~100.
# bs=2 (the baseline already used ~21.5GB at bs=4; latent adds prefill, so stay safe).
# Scored with math-verify (utils.score_math).
#
# Set WAIT_PID to defer start until that process (the anchors) exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] K-sweep waiting for PID $WAIT_PID (anchors) to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting MATH-500 K-sweep"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
N="${N:-128}"
# SUBSET picks the seeded MATH-500 split: tune K on 'train', report final on 'test'.
SUBSET="${SUBSET:-train}"
MAX_NEW="${MAX_NEW:-2048}"
BS="${BS:-2}"
K_VALUES="${K_VALUES:-0 5 10 20 40}"
PIPELINE="strategize,compute,verify"
OUT=/home/bill/src/LatentMAS/results/math500_K
mkdir -p "$OUT"
CSV="$OUT/math500_K_${SUBSET}_results.csv"
echo "K,subset,pipeline,elapsed_sec,accuracy,correct" > "$CSV"

for k in $K_VALUES; do
    LOG="$OUT/math500_K${k}_${SUBSET}.log"
    echo "[$(date +%T)] K=$k (subset=$SUBSET) START"
    python run.py --task math500 --data_subset "$SUBSET" --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
        --model_name "$MODEL" --method latent_mas --seed 42 \
        --prompt sequential --pipeline "$PIPELINE" --latent_steps "$k" \
        --latent_feedback_mode auto --latent_space_realign --latent_norm_mode scalar_mean \
        > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] K=$k DONE: acc=$ACC sec=$SEC"
    echo "$k,$SUBSET,\"$PIPELINE\",$SEC,$ACC,$COR" >> "$CSV"
done

echo ""
echo "=== MATH-500 K-SWEEP (Qwen2.5-Math-7B-Instruct, strategize->compute->verify, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo ""
echo "Anchors for reference: results/math500_anchors/math500_anchors_results.csv"
echo "=== MATH-500 K-SWEEP COMPLETE $(date +%T) ==="
