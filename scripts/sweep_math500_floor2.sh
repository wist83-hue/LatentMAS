#!/bin/bash
# MATH-500 FLOOR 2: the strategize -> compute -> verify persona set run as TEXT MAS
# (agents pass TEXT, not latent KV) on Qwen2.5-Math-7B-Instruct.
#
# This is the second floor for the latent-reasoning experiment:
#   Floor 1 = single-agent Instruct baseline           (results/math500_anchors)
#   Floor 2 = 3 personae as text MAS  <-- THIS          isolates "text decomposition"
#   Ceiling = R1-Distill                                (results/math500_anchors)
#   Latent  = same 3 personae as latent MAS (K-sweep)   tests latent KV vs text
#
# Flow (sequential): strategize emits a strategy (text) -> compute reads it and emits
# a computation (text) -> verify reads both and emits the final \boxed answer.
# This is also the target the DSPy/GEPA prompt optimization will try to lift.
#
# Budget: the Instruct model has a 4,096-token context and text MAS accumulates each
# agent's output into the next prompt, so per-agent max_new=1024 (worst case
# ~3.5K < 4096). bs=2 (text_mas KV is heavier). Scored with math-verify.
#
# Set WAIT_PID to defer start until that process (the anchors) exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] floor2 waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting Floor 2 (text MAS)"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
N="${N:-128}"
# SUBSET picks the seeded MATH-500 split: tune on 'train', report final on 'test'.
SUBSET="${SUBSET:-train}"
MAX_NEW="${MAX_NEW:-1024}"
BS="${BS:-2}"
PIPELINE="strategize,compute,verify"
OUT=/home/bill/src/LatentMAS/results/math500_floor2
mkdir -p "$OUT"
CSV="$OUT/math500_floor2_${SUBSET}_results.csv"
echo "floor,subset,method,pipeline,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

LOG="$OUT/math500_floor2_textmas_${SUBSET}.log"
echo "[$(date +%T)] Floor 2 (text MAS, $PIPELINE, subset=$SUBSET) START"
python run.py --task math500 --data_subset "$SUBSET" --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
    --model_name "$MODEL" --method text_mas --seed 42 \
    --prompt sequential --pipeline "$PIPELINE" > "$LOG" 2>&1
JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] Floor 2 DONE: acc=$ACC sec=$SEC"
echo "floor2_textmas,$SUBSET,text_mas,\"$PIPELINE\",$MAX_NEW,$BS,$SEC,$ACC,$COR" >> "$CSV"

echo ""
echo "=== MATH-500 FLOOR 2 (text MAS, strategize->compute->verify, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "Floor 1 + ceiling: results/math500_anchors/math500_anchors_results.csv"
echo "=== MATH-500 FLOOR 2 COMPLETE $(date +%T) ==="
