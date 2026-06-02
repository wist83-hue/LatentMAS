#!/bin/bash
# MATH-500 PAIR (compute -> verify): a 2-PERSONA text-MAS design, on
# Qwen2.5-Math-7B-Instruct. Goal: can a multi-persona design BEAT the single-agent
# floor (floor_instruct = 0.756 on L4+5 train)? Companion to pair_sc (strategize->
# compute); this variant puts the dedicated solver FIRST and a checker last.
#
# Flow (sequential DAG): compute solves the problem standalone (no strategy precedes
# it — its prompt adapts to empty context) -> verify reads compute's solution, checks
# it, and (as the LAST/producer agent) emits the final \boxed answer.
#
# Budget (4096 squeeze — TIGHTER than pair_sc): BOTH agents do heavy work (compute
# fully solves, verify may re-derive to check), so NO concision and a uniform
# max_new=1408 to guarantee the worst case fits: q(<=791)+compute(1408)+overhead(250)
# +verify(1408) ~= 3.9K < 4096. NOTE this caps the SOLVER (compute) at 1408 — well
# under the single agent's effective need — so compute may truncate on the hardest
# problems. That squeeze is itself a finding for this design. Truncation-check after.
#
# Set WAIT_PID to defer start until that process exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] pair_cv waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting pair_cv (text MAS)"
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
MAX_NEW="${MAX_NEW:-1408}"   # uniform; guarantees 2-agent worst case < 4096 (see header)
CONCISE="${CONCISE:-}"        # none: compute is the solver, must not be concised
BS="${BS:-2}"
PIPELINE="compute,verify"
OUT=/home/bill/src/LatentMAS/results/math500_pair_cv
mkdir -p "$OUT"
CSV="$OUT/math500_pair_cv_${SUBSET}_results.csv"
echo "cell,subset,levels,method,pipeline,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

LOG="$OUT/math500_pair_cv_${SUBSET}.log"
echo "[$(date +%T)] pair_cv (text MAS, $PIPELINE, subset=$SUBSET, levels=$LEVELS, max_new=$MAX_NEW) START"
python run.py --task math500 --data_subset "$SUBSET" --data_levels "$LEVELS" \
    --train_size "$TRAIN_SIZE" --test_size "$TEST_SIZE" \
    --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
    --model_name "$MODEL" --method text_mas --seed 42 \
    --prompt sequential --pipeline "$PIPELINE" $CONCISE > "$LOG" 2>&1
JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] pair_cv DONE: acc=$ACC sec=$SEC"
echo "pair_cv,$SUBSET,\"$LEVELS\",text_mas,\"$PIPELINE\",$MAX_NEW,$BS,$SEC,$ACC,$COR" >> "$CSV"

echo ""
echo "=== MATH-500 pair_cv (text MAS, compute->verify, levels=$LEVELS, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "Compare vs floor_instruct 0.756 / floor_agents ~0.70 / pair_sc / ceiling 0.893"
echo "=== MATH-500 pair_cv COMPLETE $(date +%T) ==="
