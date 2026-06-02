#!/bin/bash
# MATH-500 PAIR (strategize -> compute): a 2-PERSONA text-MAS design, on
# Qwen2.5-Math-7B-Instruct. Goal: can a multi-persona design BEAT the single-agent
# floor (floor_instruct = 0.756 on L4+5 train)? The 3-persona strategize->compute->
# verify run (floor_agents) landed ~0.70, dragged down by verify re-deriving and
# truncating. Dropping verify frees the window: only 2 agents share 4096, so the
# producer (compute) gets a real budget.
#
# Flow (sequential DAG): strategize emits a brief strategy -> compute reads problem
# + strategy, executes it, and (as the LAST/producer agent) emits the final \boxed
# answer itself. "Producer = last agent" is position-based (methods/text_mas.py), so
# compute boxes here even though it's not judger/verify.
#
# Budget (4096): --concise_pipeline_prompt keeps strategize (non-producer) short;
# compute (producer, exempt from concision) gets max_new=2048 to do the full solve.
# Worst realistic case: q(<=791) + strat(~400) + overhead(~250) + compute(2048)
# ~= 3.5K < 4096. Normal temp sampling (no forced-greedy cap). Truncation-check after.
#
# Set WAIT_PID to defer start until that process exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] pair_sc waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting pair_sc (text MAS)"
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
MAX_NEW="${MAX_NEW:-2048}"   # compute (producer) budget; concision keeps strategize short
CONCISE="${CONCISE:---concise_pipeline_prompt}"  # soft concision on strategize (non-producer)
BS="${BS:-2}"
PIPELINE="strategize,compute"
OUT=/home/bill/src/LatentMAS/results/math500_pair_sc
mkdir -p "$OUT"
CSV="$OUT/math500_pair_sc_${SUBSET}_results.csv"
echo "cell,subset,levels,method,pipeline,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

LOG="$OUT/math500_pair_sc_${SUBSET}.log"
echo "[$(date +%T)] pair_sc (text MAS, $PIPELINE, subset=$SUBSET, levels=$LEVELS, max_new=$MAX_NEW) START"
python run.py --task math500 --data_subset "$SUBSET" --data_levels "$LEVELS" \
    --train_size "$TRAIN_SIZE" --test_size "$TEST_SIZE" \
    --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
    --model_name "$MODEL" --method text_mas --seed 42 \
    --prompt sequential --pipeline "$PIPELINE" $CONCISE > "$LOG" 2>&1
JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] pair_sc DONE: acc=$ACC sec=$SEC"
echo "pair_sc,$SUBSET,\"$LEVELS\",text_mas,\"$PIPELINE\",$MAX_NEW,$BS,$SEC,$ACC,$COR" >> "$CSV"

echo ""
echo "=== MATH-500 pair_sc (text MAS, strategize->compute, levels=$LEVELS, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "Compare vs floor_instruct 0.756 / floor_agents ~0.70 / ceiling 0.893 (results/math500_anchors)"
echo "=== MATH-500 pair_sc COMPLETE $(date +%T) ==="
