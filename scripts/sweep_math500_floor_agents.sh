#!/bin/bash
# MATH-500 FLOOR_AGENTS: the strategize -> compute -> verify persona DAG run as TEXT
# MAS (agents pass TEXT, not latent KV) on Qwen2.5-Math-7B-Instruct.
#
# Role in the experiment (all on the level-4+5 split, 131 train / 131 test):
#   floor_instruct    = single-agent Instruct baseline        (results/math500_anchors)
#   floor_agents      = 3 personae as TEXT MAS  <-- THIS       isolates "text decomposition"
#   ceiling_r1distill = R1-Distill                             (results/math500_anchors)
#   latent (K-sweep)  = same 3 personae as LATENT MAS          tests latent KV vs text
#
# Flow (sequential DAG): strategize emits a strategy (text) -> compute reads problem
# + strategy and emits a computation (text) -> verify reads problem + strategy +
# computation and emits the final \boxed answer. This is the target the DSPy/GEPA
# prompt optimization will later try to lift; for now the prompts are simple seeds.
#
# Budget (the 4096 squeeze): the Instruct model has a HARD 4,096-token context window
# and text MAS ACCUMULATES each agent's output into the next prompt. With train+test
# prompts <=~800 tok, a uniform max_new=896 keeps the worst case (verify reading
# question + strategy + computation, then generating) at ~3.7K < 4096. Normal temp
# sampling for all agents (no forced-greedy nonjudger cap). NOTE: text MAS can't match
# the single agent's token budget inside 4096 (3 turns must share the window) — if
# floor_agents underperforms floor_instruct, that IS a finding (the squeeze, which
# motivates latent MAS / fewer agents). Re-run the truncation check after this run.
#
# Set WAIT_PID to defer start until that process (the anchors) exits.

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] floor_agents waiting for PID $WAIT_PID (anchors) to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting floor_agents (text MAS)"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
N="${N:-131}"
# SUBSET picks the seeded MATH-500 split: tune on 'train', report final on 'test'.
SUBSET="${SUBSET:-train}"
LEVELS="${LEVELS:-4,5}"
TRAIN_SIZE="${TRAIN_SIZE:-131}"
TEST_SIZE="${TEST_SIZE:-131}"
MAX_NEW="${MAX_NEW:-896}"   # uniform per-agent cap; keeps verify <4096 (see header)
BS="${BS:-2}"
PIPELINE="strategize,compute,verify"
OUT=/home/bill/src/LatentMAS/results/math500_floor_agents
mkdir -p "$OUT"
CSV="$OUT/math500_floor_agents_${SUBSET}_results.csv"
echo "floor,subset,levels,method,pipeline,max_new,bs,elapsed_sec,accuracy,correct" > "$CSV"

LOG="$OUT/math500_floor_agents_${SUBSET}.log"
echo "[$(date +%T)] floor_agents (text MAS, $PIPELINE, subset=$SUBSET, levels=$LEVELS, max_new=$MAX_NEW) START"
python run.py --task math500 --data_subset "$SUBSET" --data_levels "$LEVELS" \
    --train_size "$TRAIN_SIZE" --test_size "$TEST_SIZE" \
    --max_samples "$N" --max_new_tokens "$MAX_NEW" --generate_bs "$BS" \
    --model_name "$MODEL" --method text_mas --seed 42 \
    --prompt sequential --pipeline "$PIPELINE" > "$LOG" 2>&1
JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] floor_agents DONE: acc=$ACC sec=$SEC"
echo "floor_agents,$SUBSET,\"$LEVELS\",text_mas,\"$PIPELINE\",$MAX_NEW,$BS,$SEC,$ACC,$COR" >> "$CSV"

echo ""
echo "=== MATH-500 floor_agents (text MAS, strategize->compute->verify, levels=$LEVELS, subset=$SUBSET, n=$N) ==="
column -t -s, "$CSV"
echo "floor_instruct + ceiling: results/math500_anchors/math500_anchors_${SUBSET}_results.csv"
echo "=== MATH-500 floor_agents COMPLETE $(date +%T) ==="
