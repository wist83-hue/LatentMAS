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
# and text MAS ACCUMULATES each agent's output into the next prompt. The first run at
# uniform max_new=896 TRUNCATED the verify agent on 26/131 problems (verify ran out
# mid-computation re-deriving the answer) -> floor_agents 0.595, artificially below the
# 0.756 single-agent floor. FIX (the concision contingency): --concise_pipeline_prompt
# adds a SOFT 'essential steps only' instruction to strategize/compute (NOT verify), so
# they stay short and verify keeps room; max_new raised to 1536 so verify can finish.
# Normal temp sampling for all agents. Worst realistic case 791(q)+~450+~450+250+1536
# ~= 3.5K < 4096; re-run the truncation check after to confirm verify no longer cuts off.
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
MAX_NEW="${MAX_NEW:-1536}"  # verify budget; concision keeps strat/compute short (see header)
CONCISE="${CONCISE:---concise_pipeline_prompt}"  # soft concision on strat/compute; set CONCISE="" to disable
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
    --prompt sequential --pipeline "$PIPELINE" $CONCISE > "$LOG" 2>&1
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
