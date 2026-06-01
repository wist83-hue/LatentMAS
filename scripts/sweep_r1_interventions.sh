#!/bin/bash
# R1-Distill-Qwen-7B intervention sweep (cells A-J), GSM8K, n=64, bs=4, seed=42.
#
# Rebuilt after a power outage wiped the original /tmp chain scripts mid-run
# (the sweep had only just started cells A-E). This is the consolidated,
# self-contained version: all 10 cells in one script, no inter-process PID
# chaining, and outputs written to a PERSISTENT results/ dir rather than /tmp
# (which the reboot cleared).
#
# What it isolates:
#   A baseline (no latent, no agents)        -> reconfirms R1 baseline ~0.844
#   B multi-agent + K=0                       -> is the prompting alone the problem?
#   C default + minimal prompts + argmax K=20 -> does minimal prompting help?
#   D single-persona + argmax K=20            -> does dropping critic/refiner help?
#   E single-persona + minimal + argmax K=20  -> combined
#   F default + per-persona <think> brackets  -> do brackets alone help?
#   G default + brackets + minimal            -> brackets + minimal
#   H single-persona + brackets + minimal     -> maximal stack
#   I default + GLOBAL <think> wrap + argmax  -> one thinking block over all 3 personas
#   J global brackets + minimal               -> cleanest "R1-friendly" combo
#
# Reference (prior measurements):
#   baseline 0.844 | default+w_a K=20 0.703 | default+argmax K=20 0.766 | default+soft_embed K=20 0.719

set -uo pipefail

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUT=/home/bill/src/LatentMAS/results/r1_interventions
mkdir -p "$OUT"
CSV="$OUT/r1_interventions_results.csv"

echo "label,description,elapsed_sec,accuracy" > "$CSV"

run_cell() {
    local label="$1" desc="$2"; shift 2
    LOG="$OUT/r1_intervention_${label}.log"
    echo "[$(date +%T)] $label START"
    python run.py --task gsm8k --max_samples 64 --max_new_tokens 2048 --generate_bs 4 \
        --model_name "$MODEL" --seed 42 "$@" > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] $label DONE: acc=$ACC sec=$SEC"
    echo "$label,\"$desc\",$SEC,$ACC" >> "$CSV"
}

COMMON_LATENT="--method latent_mas --latent_steps 20 --latent_feedback_mode argmax_embed --latent_space_realign --latent_norm_mode scalar_mean"

# A: baseline (reconfirms ~0.844)
run_cell A_baseline "single-agent baseline, no latent_mas, no interventions" \
    --method baseline

# B: multi-agent + K=0 -- is the multi-agent prompting itself what hurts R1?
run_cell B_multiagent_K0 "default pipeline, 0 latent steps - tests prompting overhead alone" \
    --method latent_mas --latent_steps 0 --latent_space_realign --latent_norm_mode scalar_mean

# C: default + minimal prompts + argmax_embed K=20
run_cell C_minimal_argmax_K20 "default pipeline + minimal prompts + argmax_embed K=20" \
    $COMMON_LATENT --minimal_persona_prompts

# D: single-persona pipeline + argmax_embed K=20
run_cell D_single_persona_argmax_K20 "single-persona pipeline (planner,judger) + argmax_embed K=20" \
    $COMMON_LATENT --pipeline "planner,judger"

# E: single-persona + minimal + argmax_embed K=20
run_cell E_single_minimal_argmax_K20 "single-persona + minimal + argmax_embed K=20" \
    $COMMON_LATENT --pipeline "planner,judger" --minimal_persona_prompts

# F: default + per-persona <think> brackets
run_cell F_brackets_argmax_K20 "default pipeline + <think> brackets + argmax_embed K=20" \
    $COMMON_LATENT --latent_thinking_brackets

# G: default + brackets + minimal
run_cell G_brackets_minimal_argmax_K20 "default pipeline + brackets + minimal + argmax_embed K=20" \
    $COMMON_LATENT --latent_thinking_brackets --minimal_persona_prompts

# H: single-persona + brackets + minimal (maximal stack)
run_cell H_all_interventions "single-persona + brackets + minimal + argmax_embed K=20" \
    $COMMON_LATENT --pipeline "planner,judger" --latent_thinking_brackets --minimal_persona_prompts

# I: default + GLOBAL <think> wrap
run_cell I_global_brackets_argmax_K20 "default pipeline + GLOBAL brackets + argmax_embed K=20" \
    $COMMON_LATENT --latent_thinking_brackets_global

# J: global brackets + minimal
run_cell J_global_brackets_minimal_argmax_K20 "global brackets + minimal prompts + argmax_embed K=20" \
    $COMMON_LATENT --latent_thinking_brackets_global --minimal_persona_prompts

echo ""
echo "=== FULL R1-Distill-Qwen-7B GSM8K interventions matrix ==="
column -t -s, "$CSV"
echo ""
echo "Reference baseline: 0.844"
echo "=== R1 INTERVENTIONS SWEEP COMPLETE $(date +%T) ==="
