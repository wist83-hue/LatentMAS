#!/bin/bash
# AIME-24 on Qwen3-8B at the PAPER's token budget (max_new_tokens=8096), thinking ON.
#
# Why: we'd been running AIME at max_new_tokens=20000, but the paper specifies
# 8,096 for reasoning-intensive benchmarks (Implementation Details C.2). At 20K
# our thinking-ON baseline hit 70.0% (vs paper's Single 50.0) and LatentMAS K=10
# *lost* by -13pp. The paper ran thinking ON for BOTH baseline and latent (no
# enable_thinking toggle in their code) and reported Single 50.0 / LatentMAS 56.7
# (+6.7pp). This re-run matches the paper's budget to test whether the latent
# gain reproduces once the baseline is no longer over-budgeted at 20K.
#
# Protocol: thinking ON (default, NO --disable_thinking), temp/top-p left at
# code defaults, full AIME24 (n=30), seed 42, bs=1.
#
# Set WAIT_PID to a process id to defer the start until that process exits
# (used to chain this behind the R1 intervention sweep).
#
# Paper Qwen3-8B AIME24 (Table 3, sequential): Single 50.0 | TextMAS 53.3 | LatentMAS 56.7

set -uo pipefail

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date +%T)] waiting for PID $WAIT_PID (R1 sweep) to finish before starting AIME..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%T)] PID $WAIT_PID exited; starting AIME @ 8096 tokens"
fi

source /home/bill/anaconda3/etc/profile.d/conda.sh && conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bill/src/LatentMAS

MODEL="Qwen/Qwen3-8B"
OUT=/home/bill/src/LatentMAS/results/aime_8096
mkdir -p "$OUT"
CSV="$OUT/aime_8096_results.csv"

echo "model,mode,max_new,elapsed_sec,accuracy,correct" > "$CSV"

run_cell() {
    local label="$1"; shift
    LOG="$OUT/aime_8096_${label}.log"
    echo "[$(date +%T)] $label START"
    python run.py --task aime2024 --max_samples -1 --max_new_tokens 8096 --generate_bs 1 \
        --model_name "$MODEL" --seed 42 "$@" > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    SEC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] $label DONE: acc=$ACC sec=$SEC"
    echo "qwen3_8b,$label,8096,$SEC,$ACC,$COR" >> "$CSV"
}

# Direct comparison to our 20K numbers (baseline 0.700, latent K=10 0.5667), both thinking ON.
run_cell baseline      --method baseline
run_cell latent_K10    --method latent_mas --latent_steps 10

echo ""
echo "=== AIME24 Qwen3-8B @ max_new=8096 (thinking ON) ==="
column -t -s, "$CSV"
echo ""
echo "Reference -- paper: Single 50.0 | LatentMAS 56.7    ours@20K: baseline 70.0 | latent 56.67"
echo "=== AIME 8096 SWEEP COMPLETE $(date +%T) ==="
