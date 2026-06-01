#!/usr/bin/env bash
# Ablation sweep: replace latent vectors with {zero, shuffled, gaussian} to test
# whether the real latent path carries answer-relevant signal. If accuracy is
# unchanged across modes, the judger is routing around the latent path entirely.
source "$(dirname "$0")/_common.sh"

OUT="$RESULTS_DIR/sweep_ablation_${MODEL//\//_}_${TASK}.csv"
echo "ablation,elapsed_sec,accuracy" > "$OUT"

run_abl() {
    local mode="$1"
    local log="$RESULTS_DIR/sweep_ablation_${MODEL//\//_}_${mode}.log"
    echo "[$(date +%T)] ablation=$mode START"
    python run.py --method latent_mas --task "$TASK" --max_samples "$N" \
        --max_new_tokens "$MAX_NEW" --model_name "$MODEL" --generate_bs "$BS" \
        --latent_steps "$K" --latent_ablation "$mode" > "$log" 2>&1
    acc=$(parse_json_field "$log" accuracy)
    sec=$(parse_json_field "$log" total_time_sec)
    echo "[$(date +%T)] ablation=$mode DONE acc=$acc sec=$sec"
    echo "$mode,$sec,$acc" >> "$OUT"
}

run_abl none
run_abl zero
run_abl shuffle
run_abl gaussian

echo "=== ABLATION SWEEP RESULTS ($MODEL, $TASK, N=$N, K=$K) ==="
column -t -s, "$OUT"
