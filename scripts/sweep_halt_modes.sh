#!/usr/bin/env bash
# Compare the four latent-loop halt criteria at K_max=K.
# Tests which (if any) early-stopping signal preserves accuracy while saving time.
#
# Halt modes: none / velocity (thr=5.0) / entropy (thr=4.0) / argmax-stable / kl
source "$(dirname "$0")/_common.sh"

OUT="$RESULTS_DIR/sweep_halt_${MODEL//\//_}_${TASK}.csv"
echo "label,elapsed_sec,accuracy" > "$OUT"

# K should be ample enough that early-halt can fire (default K=100 for the sweep).
: "${HALT_K:=100}"

run_halt() {
    local label="$1"; shift
    local log="$RESULTS_DIR/sweep_halt_${MODEL//\//_}_${label}.log"
    echo "[$(date +%T)] $label START"
    python run.py --method latent_mas --task "$TASK" --max_samples "$N" \
        --max_new_tokens "$MAX_NEW" --model_name "$MODEL" --generate_bs "$BS" \
        --latent_steps "$HALT_K" "$@" > "$log" 2>&1
    acc=$(parse_json_field "$log" accuracy)
    sec=$(parse_json_field "$log" total_time_sec)
    echo "[$(date +%T)] $label DONE acc=$acc sec=$sec"
    echo "$label,$sec,$acc" >> "$OUT"
}

run_halt none
run_halt velocity_5.0     --latent_halt_threshold 5.0
run_halt entropy_4.0      --latent_halt_entropy_nats 4.0
run_halt argmax_stable_3  --latent_halt_argmax_steps 3
run_halt kl_0.1           --latent_halt_kl_nats 0.1

echo "=== HALT-MODE SWEEP RESULTS ($MODEL, $TASK, N=$N, K_max=$HALT_K) ==="
column -t -s, "$OUT"
