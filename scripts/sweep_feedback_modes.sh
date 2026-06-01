#!/usr/bin/env bash
# Compare --latent_feedback_mode {w_a, coconut, argmax_embed, soft_embed} at fixed K.
# This was the sweep that revealed soft_embed τ=2 beats both paper-faithful w_a
# AND baseline on Qwen3-4B GSM8K (0.94 vs 0.78 vs 0.89).
source "$(dirname "$0")/_common.sh"

OUT="$RESULTS_DIR/sweep_feedback_${MODEL//\//_}_${TASK}.csv"
echo "label,elapsed_sec,accuracy" > "$OUT"

run_mode() {
    local label="$1"; shift
    local log="$RESULTS_DIR/sweep_feedback_${MODEL//\//_}_${label}.log"
    echo "[$(date +%T)] $label START"
    python run.py --method latent_mas --task "$TASK" --max_samples "$N" \
        --max_new_tokens "$MAX_NEW" --model_name "$MODEL" --generate_bs "$BS" \
        --latent_steps "$K" "$@" > "$log" 2>&1
    acc=$(parse_json_field "$log" accuracy)
    sec=$(parse_json_field "$log" total_time_sec)
    echo "[$(date +%T)] $label DONE acc=$acc sec=$sec"
    echo "$label,$sec,$acc" >> "$OUT"
}

run_mode w_a               --latent_feedback_mode w_a
run_mode w_a_scalar_mean   --latent_feedback_mode w_a --latent_norm_mode scalar_mean
run_mode coconut           --latent_feedback_mode coconut
run_mode argmax_embed      --latent_feedback_mode argmax_embed
run_mode soft_embed_t0.5   --latent_feedback_mode soft_embed --latent_soft_embed_temperature 0.5
run_mode soft_embed_t1.0   --latent_feedback_mode soft_embed --latent_soft_embed_temperature 1.0
run_mode soft_embed_t2.0   --latent_feedback_mode soft_embed --latent_soft_embed_temperature 2.0

echo "=== FEEDBACK-MODE SWEEP RESULTS ($MODEL, $TASK, N=$N, K=$K) ==="
column -t -s, "$OUT"
