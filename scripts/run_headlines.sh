#!/usr/bin/env bash
# Headline 3-way comparison: baseline / text_mas / latent_mas on a single task.
# Useful for quick sanity-check that all three pipelines work and to see
# accuracy-vs-time tradeoffs at a given config.
source "$(dirname "$0")/_common.sh"

OUT="/tmp/headlines_${MODEL//\//_}_${TASK}.csv"
echo "method,elapsed_sec,accuracy,correct" > "$OUT"

run_cell() {
    local label="$1"; shift
    local log="/tmp/headlines_${MODEL//\//_}_${TASK}_${label}.log"
    echo "[$(date +%T)] $label START"
    python run.py --task "$TASK" --max_samples "$N" --max_new_tokens "$MAX_NEW" \
        --model_name "$MODEL" --generate_bs "$BS" "$@" > "$log" 2>&1
    local acc=$(parse_json_field "$log" accuracy)
    local sec=$(parse_json_field "$log" total_time_sec)
    local cor=$(parse_json_field "$log" correct)
    echo "[$(date +%T)] $label DONE acc=$acc sec=$sec"
    echo "$label,$sec,$acc,$cor" >> "$OUT"
}

# text_mas defaults to a smaller bs because each agent emits a full
# max_new_tokens budget and the KV cache balloons; override TEXT_MAS_BS to
# tune (e.g. TEXT_MAS_BS=2 for 14B-class models).
: "${TEXT_MAS_BS:=4}"

run_cell baseline   --method baseline
BS="$TEXT_MAS_BS" run_cell text_mas   --method text_mas
run_cell latent_mas --method latent_mas --latent_steps "$K"

echo "=== HEADLINES RESULTS ($MODEL, $TASK, N=$N, K=$K) ==="
column -t -s, "$OUT"
