#!/usr/bin/env bash
# K-sweep: vary --latent_steps across a fixed range, otherwise keep config.
# Tests whether latent reasoning helps with more steps or whether drift sets in.
source "$(dirname "$0")/_common.sh"

# Override K_VALUES to change the sweep range.
: "${K_VALUES:=5 10 20 40 80}"

OUT="/tmp/sweep_K_${MODEL//\//_}_${TASK}.csv"
echo "K,elapsed_sec,accuracy,correct" > "$OUT"

for k in $K_VALUES; do
    log="/tmp/sweep_K_${MODEL//\//_}_${TASK}_k${k}.log"
    echo "[$(date +%T)] K=$k START"
    python run.py --method latent_mas --task "$TASK" --max_samples "$N" \
        --max_new_tokens "$MAX_NEW" --model_name "$MODEL" --generate_bs "$BS" \
        --latent_steps "$k" > "$log" 2>&1
    acc=$(parse_json_field "$log" accuracy)
    sec=$(parse_json_field "$log" total_time_sec)
    cor=$(parse_json_field "$log" correct)
    echo "[$(date +%T)] K=$k DONE acc=$acc sec=$sec"
    echo "$k,$sec,$acc,$cor" >> "$OUT"
done

echo "=== K-SWEEP RESULTS ($MODEL, $TASK, N=$N) ==="
column -t -s, "$OUT"
