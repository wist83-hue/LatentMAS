#!/usr/bin/env bash
# Compare 4 cells testing whether "latent reasoning" (argmax_embed K=N) is
# fundamentally different from "short greedy text reasoning" (text_mas with
# --text_mas_nonjudger_max_tokens=N).
#
# argmax_embed is mathematically equivalent to: per-agent greedy decode capped
# at N tokens, then hide the tokens (just don't show them to the user). The
# only difference is bookkeeping. So if text_mas_short ≈ argmax_embed under
# matched prompting, the LatentMAS paper's "latent reasoning" claim reduces to
# "use shorter agent outputs."
source "$(dirname "$0")/_common.sh"

OUT="/tmp/compare_short_${MODEL//\//_}_${TASK}.csv"
echo "label,concise,elapsed_sec,accuracy" > "$OUT"

run_cell() {
    local label="$1" concise="$2"; shift 2
    local log="/tmp/compare_short_${MODEL//\//_}_${label}_c${concise}.log"
    local extra=""
    [ "$concise" = "y" ] && extra="--concise_nonjudger_prompt"
    echo "[$(date +%T)] $label concise=$concise START"
    python run.py --task "$TASK" --max_samples "$N" --max_new_tokens "$MAX_NEW" \
        --model_name "$MODEL" --generate_bs "$BS" $extra "$@" > "$log" 2>&1
    acc=$(parse_json_field "$log" accuracy)
    sec=$(parse_json_field "$log" total_time_sec)
    echo "[$(date +%T)] $label concise=$concise DONE acc=$acc sec=$sec"
    echo "$label,$concise,$sec,$acc" >> "$OUT"
}

# Cell 1: full-budget text_mas (upper-bound reference)
run_cell text_mas_full n --method text_mas
# Cell 2: short-budget text_mas + concise prompting (gives short text its best shot)
run_cell text_mas_short${K} y --method text_mas --text_mas_nonjudger_max_tokens "$K"
# Cell 3: argmax_embed latent without matched prompting
run_cell argmax_embed n --method latent_mas --latent_steps "$K" --latent_feedback_mode argmax_embed
# Cell 4: argmax_embed latent WITH matched concise prompting (apples-to-apples vs Cell 2)
run_cell argmax_embed_conc y --method latent_mas --latent_steps "$K" --latent_feedback_mode argmax_embed

echo "=== SHORT-TEXT vs LATENT RESULTS ($MODEL, $TASK, N=$N, K=$K) ==="
column -t -s, "$OUT"
