#!/usr/bin/env bash
# #3 structural fix + #2 bs=1 confirmation, all greedy, train idx 0-19, K=10,
# strategize->solve. Slots into the batch-controlled decomposition table.
#   structural   = --latent_only --latent_in_producer_turn (latent INSIDE the
#                  producer assistant turn, no strategize text). Isolates PLACEMENT.
#   fullkv none/gaussian @ bs=1 = the #2 control re-confirmed off-batch.
set -u
cd /home/bill/src/LatentMAS
source /home/bill/anaconda3/etc/profile.d/conda.sh
conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
OUT=results/debug5/structural
mkdir -p "$OUT"

BASE=(--task math500 --data_subset train --data_levels 4,5
    --train_size 131 --test_size 131
    --max_samples 20 --max_new_tokens 3072
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" --method latent_mas --seed 42 --greedy
    --prompt sequential --pipeline strategize,solve --latent_steps 10
    --latent_feedback_mode auto --latent_space_realign --latent_norm_mode scalar_mean)

run () {  # name  log  extra-args...
    local name="$1"; shift
    local log="$1"; shift
    echo "[$(date +%T)] $name START"
    python run.py "${BASE[@]}" "$@" > "$log" 2>&1
    local js; js=$(grep -E '^\{.*accuracy' "$log" | tail -1)
    local acc cor; acc=$(echo "$js" | python -c "import sys,json;print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo '?')
    cor=$(echo "$js" | python -c "import sys,json;print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo '?')
    echo "[$(date +%T)] $name DONE: acc=$acc correct=$cor"
}

# #3 structural placement (latent inside producer turn), bs=1 and bs=4
run "structural bs1" "$OUT/structural_K10_bs1_idx0-19.log" --generate_bs 1 --latent_only --latent_in_producer_turn
run "structural bs4" "$OUT/structural_K10_bs4_idx0-19.log" --generate_bs 4 --latent_only --latent_in_producer_turn

# #2 bs=1 confirmation: full-KV real vs gaussian (drop --latent_only = keep strategize prompt)
run "fullkv-none bs1"     "$OUT/fullkv_none_K10_bs1_idx0-19.log"     --generate_bs 1 --latent_ablation none
run "fullkv-gaussian bs1" "$OUT/fullkv_gaussian_K10_bs1_idx0-19.log" --generate_bs 1 --latent_ablation gaussian

echo "=== ALL DONE ==="
