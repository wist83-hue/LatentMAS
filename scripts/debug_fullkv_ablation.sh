#!/usr/bin/env bash
# #2 control: is the full-KV +10pp from the latent VECTORS or the restored strategize TEXT?
# Run full-KV (keep strategize prompt) K=10, greedy, train idx 0-19, bs=20 (full-KV is batch-stable):
#   (a) REAL latent vectors   (b) GAUSSIAN-ablated latent vectors (random, magnitude-matched)
# If gaussian ~= real, the +10pp over latent_only is the restored TEXT, not the latent content.
set -u
cd /home/bill/src/LatentMAS
source /home/bill/anaconda3/etc/profile.d/conda.sh
conda activate latentmas
export HF_HOME=/home/bill/src/LatentMAS/huggingface
OUT=results/debug5/ablation_gaussian
mkdir -p "$OUT"

COMMON=(--task math500 --data_subset train --data_levels 4,5
    --train_size 131 --test_size 131
    --max_samples 20 --max_new_tokens 3072 --generate_bs 20
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" --method latent_mas --seed 42 --greedy
    --prompt sequential --pipeline strategize,solve --latent_steps 10
    --latent_feedback_mode auto --latent_space_realign --latent_norm_mode scalar_mean)

for MODE in none gaussian; do
    LOG="$OUT/fullkv_${MODE}_K10_idx0-19.log"
    echo "[$(date +%T)] full-KV ablation=$MODE START"
    python run.py "${COMMON[@]}" --latent_ablation "$MODE" > "$LOG" 2>&1
    JSON=$(grep -E '^\{.*accuracy' "$LOG" | tail -1)
    ACC=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['accuracy'])" 2>/dev/null || echo "?")
    COR=$(echo "$JSON" | python -c "import sys,json; print(json.load(sys.stdin)['correct'])" 2>/dev/null || echo "?")
    echo "[$(date +%T)] full-KV ablation=$MODE DONE: acc=$ACC correct=$COR"
done
echo "=== DONE ==="
