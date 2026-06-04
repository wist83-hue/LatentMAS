#!/usr/bin/env bash
set -u
cd /home/bill/src/C2C
source /home/bill/anaconda3/etc/profile.d/conda.sh
conda activate rosetta
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cell in c2c single; do
  echo "[$(date +%T)] $cell START"
  python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/repl_${cell}_mmlu_full.yaml > repl_${cell}_mmlu_full.log 2>&1
  rc=$?
  acc=$(grep -E "Overall accuracy" repl_${cell}_mmlu_full.log | tail -1)
  echo "[$(date +%T)] $cell DONE rc=$rc | $acc"
done
echo "=== ALL DONE ===  paper: receiver-only 35.53 | C2C 42.92"
