#!/usr/bin/env bash
# QUEUED: big->little C2C test. Holds receiver=Qwen3-0.6B fixed, varies sharer (4B, 4B-Base).
# Tests whether sharer SOPHISTICATION transfers, vs the trained-adapter effect.
# Compare vs: Single 35.07 | Identical(self) 37.25 | C2C-0.5B sharer 42.70  (all same receiver, mmlu-redux, pinned 57 subj)
# Waits for the identical-250k training to free the GPU (4B sharer ~10GB won't fit alongside the 16GB train).
set -u
cd /home/bill/src/C2C
source /home/bill/anaconda3/etc/profile.d/conda.sh
conda activate rosetta
export HF_HOME=/home/bill/src/LatentMAS/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASEHUB=/home/bill/src/LatentMAS/huggingface/hub

echo "[$(date +%T)] queued: waiting for SFT_train to finish (GPU free)..."
until ! pgrep -f "SFT_train.py" >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%T)] GPU free."
echo "[$(date +%T)] waiting for Qwen3-4B-Base download..."
until ls $BASEHUB/models--Qwen--Qwen3-4B-Base/snapshots/*/config.json >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%T)] 4B-Base ready."
sleep 10

run () {  # name recipe
    echo "[$(date +%T)] $1 START"
    python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/$2 > $1.log 2>&1
    echo "[$(date +%T)] $1 DONE rc=$?"
}
run big2little_4b      repl_big2little_4b.yaml
run big2little_4bbase  repl_big2little_4bbase.yaml

python - <<'PY'
import re,glob
def micro(f):
    d={}
    for s,p,n,k in re.findall(r"(\w+) accuracy: ([\d.]+)% \(evaluated on (\d+) samples, skipped (\d+)\)", open(f).read()): d[s]=(float(p),int(n))
    c=sum(round(p/100*n) for p,n in d.values()); t=sum(n for _,n in d.values()); return (100*c/t if t else 0), len(d)
print("\n=== SHARER-SIZE COMPARISON (receiver=Qwen3-0.6B, mmlu-redux) ===")
print("  Single (no sharer)      35.07")
print("  Identical (self, 50k)   37.25")
print("  C2C  + Qwen2.5-0.5B      42.70")
for nm,f in [("C2C  + Qwen3-4B        ","big2little_4b.log"),("C2C  + Qwen3-4B-Base   ","big2little_4bbase.log")]:
    try: a,n=micro(f); print(f"  {nm} {a:5.2f}   ({n} subj)")
    except Exception as e: print(f"  {nm} ERR {e}")
PY
echo "=== BIG2LITTLE QUEUE DONE ==="
