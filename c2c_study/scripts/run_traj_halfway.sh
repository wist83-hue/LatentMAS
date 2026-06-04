#!/usr/bin/env bash
set -u; cd /home/bill/src/C2C
source /home/bill/anaconda3/etc/profile.d/conda.sh; conda activate rosetta
export HF_HOME=/home/bill/src/LatentMAS/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for step in 150 300; do
  echo "[$(date +%T)] traj ckpt-$step START (concurrent w/ training)"
  python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/repl_traj_${step}.yaml > traj_ckpt${step}.log 2>&1
  echo "[$(date +%T)] traj ckpt-$step DONE rc=$? oom=$(grep -c OutOfMemory traj_ckpt${step}.log)"
done
python - <<'PY'
import re
def micro(f):
    d={}
    for s,p,n,k in re.findall(r"(\w+) accuracy: ([\d.]+)% \(evaluated on (\d+) samples, skipped (\d+)\)", open(f).read()): d[s]=(float(p),int(n))
    c=sum(round(p/100*n) for p,n in d.values()); t=sum(n for _,n in d.values()); return (100*c/t if t else 0), len(d)
print("\n=== IDENTICAL TRAJECTORY (mmlu-redux, receiver=sharer=Qwen3-0.6B) ===")
print("  50k  effective  -> 37.25")
for step,eff in [(150,"~88k"),(300,"~127k")]:
    try: a,n=micro(f"traj_ckpt{step}.log"); print(f"  ckpt-{step} ({eff}) -> {a:.2f}  ({n} subj)")
    except Exception as e: print(f"  ckpt-{step}: ERR {e}")
print("  [baselines: Single 35.07 | C2C+0.5B 42.70]")
PY
echo "=== TRAJ HALFWAY DONE ==="
