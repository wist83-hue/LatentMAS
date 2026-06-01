# Shared setup sourced by all sweep scripts.
# Usage at top of each script:   source "$(dirname "$0")/_common.sh"
#
# Honored env vars (override at invocation):
#   CONDA_BASE   default: $HOME/anaconda3
#   CONDA_ENV    default: latentmas
#   HF_HOME      default: <repo>/huggingface
#   MODEL        default: Qwen/Qwen3-4B
#   TASK         default: gsm8k
#   N            default: 64  (max_samples; -1 for full dataset)
#   K            default: 20  (latent_steps where applicable)
#   BS           default: 8
#   MAX_NEW      default: 2048
#   RESULTS_DIR  default: <repo>/results  (persistent; never /tmp)

set -uo pipefail
: "${CONDA_BASE:=$HOME/anaconda3}"
: "${CONDA_ENV:=latentmas}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
: "${HF_HOME:=$REPO_DIR/huggingface}"
export HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_DIR"

# Persistent output dir for CSVs + per-cell logs. Deliberately NOT /tmp:
# /tmp is cleared on reboot and a power outage once wiped a full overnight
# sweep's results. Override RESULTS_DIR to relocate. (results/ is gitignored.)
: "${RESULTS_DIR:=$REPO_DIR/results}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

: "${MODEL:=Qwen/Qwen3-4B}"
: "${TASK:=gsm8k}"
: "${N:=64}"
: "${K:=20}"
: "${BS:=8}"
: "${MAX_NEW:=2048}"

# Pull accuracy + elapsed_sec out of run.py's final JSON line.
parse_json_field() {
    local file="$1" field="$2"
    grep -E '^\{.*accuracy' "$file" | tail -1 | python -c "
import sys, json
try:
    print(json.load(sys.stdin).get('$field', '?'))
except Exception:
    print('?')
" 2>/dev/null
}

echo "[$(basename "$0")] MODEL=$MODEL TASK=$TASK N=$N K=$K BS=$BS MAX_NEW=$MAX_NEW"
