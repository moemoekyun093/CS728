#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create it first." >&2
  exit 1
fi

source .venv/bin/activate

COMMON_ARGS=(
  --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200
  --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20
  --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
)

run_exp() {
  local run_name="$1"
  shift
  local log_path="runs/logs/${run_name}.log"
  local name_prefix="runs/npz/${run_name}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${run_name}"
  python -u train.py "$@" "${COMMON_ARGS[@]}" --name "$name_prefix" 2>&1 | tee "$log_path"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${run_name}"
}

# A1-A5 (mem)
run_exp A1_mem_rnn_tanh_noclip \
  --task mem --model rnn --alpha 0.0 --clipstyle nothing

run_exp A2_mem_rnn_tanh_clip005 \
  --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.05

run_exp A3_mem_rnn_tanh_clip001 \
  --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.01

run_exp A4_mem_gru_noclip \
  --task mem --model gru --alpha 0.0 --clipstyle nothing --diagGates

run_exp A5_mem_gru_clip005 \
  --task mem --model gru --alpha 0.0 --clipstyle rescale --cutoff 0.05 --diagGates

# B1-B2 (mul)
run_exp B1_mul_rnn_tanh_noclip \
  --task mul --model rnn --alpha 0.0 --clipstyle nothing

run_exp B2_mul_gru_noclip \
  --task mul --model gru --alpha 0.0 --clipstyle nothing --diagGates

