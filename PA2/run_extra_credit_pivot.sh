#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p runs/logs runs/npz

COMMON=(
  --task torder --model rnn
  --nhid 50 --bs 20 --min_length 20 --max_length 80
  --maxiters 3000 --ebs 2000 --cbs 200 --checkFreq 20
  --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
)

run_one () {
  local name="$1"; shift
  echo "=== START $name ==="
  python -u train.py "$@" "${COMMON[@]}" --name "runs/npz/${name}" 2>&1 | tee "runs/logs/${name}.log"
  echo "=== DONE  $name ==="
}

run_one EC_p1_smart_a0_cut02_lr001 --init smart_tanh --alpha 0.0 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_p2_smart_a0_cut05_lr001 --init smart_tanh --alpha 0.0 --clipstyle rescale --cutoff 0.50 --lr 0.01
run_one EC_p3_basic_a0_cut02_lr001 --init basic_tanh --alpha 0.0 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_p4_smart_a0_noclip_lr001 --init smart_tanh --alpha 0.0 --clipstyle nothing --lr 0.01
run_one EC_p5_smart_a0_cut02_lr0005 --init smart_tanh --alpha 0.0 --clipstyle rescale --cutoff 0.20 --lr 0.005
run_one EC_p6_smart_a0_cut02_lr002 --init smart_tanh --alpha 0.0 --clipstyle rescale --cutoff 0.20 --lr 0.02
