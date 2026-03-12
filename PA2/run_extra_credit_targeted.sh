#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p runs/logs runs/npz

COMMON=(
  --task torder --model rnn --init smart_tanh
  --nhid 50 --bs 20 --min_length 50 --max_length 200
  --maxiters 2000 --ebs 2000 --cbs 200 --checkFreq 20
  --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
)

run_one () {
  local name="$1"; shift
  echo "=== START $name ==="
  python -u train.py "$@" "${COMMON[@]}" --name "runs/npz/${name}" 2>&1 | tee "runs/logs/${name}.log"
  echo "=== DONE  $name ==="
}

run_one EC_t1_alpha2_cut020_lr001  --alpha 2.0 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_t2_alpha05_cut020_lr001 --alpha 0.5 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_t3_alpha01_cut020_lr001 --alpha 0.1 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_t4_alpha0_cut020_lr001  --alpha 0.0 --clipstyle rescale --cutoff 0.20 --lr 0.01
run_one EC_t5_alpha0_noclip_lr001  --alpha 0.0 --clipstyle nothing --lr 0.01
