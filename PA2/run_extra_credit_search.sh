#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p runs/logs runs/npz

# Fast screening (short runs) to find configs that escape NLL ~1.386
COMMON=(
  --task torder --model rnn --init smart_tanh
  --nhid 50 --bs 20 --min_length 50 --max_length 200
  --maxiters 3000 --ebs 2000 --cbs 200 --checkFreq 20
  --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
)

run_one () {
  local name="$1"; shift
  echo "=== START $name ==="
  python -u train.py "$@" "${COMMON[@]}" --name "runs/npz/${name}" 2>&1 | tee "runs/logs/${name}.log"
  echo "=== DONE  $name ==="
}

# Baseline problematic config
run_one EC_s0_alpha2_cut005_lr001  --alpha 2.0 --clipstyle rescale --cutoff 0.05 --lr 0.01

# Try weaker omega with same clipping
run_one EC_s1_alpha1_cut005_lr001  --alpha 1.0 --clipstyle rescale --cutoff 0.05 --lr 0.01
run_one EC_s2_alpha05_cut005_lr001 --alpha 0.5 --clipstyle rescale --cutoff 0.05 --lr 0.01
run_one EC_s3_alpha0_cut005_lr001  --alpha 0.0 --clipstyle rescale --cutoff 0.05 --lr 0.01

# Try less aggressive clipping
run_one EC_s4_alpha2_cut010_lr001  --alpha 2.0 --clipstyle rescale --cutoff 0.10 --lr 0.01
run_one EC_s5_alpha2_cut020_lr001  --alpha 2.0 --clipstyle rescale --cutoff 0.20 --lr 0.01

# Try lower LR with original alpha/cutoff
run_one EC_s6_alpha2_cut005_lr0005 --alpha 2.0 --clipstyle rescale --cutoff 0.05 --lr 0.005

# Try no clipping with omega
run_one EC_s7_alpha2_noclip_lr001  --alpha 2.0 --clipstyle nothing --lr 0.01
