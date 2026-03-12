#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p runs/logs runs/npz

COMMON=(
  --task torder --model rnn --init smart_tanh
  --min_length 50 --max_length 200
  --bs 20 --ebs 5000 --cbs 500 --checkFreq 20
  --maxiters 3000
  --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
)

run_one() {
  local name="$1"; shift
  echo "=== START $name ==="
  python -u train.py "$@" "${COMMON[@]}" --name "runs/npz/${name}" 2>&1 | tee "runs/logs/${name}.log"
  echo "=== DONE  $name ==="
}

# Baseline reference
run_one EC_fair_s0_n50_lr001_cut005_a2  --nhid 50  --lr 0.01  --clipstyle rescale --cutoff 0.05 --alpha 2.0

# Remove omega, relax clipping
run_one EC_fair_s1_n50_lr001_cut02_a0   --nhid 50  --lr 0.01  --clipstyle rescale --cutoff 0.20 --alpha 0.0
run_one EC_fair_s2_n50_lr002_cut02_a0   --nhid 50  --lr 0.02  --clipstyle rescale --cutoff 0.20 --alpha 0.0
run_one EC_fair_s3_n50_lr0005_cut02_a0  --nhid 50  --lr 0.005 --clipstyle rescale --cutoff 0.20 --alpha 0.0

# Larger hidden state
run_one EC_fair_s4_n100_lr001_cut02_a0  --nhid 100 --lr 0.01  --clipstyle rescale --cutoff 0.20 --alpha 0.0
run_one EC_fair_s5_n100_lr001_cut05_a0  --nhid 100 --lr 0.01  --clipstyle rescale --cutoff 0.50 --alpha 0.0

# Stronger capacity + no clipping comparison
run_one EC_fair_s6_n200_lr001_cut05_a0  --nhid 200 --lr 0.01  --clipstyle rescale --cutoff 0.50 --alpha 0.0
run_one EC_fair_s7_n100_lr001_noclip_a0 --nhid 100 --lr 0.01  --clipstyle nothing               --alpha 0.0
