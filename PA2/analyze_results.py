#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    "A1_mem_rnn_tanh_noclip",
    "A2_mem_rnn_tanh_clip005",
    "A3_mem_rnn_tanh_clip001",
    "A4_mem_gru_noclip",
    "A5_mem_gru_clip005",
    "B1_mul_rnn_tanh_noclip",
    "B2_mul_gru_noclip",
]

NPZ_DIR = Path("runs/npz")
FIG_DIR = Path("runs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def last_finite_row(arr: np.ndarray) -> np.ndarray:
    # Pick the last row that has at least one finite entry.
    idx = -1
    for i in range(arr.shape[0] - 1, -1, -1):
        row = arr[i]
        if np.isfinite(row).any():
            idx = i
            break
    if idx < 0:
        return np.array([], dtype=np.float64)
    row = arr[idx]
    return row[np.isfinite(row)]


def finite_prefix(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    return x[np.isfinite(x) & (x >= 0)]


summaries = []

for run in RUNS:
    npz_path = NPZ_DIR / f"{run}_final_state.npz"
    if not npz_path.exists():
        summaries.append({"run": run, "status": "missing_npz"})
        continue

    z = np.load(npz_path, allow_pickle=True)
    grad_time = z["grad_time"]
    sat_time = z["sat_time"]
    valid_err = finite_prefix(z["valid_error"])
    rho = finite_prefix(z["rho_Whh"])

    g = last_finite_row(grad_time)
    s = last_finite_row(sat_time)

    zsat = None
    rsat = None
    if "gate_z_sat_time" in z.files:
        zsat = last_finite_row(z["gate_z_sat_time"])
    if "gate_r_sat_time" in z.files:
        rsat = last_finite_row(z["gate_r_sat_time"])

    # Only treat gate diagnostics as enabled when both gate traces contain
    # finite values in the saved state (equivalent to running with --diagGates).
    diag_gates_enabled = (
        zsat is not None
        and rsat is not None
        and zsat.size > 0
        and rsat.size > 0
    )

    # Gradient histogram
    if g.size > 0:
        plt.figure(figsize=(5, 3.2))
        plt.hist(np.log10(g + 1e-12), bins=60)
        plt.title(f"{run}: log10 ||dL/dh_t||")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_grad_hist.png", dpi=160)
        plt.close()

    # Hidden saturation histogram
    if s.size > 0:
        plt.figure(figsize=(5, 3.2))
        plt.hist(s, bins=60, range=(0, 1))
        plt.title(f"{run}: hidden saturation distance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_sat_hist.png", dpi=160)
        plt.close()

    # Validation error curve
    if valid_err.size > 0:
        plt.figure(figsize=(5, 3.2))
        plt.plot(valid_err)
        plt.title(f"{run}: validation error (%)")
        plt.xlabel("checkpoint")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_valid_curve.png", dpi=160)
        plt.close()

    # rho curve
    if rho.size > 0:
        plt.figure(figsize=(5, 3.2))
        plt.plot(rho)
        plt.title(f"{run}: rho_Whh")
        plt.xlabel("checkpoint")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_rho_curve.png", dpi=160)
        plt.close()

    # GRU gate histograms
    if diag_gates_enabled:
        plt.figure(figsize=(5, 3.2))
        plt.hist(zsat, bins=60, range=(0, 0.5))
        plt.title(f"{run}: z-gate saturation distance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_zsat_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 3.2))
        plt.hist(rsat, bins=60, range=(0, 0.5))
        plt.title(f"{run}: r-gate saturation distance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{run}_rsat_hist.png", dpi=160)
        plt.close()

    summaries.append(
        {
            "run": run,
            "status": "ok",
            "best_valid_error_pct": float(valid_err.min()) if valid_err.size else None,
            "last_valid_error_pct": float(valid_err[-1]) if valid_err.size else None,
            "last_rho": float(rho[-1]) if rho.size else None,
            "grad_log10_mean": float(np.log10(g + 1e-12).mean()) if g.size else None,
            "sat_mean": float(s.mean()) if s.size else None,
            "zsat_mean": float(zsat.mean()) if diag_gates_enabled else None,
            "rsat_mean": float(rsat.mean()) if diag_gates_enabled else None,
        }
    )

with open("runs/summary.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2)

with open("runs/summary.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "run",
        "status",
        "best_valid_error_pct",
        "last_valid_error_pct",
        "last_rho",
        "grad_log10_mean",
        "sat_mean",
        "zsat_mean",
        "rsat_mean",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in summaries:
        w.writerow(row)

print("Wrote runs/summary.json, runs/summary.csv and figures under runs/figures")
