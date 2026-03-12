#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

RUN_ORDER = [
    ("A1_mem_rnn_tanh_noclip", "A1"),
    ("A2_mem_rnn_tanh_clip005", "A2"),
    ("A3_mem_rnn_tanh_clip001", "A3"),
    ("A4_mem_gru_noclip", "A4"),
    ("A5_mem_gru_clip005", "A5"),
    ("B1_mul_rnn_tanh_noclip", "B1"),
    ("B2_mul_gru_noclip", "B2"),
]

SUMMARY_CSV = Path("runs/summary.csv")
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEX_PATH = REPORT_DIR / "pa2_report.tex"

if not SUMMARY_CSV.exists():
    raise SystemExit("Missing runs/summary.csv. Run analyze_results.py first.")

rows = {}
with SUMMARY_CSV.open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows[r["run"]] = r


def fval(s: str) -> str:
    if s is None or s == "" or s == "None":
        return "N/A"
    try:
        return f"{float(s):.4f}"
    except Exception:
        return s


with TEX_PATH.open("w", encoding="utf-8") as f:
    f.write(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\title{CS728 PA2 Report: Training Dynamics of RNNs and GRUs}
\author{Student Name / Roll Number}
\date{\today}
\begin{document}
\maketitle

\section{Objective}
This report studies the training dynamics of vanilla RNNs and GRUs on long-range synthetic tasks, focusing on gradient-through-time behavior, saturation, clipping effects, and the recurrent spectral-radius proxy $\rho(W_{hh})$.

\section{Experimental Setup}
All required runs were executed with the exact assignment commands and shared hyperparameters:
\begin{verbatim}
--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200
--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20
--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05
\end{verbatim}

\section{Run Summary}
\begin{table}[H]
\centering
\begin{tabular}{lrrrrrr}
\toprule
Run & Best valid err(\%) & Last valid err(\%) & Last $\rho$ & Mean log10($g_t$) & Mean sat & z/r sat \\
\midrule
""")

    for run, short in RUN_ORDER:
        r = rows.get(run, {})
        zsat = fval(r.get("zsat_mean", ""))
        rsat = fval(r.get("rsat_mean", ""))
        zr = f"{zsat}/{rsat}" if zsat != "N/A" or rsat != "N/A" else "N/A"
        f.write(
            f"{short} & {fval(r.get('best_valid_error_pct',''))} & {fval(r.get('last_valid_error_pct',''))} & "
            f"{fval(r.get('last_rho',''))} & {fval(r.get('grad_log10_mean',''))} & {fval(r.get('sat_mean',''))} & {zr} \\\\\n"
        )

    f.write(r"""\bottomrule
\end{tabular}
\caption{Numerical summary from final checkpoints.}
\end{table}

\section{Per-run Plots and Observations}
For each run, we include: (i) histogram of $\log_{10}\|\partial L/\partial h_t\|$, (ii) hidden saturation-distance histogram, (iii) validation curve, and (iv) $\rho(W_{hh})$ curve. For GRU runs we also include gate saturation histograms.
""")

    for run, short in RUN_ORDER:
        f.write(f"\n\\subsection{{{short}: {run.replace('_', ' ')}}}\n")
        f.write(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_grad_hist.png}
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_sat_hist.png}
\caption{Gradient-through-time and hidden saturation histograms.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_valid_curve.png}
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_rho_curve.png}
\caption{Validation error and $\rho(W_{hh})$ over checkpoints.}
\end{figure}
""")
        if run in {"A4_mem_gru_noclip", "A5_mem_gru_clip005", "B2_mul_gru_noclip"}:
            f.write(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_zsat_hist.png}
\includegraphics[width=0.48\textwidth]{../runs/figures/""" + run + r"""_rsat_hist.png}
\caption{GRU gate saturation-distance histograms (update and reset gates).}
\end{figure}
""")

        f.write(
            "\\paragraph{Observation template} "
            "Discuss whether gradients are vanishing/exploding/spread, whether hidden units are saturated, "
            "and how this aligns (or does not align) with validation behavior for this run.\n"
        )

    f.write(r"""
\section{Required Questions}
\begin{enumerate}
\item For each run, interpret the $\log_{10}\|\partial L/\partial h_t\|$ histogram: vanishing vs exploding vs spread.
\item For each run, interpret hidden saturation-distance histogram and whether saturation always explains gradient behavior.
\item Compare clipping vs no clipping (A1 vs A2/A3; A4 vs A5): effects on gradient norm, gradient histogram, and saturation.
\item Compare RNN vs GRU (A1/A2/A3 vs A4/A5; B1 vs B2): relate gate saturation to gradient-through-time and discuss GRU-specific failure modes.
\item Report $\rho(W_{hh})$ trends and correlate with gradient dynamics.
\end{enumerate}

\section{Notes}
For memorization (classification), valid error is strict sequence-level error under report=all. For multiplication (regression), interpret valid error together with valid nll and threshold-based errors printed in logs.

\section{Appendix: Commands}
Attach the exact command log file submitted separately (generated from \texttt{run\_required\_experiments.sh}).

\end{document}
""")

print(f"Wrote {TEX_PATH}")
