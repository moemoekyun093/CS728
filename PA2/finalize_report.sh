#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

python analyze_results.py
python build_report.py

if command -v pdflatex >/dev/null 2>&1; then
  cd report
  pdflatex -interaction=nonstopmode pa2_report.tex >/tmp/pa2_report_build1.log 2>&1 || true
  pdflatex -interaction=nonstopmode pa2_report.tex >/tmp/pa2_report_build2.log 2>&1 || true
  cd - >/dev/null
  echo "PDF build attempted: report/pa2_report.pdf"
else
  echo "pdflatex not found. Generated LaTeX source: report/pa2_report.tex"
fi

echo "Done: summary in runs/summary.{json,csv}, figures in runs/figures, report in report/"
