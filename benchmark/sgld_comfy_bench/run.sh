#!/usr/bin/env bash
# Usage: ./run.sh <step> [args]
#   check            machine.json: hardware, versions, kernels, model files            (~1 min)
#   a [--gpus 1|2] [--only IDS]   track A (P0 sweep first, then 1-GPU, then 2-GPU)     (hours)
#   b [--gpus 1|2] [--only IDS]   track B (SGLang serve vs vllm-omni serve)           (hours)
#   functional       track A functional matrix (h200 and 5090)                         (~1 h)
#   quality          PSNR/SSIM of lossy configs vs their lossless baseline, frame sheets
#   report           SUITE/REPORT.md from what has run
# Results: $WORKSPACE/results/<CARD>/ (override with SUITE=...). Every step resumes: finished
# experiments are skipped, so re-run the same command after an interruption.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
SUITE="${SUITE:-$WORKSPACE/results/$CARD}"
mkdir -p "$SUITE"
step="${1:-}"; shift || true
case "$step" in
  check)      python bench/check_env.py --out "$SUITE" "$@" ;;
  a)          python bench/track_a.py --suite "$SUITE/A" "$@" ;;
  b)          python bench/track_b.py --suite "$SUITE/B" "$@" ;;
  functional) python bench/functional.py --suite "$SUITE/F" "$@" ;;
  quality)    python bench/quality.py --suite "$SUITE" "$@" ;;
  report)     python bench/report.py --suite "$SUITE" "$@" ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
echo "suite: $SUITE"
