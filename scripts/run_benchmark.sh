#!/usr/bin/env bash
# scripts/run_benchmark.sh — Vectro+ ground-truth recall & QPS benchmark
#
# Usage:
#   ./scripts/run_benchmark.sh                        # synthetic 10k × 128-d (default)
#   ./scripts/run_benchmark.sh --dataset my.stream1   # real dataset
#   ./scripts/run_benchmark.sh --vectors 50000 --dim 768 --queries 500 --save-report
#
# Prerequisites: cargo must be in PATH. Run from the vectro-plus workspace root.
#
# The script:
#   1. Builds the vectro binary in release mode.
#   2. Runs `vectro bench-gt` with the given (or default) arguments.
#   3. Verifies exit code and, when --save-report is set, prints the report path.
#
# Exit codes:
#   0 — all recall gates passed
#   1 — user / argument error
#   2 — runtime error (build failure, bench failure, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────────────
VECTORS=10000
DIM=128
QUERIES=100
K=10
DATASET=""
SAVE_REPORT=false
EXTRA_ARGS=()

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vectors)       VECTORS="$2";     shift 2 ;;
    --dim)           DIM="$2";         shift 2 ;;
    --queries)       QUERIES="$2";     shift 2 ;;
    --k)             K="$2";           shift 2 ;;
    --dataset)       DATASET="$2";     shift 2 ;;
    --save-report)   SAVE_REPORT=true; shift   ;;
    --help|-h)
      sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Step 1: Build release binary ─────────────────────────────────────────────
echo "▶ Building vectro (release)…"
if ! cargo build --release -p vectro_cli 2>&1; then
  echo "Build failed." >&2
  exit 2
fi
VECTRO="$ROOT_DIR/target/release/vectro"

# ── Step 2: Assemble bench-gt arguments ──────────────────────────────────────
BENCH_ARGS=(bench-gt --k "$K" --queries "$QUERIES")

if [[ -n "$DATASET" ]]; then
  BENCH_ARGS+=(--dataset "$DATASET")
else
  BENCH_ARGS+=(--vectors "$VECTORS" --dim "$DIM")
fi

if $SAVE_REPORT; then
  BENCH_ARGS+=(--save-report)
fi

# ── Step 3: Run benchmark ─────────────────────────────────────────────────────
echo "▶ Running: vectro ${BENCH_ARGS[*]}"
echo ""
if ! "$VECTRO" "${BENCH_ARGS[@]}"; then
  echo "bench-gt reported a recall gate failure — see output above." >&2
  exit 2
fi

echo ""
echo "✅ run_benchmark.sh complete."
