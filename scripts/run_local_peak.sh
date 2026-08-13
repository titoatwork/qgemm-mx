#!/usr/bin/env bash
# run_local_peak.sh — local smoke: CPU + GPU tests, occupancy table, optional benches.
#
# Fail-fast on tests and on any bench that is actually invoked. Short benches
# run only when the binary is already present (this script does not build them).
#
# Usage:
#   bash scripts/run_local_peak.sh
#   ARCH=sm_86 QGEMM_MEASURED_BW_GBPS=124.2 bash scripts/run_local_peak.sh
#   SKIP_BENCH=1 bash scripts/run_local_peak.sh
#
# Short benches need a measured RO denominator (same contract as the harness).
# Set QGEMM_MEASURED_BW_GBPS or they are skipped.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCH="${ARCH:-sm_86}"
DEV="${DEV:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
PY="${PYTHON:-python3}"

echo "=== local peak ==="
echo "ARCH=$ARCH  DEV=$DEV  SKIP_BENCH=$SKIP_BENCH"
echo

echo "=== make test-cpu ==="
make ARCH="$ARCH" test-cpu
echo

echo "=== make test-gpu ==="
make ARCH="$ARCH" test-gpu
echo

echo "=== occupancy_table.py ==="
"$PY" python/occupancy_table.py
echo

if [[ "$SKIP_BENCH" == "1" ]]; then
  echo "short benches skipped (SKIP_BENCH=1)."
  echo "local peak ok."
  exit 0
fi

echo "=== short benches (existing binaries only) ==="
BW="${QGEMM_MEASURED_BW_GBPS:-}"
if [[ -z "$BW" ]]; then
  echo "QGEMM_MEASURED_BW_GBPS unset — skip benches (not inventing a denominator)."
  echo "local peak ok."
  exit 0
fi
echo "denominator $BW GB/s (QGEMM_MEASURED_BW_GBPS; smoke only, not a published run)"

run_if_present() {
  local bin="$1"
  shift
  if [[ -x "$bin" ]]; then
    echo
    echo "--- $(basename "$bin") $* ---"
    "$bin" "$@"
  else
    echo
    echo "skip $(basename "$bin") (not built)"
  fi
}

# Tiny M / small packed GEMM so this stays a smoke, not a sweep.
run_if_present ./build/bench_qgemv_ref "$DEV" "$BW" 256 256 4
run_if_present ./build/bench_cublas "$DEV" "$BW" kv_gqa 1
run_if_present ./build/bench_naive_dequant "$DEV" "$BW" kv_gqa 1
run_if_present ./build/bench_stream_ideal "$DEV" "$BW" kv_gqa

echo
echo "local peak ok."
