#!/usr/bin/env bash
# sanitize_qgemv.sh — compute-sanitizer memcheck on the fused-ref GPU test.
#
# Skips (exit 0) if build/test_qgemv_gpu is not present. Fails if memcheck
# reports an error (error-exitcode).
#
# Usage:
#   bash scripts/sanitize_qgemv.sh
#   CUDA_HOME=/usr/local/cuda-12.6 bash scripts/sanitize_qgemv.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/build/test_qgemv_gpu"
if [[ ! -x "$BIN" ]]; then
  echo "build/test_qgemv_gpu missing — skip (make test-gpu)"
  exit 0
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ -n "${COMPUTE_SANITIZER:-}" && -x "${COMPUTE_SANITIZER}" ]]; then
  SAN="$COMPUTE_SANITIZER"
elif [[ -x "$CUDA_HOME/bin/compute-sanitizer" ]]; then
  SAN="$CUDA_HOME/bin/compute-sanitizer"
elif command -v compute-sanitizer >/dev/null 2>&1; then
  SAN="$(command -v compute-sanitizer)"
else
  echo "compute-sanitizer not found — skip"
  echo "  looked in \$COMPUTE_SANITIZER, $CUDA_HOME/bin, PATH"
  exit 0
fi

echo "=== compute-sanitizer memcheck ==="
echo "sanitizer  $SAN"
echo "target     $BIN"
echo
"$SAN" --tool memcheck --error-exitcode 1 "$BIN"
echo
echo "sanitize_qgemv: memcheck clean."
