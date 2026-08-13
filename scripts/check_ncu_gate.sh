#!/usr/bin/env bash
# G2/G3: can we read NVIDIA GPU performance counters?
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/results/env/ncu_gate.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== ncu gate $(date -Is) ==="
  command -v ncu && ncu --version || echo "ncu: NOT FOUND"
  echo
  if [[ -x "$ROOT/build/bandwidth" ]]; then
    BIN="$ROOT/build/bandwidth"
  else
    echo "build/bandwidth missing — run: make bandwidth (or make probe)"
    echo "STATUS=SKIP"
    exit 0
  fi
  # Minimal counter set; fails with ERR_NVGPUCTRPERM if blocked
  if ncu --target-processes all \
      --metrics dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
      "$BIN" 64 >"$OUT.ncu_raw" 2>&1; then
    echo "STATUS=PASS"
    tail -20 "$OUT.ncu_raw" || true
  else
    echo "STATUS=FAIL"
    echo "Raw error:"
    cat "$OUT.ncu_raw" || true
    echo
    echo "WSL2 fix (Windows host): NVIDIA Control Panel → Desktop → Enable Developer Settings"
    echo "  → Developer → Manage GPU Performance Counters → Allow access for all users"
    echo "  then: wsl --shutdown  and reopen."
  fi
} | tee "$OUT"
