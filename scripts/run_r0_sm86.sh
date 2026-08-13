#!/usr/bin/env bash
# run_r0_sm86.sh — full local R0 sweep on the sm_86 lab GPU.
#
# 1. Capture environment
# 2. Build + run device/bandwidth probes
# 3. Parse measured RO BW (or honor QGEMM_MEASURED_BW_GBPS)
# 4. Run cuBLAS FP16 baseline
# 5. Optionally run stream-ideal (set RUN_STREAM=0 to skip)
#
# Usage:
#   bash scripts/run_r0_sm86.sh
#   QGEMM_MEASURED_BW_GBPS=124.2 bash scripts/run_r0_sm86.sh
#   RUN_STREAM=0 bash scripts/run_r0_sm86.sh
#
# Outputs under results/raw/ and results/env/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCH="${ARCH:-sm_86}"
DEV="${DEV:-0}"
RUN_STREAM="${RUN_STREAM:-1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p results/raw results/env

echo "=== R0 sm_86 runner ==="
echo "ARCH=$ARCH  DEV=$DEV  RUN_STREAM=$RUN_STREAM"
echo

# --- environment ---
ENV_OUT="results/env/env_${ARCH}_${TS}.txt"
bash scripts/env_capture.sh "$ENV_OUT"

# --- build probes + benches ---
make ARCH="$ARCH" all
if [[ "$RUN_STREAM" == "1" ]]; then
  make ARCH="$ARCH" bench_stream
fi

# --- probes ---
PROBE_LOG="results/raw/probe_${ARCH}_${TS}.txt"
{
  echo "=== device_props ==="
  ./build/device_props
  echo
  echo "=== bandwidth ==="
  ./build/bandwidth 256
} | tee "$PROBE_LOG"

# --- denominator: env override wins, else parse probe log ---
if [[ -n "${QGEMM_MEASURED_BW_GBPS:-}" ]]; then
  BW="$QGEMM_MEASURED_BW_GBPS"
  echo
  echo "using QGEMM_MEASURED_BW_GBPS=$BW"
else
  # Probe prints: >>> USE THIS AS THE PROJECT DENOMINATOR: 124.2 GB/s (read-only)
  BW="$(grep -oE 'DENOMINATOR: [0-9]+(\.[0-9]+)? GB/s' "$PROBE_LOG" \
        | head -1 | awk '{print $2}')"
  if [[ -z "$BW" ]]; then
    echo "error: could not parse read-only BW from probe log" >&2
    echo "       set QGEMM_MEASURED_BW_GBPS and re-run" >&2
    exit 1
  fi
  echo
  echo "parsed measured RO bandwidth: $BW GB/s"
fi

# --- cuBLAS FP16 ---
CUBLAS_OUT="results/raw/cublas_fp16_${ARCH}_${TS}.csv"
echo
echo "=== bench_cublas  dev=$DEV  bw=$BW  -> $CUBLAS_OUT ==="
./build/bench_cublas "$DEV" "$BW" | tee "$CUBLAS_OUT"

# --- stream ideal (optional) ---
if [[ "$RUN_STREAM" == "1" ]]; then
  STREAM_OUT="results/raw/stream_ideal_${ARCH}_${TS}.csv"
  echo
  echo "=== bench_stream_ideal  dev=$DEV  bw=$BW  -> $STREAM_OUT ==="
  ./build/bench_stream_ideal "$DEV" "$BW" | tee "$STREAM_OUT"
fi

echo
echo "R0 sm_86 complete."
echo "  env:    $ENV_OUT"
echo "  probe:  $PROBE_LOG"
echo "  cublas: $CUBLAS_OUT"
[[ "$RUN_STREAM" == "1" ]] && echo "  stream: $STREAM_OUT"
echo "  denominator used: $BW GB/s"
echo
echo "summarize with:"
echo "  python/analyze_csv.py $CUBLAS_OUT"
