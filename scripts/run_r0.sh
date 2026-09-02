#!/usr/bin/env bash
# run_r0.sh — full R0 sweep on any arch. The DGX session should be one command.
#
# The sm_86 path is development; sm_90 is authoritative (see docs/PROTOCOL.md).
# Nothing here invents a denominator: if the bandwidth probe cannot be parsed
# and QGEMM_MEASURED_BW_GBPS is unset, the run stops rather than falling back
# to a spec-sheet peak.
#
# Usage:
#   ARCH=sm_90 bash scripts/run_r0.sh              # DGX / H100
#   ARCH=sm_86 bash scripts/run_r0.sh              # dev box
#   ARCH=sm_90 RUN_GPU_TESTS=0 bash scripts/run_r0.sh
#   ARCH=sm_90 QGEMM_MEASURED_BW_GBPS=3100 bash scripts/run_r0.sh
#
# Env:
#   ARCH                      sm_86 | sm_90        (default sm_86)
#   DEV                       CUDA device index    (default 0)
#   RUN_STREAM                1 | 0                (default 1)
#   RUN_GPU_TESTS             1 | 0                (default 1)
#   QGEMM_MEASURED_BW_GBPS    override the parsed denominator
#
# Outputs under results/raw/ and results/env/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCH="${ARCH:-sm_86}"
DEV="${DEV:-0}"
RUN_STREAM="${RUN_STREAM:-1}"
RUN_GPU_TESTS="${RUN_GPU_TESTS:-1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p results/raw results/env

echo "=== R0 runner ==="
echo "ARCH=$ARCH  DEV=$DEV  commit=$SHA  RUN_STREAM=$RUN_STREAM  RUN_GPU_TESTS=$RUN_GPU_TESTS"
if [[ "$ARCH" != "sm_90" ]]; then
  echo
  echo "NOTE: $ARCH is a development arch. PROTOCOL.md: authoritative numbers are sm_90 only."
fi
echo

# --- environment: driver, clocks, throttle, co-tenants, MIG/MPS ---
ENV_OUT="results/env/env_${ARCH}_${TS}.txt"
bash scripts/env_capture.sh "$ENV_OUT"

# --- build ---
make ARCH="$ARCH" all
if [[ "$RUN_STREAM" == "1" ]]; then
  make ARCH="$ARCH" bench_stream
fi

# --- correctness before timing: never trust numbers from an unvalidated device ---
if [[ "$RUN_GPU_TESTS" == "1" ]]; then
  echo
  echo "=== correctness on $ARCH (host pack == device decode) ==="
  make ARCH="$ARCH" test-cpu
  make ARCH="$ARCH" test-gpu
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

# --- denominator: env override wins, else parse the probe log ---
if [[ -n "${QGEMM_MEASURED_BW_GBPS:-}" ]]; then
  BW="$QGEMM_MEASURED_BW_GBPS"
  echo
  echo "using QGEMM_MEASURED_BW_GBPS=$BW"
else
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

# --- cuBLAS FP16 baseline: validates the harness against a known-good reference ---
CUBLAS_OUT="results/raw/cublas_fp16_${ARCH}_${TS}.csv"
echo
echo "=== bench_cublas  dev=$DEV  bw=$BW  -> $CUBLAS_OUT ==="
./build/bench_cublas "$DEV" "$BW" | tee "$CUBLAS_OUT"

# --- stream ideal ---
if [[ "$RUN_STREAM" == "1" ]]; then
  STREAM_OUT="results/raw/stream_ideal_${ARCH}_${TS}.csv"
  echo
  echo "=== bench_stream_ideal  dev=$DEV  bw=$BW  -> $STREAM_OUT ==="
  ./build/bench_stream_ideal "$DEV" "$BW" | tee "$STREAM_OUT"
fi

echo
echo "R0 $ARCH complete."
echo "  env:    $ENV_OUT"
echo "  probe:  $PROBE_LOG"
echo "  cublas: $CUBLAS_OUT"
[[ "$RUN_STREAM" == "1" ]] && echo "  stream: $STREAM_OUT"
echo "  denominator used: $BW GB/s"
echo "  commit: $SHA"

# --- paste-ready block for PREREGISTRATION.md ---
if [[ "$ARCH" == "sm_90" ]]; then
  cat <<PRE

=== paste into PREREGISTRATION.md §1, sm_90 table ===
Fill from the device_props / bandwidth output above. G1 turns GREEN once this
table has real numbers and this run is committed.

| Quantity | Value | Source |
|---|---:|---|
| SMs |  | \`cudaGetDeviceProperties\` |
| L2 |  | same |
| Achieved read-only BW | ${BW} GB/s | \`build/bandwidth 256\`, commit ${SHA} |
| Ridge point |  | observed TC peak / ${BW} |
| Launch overhead |  | \`measure_launch_overhead\` |
| Nsight counter access |  | G3 |
PRE
fi
