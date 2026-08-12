#!/usr/bin/env bash
# env_capture.sh — record the machine state alongside every measurement run.
#
# Success criterion 8.1 requires a third party to reproduce results from the
# instructions. That is impossible without knowing the driver, the toolkit, the
# clocks the GPU actually ran at, and what else was resident on the device.
# On a shared DGX the last of those matters most: a co-tenant job still contends
# for host bandwidth, PCIe, and power headroom.
#
# Usage:  bash scripts/env_capture.sh [outfile]
# Writes to results/env_<host>_<timestamp>.txt unless given a path.

set -uo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
OUT="${1:-results/env_${HOST}_${TS}.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== capture ==="
  echo "utc              $TS"
  echo "host             $HOST"
  echo "user             ${USER:-unknown}"
  echo "kernel           $(uname -sr)"
  if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "virtualization   WSL2  <-- launch overhead is inflated vs native Linux;"
    echo "                       do not compare absolute launch cost across hosts"
  fi
  echo "cpu              $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  echo "sockets/NUMA     $(lscpu 2>/dev/null | grep -m1 'NUMA node(s)' | awk '{print $NF}')"

  echo
  echo "=== toolchain ==="
  echo "nvcc             $(nvcc --version 2>/dev/null | tail -1)"
  echo "g++              $(g++ --dumpversion 2>/dev/null)"
  echo "git rev          $(git rev-parse --short HEAD 2>/dev/null || echo 'not a repo')"
  echo "git dirty        $(test -n "$(git status --porcelain 2>/dev/null)" && echo YES || echo no)"

  echo
  echo "=== gpu inventory ==="
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,vbios_version,memory.total,compute_mode,persistence_mode \
             --format=csv 2>/dev/null || echo "nvidia-smi unavailable"

  echo
  echo "=== clocks, power, thermals at capture time ==="
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,clocks_throttle_reasons.active \
             --format=csv 2>/dev/null || true
  echo
  echo "NOTE: locked clocks require root (nvidia-smi -lgc). If clocks.sm is well"
  echo "      below clocks.max.sm, or a throttle reason is active, the run is"
  echo "      thermally or power limited and absolute latencies are not portable."

  echo
  echo "=== other processes on the device (contention check) ==="
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null || true
  echo "(any row here other than your own benchmark invalidates exclusive-access claims)"

  echo
  echo "=== MIG / MPS state ==="
  nvidia-smi --query-gpu=mig.mode.current --format=csv 2>/dev/null || true
  pgrep -a nvidia-cuda-mps-control >/dev/null 2>&1 && echo "MPS: RUNNING (turn it off for latency work)" || echo "MPS: not running"

  echo
  echo "=== relevant environment ==="
  for v in CUDA_VISIBLE_DEVICES CUDA_HOME CUDA_MODULE_LOADING CUBLAS_WORKSPACE_CONFIG \
           CUDA_DEVICE_MAX_CONNECTIONS OMP_NUM_THREADS; do
    echo "$v=${!v:-<unset>}"
  done
} | tee "$OUT"

echo
echo "captured -> $OUT"
