#!/usr/bin/env bash
# run_r0_sm86.sh — thin wrapper kept for the paths referenced in README.md,
# docs/BASELINE_STUDY.md and docs/MASTER_EXECUTION_PLAN.md.
# The real runner is scripts/run_r0.sh and is arch-generic.
set -euo pipefail
ARCH="${ARCH:-sm_86}" exec bash "$(dirname "${BASH_SOURCE[0]}")/run_r0.sh" "$@"
