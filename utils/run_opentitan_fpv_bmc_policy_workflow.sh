#!/usr/bin/env bash
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Workflow wrapper for OpenTitan FPV BMC policy baseline governance.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_opentitan_fpv_bmc_policy_workflow.sh [options] <update|check> [run_formal_all args...]

Modes:
  update                  Update policy-governed OpenTitan FPV BMC baselines
  check                   Enforce policy-governed OpenTitan FPV BMC drift gates

Options:
  --run-formal-all PATH   run_formal_all.sh path
                          (default: utils/run_formal_all.sh)
  --baseline-dir DIR      Baseline directory for policy artifacts
                          (default: utils/opentitan_fpv_policy/baselines)
  --presets-file FILE     Task-profile presets TSV path
                          (default: utils/opentitan_fpv_policy/task_profile_status_presets.tsv)
  --no-strict-gate        check mode: do not add --strict-gate automatically
  -h, --help              Show this help

Notes:
  - This wrapper injects and owns OpenTitan FPV BMC policy/baseline flags.
  - Pass all target-selection and lane-filter args after the mode token.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_FORMAL_ALL="${SCRIPT_DIR}/run_formal_all.sh"
BASELINE_DIR="${SCRIPT_DIR}/opentitan_fpv_policy/baselines"
PRESETS_FILE="${SCRIPT_DIR}/opentitan_fpv_policy/task_profile_status_presets.tsv"
USE_STRICT_GATE_IN_CHECK=1
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-formal-all)
      RUN_FORMAL_ALL="$2"; shift 2 ;;
    --baseline-dir)
      BASELINE_DIR="$2"; shift 2 ;;
    --presets-file)
      PRESETS_FILE="$2"; shift 2 ;;
    --no-strict-gate)
      USE_STRICT_GATE_IN_CHECK=0; shift ;;
    update|check)
      MODE="$1"; shift; break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option before mode: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "missing mode: expected update or check" >&2
  usage
  exit 1
fi
if [[ ! -x "$RUN_FORMAL_ALL" ]]; then
  echo "run_formal_all.sh not executable: $RUN_FORMAL_ALL" >&2
  exit 1
fi
if [[ ! -r "$PRESETS_FILE" ]]; then
  echo "task-profile presets file not readable: $PRESETS_FILE" >&2
  exit 1
fi

readonly MANAGED_FLAGS=(
  --with-opentitan-fpv-bmc
  --opentitan-fpv-bmc-assertion-status-policy-task-profile-presets-file
  --opentitan-fpv-bmc-summary-baseline-file
  --opentitan-fpv-bmc-assertion-results-baseline-file
  --opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline-file
  --update-opentitan-fpv-bmc-summary-baseline
  --update-opentitan-fpv-bmc-assertion-results-baseline
  --update-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline
  --fail-on-opentitan-fpv-bmc-summary-drift
  --fail-on-opentitan-fpv-bmc-assertion-results-drift
  --fail-on-opentitan-fpv-bmc-assertion-status-policy
  --fail-on-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift
)
for arg in "$@"; do
  for managed in "${MANAGED_FLAGS[@]}"; do
    if [[ "$arg" == "$managed" ]]; then
      echo "workflow-managed option must not be passed explicitly: $arg" >&2
      exit 1
    fi
  done
done

mkdir -p "$BASELINE_DIR"
SUMMARY_BASELINE="${BASELINE_DIR}/opentitan-fpv-bmc-fpv-summary-baseline.tsv"
ASSERTION_BASELINE="${BASELINE_DIR}/opentitan-fpv-bmc-assertion-results-baseline.tsv"
GROUPED_POLICY_BASELINE="${BASELINE_DIR}/opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline.tsv"

workflow_args=(
  --with-opentitan-fpv-bmc
  --opentitan-fpv-bmc-assertion-status-policy-task-profile-presets-file "$PRESETS_FILE"
  --opentitan-fpv-bmc-summary-baseline-file "$SUMMARY_BASELINE"
  --opentitan-fpv-bmc-assertion-results-baseline-file "$ASSERTION_BASELINE"
  --opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline-file "$GROUPED_POLICY_BASELINE"
)

if [[ "$MODE" == "update" ]]; then
  workflow_args+=(
    --update-opentitan-fpv-bmc-summary-baseline
    --update-opentitan-fpv-bmc-assertion-results-baseline
    --update-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline
  )
elif [[ "$MODE" == "check" ]]; then
  workflow_args+=(
    --fail-on-opentitan-fpv-bmc-summary-drift
    --fail-on-opentitan-fpv-bmc-assertion-results-drift
    --fail-on-opentitan-fpv-bmc-assertion-status-policy
    --fail-on-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift
  )
  if [[ "$USE_STRICT_GATE_IN_CHECK" == "1" ]]; then
    workflow_args+=(--strict-gate)
  fi
else
  echo "unsupported mode: $MODE" >&2
  exit 1
fi

exec "$RUN_FORMAL_ALL" "$@" "${workflow_args[@]}"
