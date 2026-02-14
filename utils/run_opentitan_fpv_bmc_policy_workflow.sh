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
  --baseline-prefix NAME  Baseline filename prefix
                          (default: opentitan-fpv-bmc)
  --presets-file FILE     Task-profile presets TSV path
                          (default: utils/opentitan_fpv_policy/task_profile_status_presets.tsv)
  --opentitan-fpv-bmc-verilog-cache-mode MODE
                          OpenTitan FPV BMC verilog frontend cache mode
                          (`off|read|readwrite|auto`)
  --opentitan-fpv-bmc-verilog-cache-dir DIR
                          OpenTitan FPV BMC verilog frontend cache base dir
  --check-bmc-launch-reason-key-allowlist-file FILE
                          check mode: forward as
                          --bmc-launch-reason-key-allowlist-file
  --check-lec-launch-reason-key-allowlist-file FILE
                          check mode: forward as
                          --lec-launch-reason-key-allowlist-file
  --check-max-bmc-launch-reason-event-rows N
                          check mode: forward as
                          --max-bmc-launch-reason-event-rows
  --check-max-lec-launch-reason-event-rows N
                          check mode: forward as
                          --max-lec-launch-reason-event-rows
  --check-fail-on-any-bmc-launch-reason-events
                          check mode: forward as
                          --fail-on-any-bmc-launch-reason-events
  --check-fail-on-any-lec-launch-reason-events
                          check mode: forward as
                          --fail-on-any-lec-launch-reason-events
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
BASELINE_PREFIX="opentitan-fpv-bmc"
PRESETS_FILE="${SCRIPT_DIR}/opentitan_fpv_policy/task_profile_status_presets.tsv"
USE_STRICT_GATE_IN_CHECK=1
OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE=""
OPENTITAN_FPV_BMC_VERILOG_CACHE_DIR=""
CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE=""
CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE=""
CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS=""
CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS=""
CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS=0
CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS=0
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-formal-all)
      RUN_FORMAL_ALL="$2"; shift 2 ;;
    --baseline-dir)
      BASELINE_DIR="$2"; shift 2 ;;
    --baseline-prefix)
      BASELINE_PREFIX="$2"; shift 2 ;;
    --presets-file)
      PRESETS_FILE="$2"; shift 2 ;;
    --opentitan-fpv-bmc-verilog-cache-mode)
      OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE="$2"; shift 2 ;;
    --opentitan-fpv-bmc-verilog-cache-dir)
      OPENTITAN_FPV_BMC_VERILOG_CACHE_DIR="$2"; shift 2 ;;
    --check-bmc-launch-reason-key-allowlist-file)
      CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE="$2"; shift 2 ;;
    --check-lec-launch-reason-key-allowlist-file)
      CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE="$2"; shift 2 ;;
    --check-max-bmc-launch-reason-event-rows)
      CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS="$2"; shift 2 ;;
    --check-max-lec-launch-reason-event-rows)
      CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS="$2"; shift 2 ;;
    --check-fail-on-any-bmc-launch-reason-events)
      CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS=1; shift ;;
    --check-fail-on-any-lec-launch-reason-events)
      CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS=1; shift ;;
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
if [[ -z "$BASELINE_PREFIX" || ! "$BASELINE_PREFIX" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid --baseline-prefix: $BASELINE_PREFIX (expected [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ -n "$OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE" ]]; then
  case "$OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE" in
    off|read|readwrite|auto) ;;
    *)
      echo "invalid --opentitan-fpv-bmc-verilog-cache-mode: expected off|read|readwrite|auto" >&2
      exit 1
      ;;
  esac
fi
if [[ -n "$OPENTITAN_FPV_BMC_VERILOG_CACHE_DIR" && -z "$OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE" ]]; then
  echo "--opentitan-fpv-bmc-verilog-cache-dir requires --opentitan-fpv-bmc-verilog-cache-mode" >&2
  exit 1
fi
if [[ -n "$CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" && ! -r "$CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
  echo "BMC launch reason allowlist file not readable: $CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" >&2
  exit 1
fi
if [[ -n "$CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" && ! -r "$CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
  echo "LEC launch reason allowlist file not readable: $CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" >&2
  exit 1
fi
if [[ -n "$CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS" && ! "$CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS" =~ ^[0-9]+$ ]]; then
  echo "invalid --check-max-bmc-launch-reason-event-rows: $CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS" >&2
  exit 1
fi
if [[ -n "$CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS" && ! "$CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS" =~ ^[0-9]+$ ]]; then
  echo "invalid --check-max-lec-launch-reason-event-rows: $CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS" >&2
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
  --opentitan-fpv-bmc-verilog-cache-mode
  --opentitan-fpv-bmc-verilog-cache-dir
  --bmc-launch-reason-key-allowlist-file
  --lec-launch-reason-key-allowlist-file
  --max-bmc-launch-reason-event-rows
  --max-lec-launch-reason-event-rows
  --fail-on-any-bmc-launch-reason-events
  --fail-on-any-lec-launch-reason-events
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
SUMMARY_BASELINE="${BASELINE_DIR}/${BASELINE_PREFIX}-fpv-summary-baseline.tsv"
ASSERTION_BASELINE="${BASELINE_DIR}/${BASELINE_PREFIX}-assertion-results-baseline.tsv"
GROUPED_POLICY_BASELINE="${BASELINE_DIR}/${BASELINE_PREFIX}-assertion-status-policy-grouped-violations-baseline.tsv"

workflow_args=(
  --with-opentitan-fpv-bmc
  --opentitan-fpv-bmc-assertion-status-policy-task-profile-presets-file "$PRESETS_FILE"
  --opentitan-fpv-bmc-summary-baseline-file "$SUMMARY_BASELINE"
  --opentitan-fpv-bmc-assertion-results-baseline-file "$ASSERTION_BASELINE"
  --opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline-file "$GROUPED_POLICY_BASELINE"
)
if [[ -n "$OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE" ]]; then
  workflow_args+=(
    --opentitan-fpv-bmc-verilog-cache-mode "$OPENTITAN_FPV_BMC_VERILOG_CACHE_MODE"
  )
fi
if [[ -n "$OPENTITAN_FPV_BMC_VERILOG_CACHE_DIR" ]]; then
  workflow_args+=(
    --opentitan-fpv-bmc-verilog-cache-dir "$OPENTITAN_FPV_BMC_VERILOG_CACHE_DIR"
  )
fi

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
  if [[ "$CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS" == "1" ]]; then
    workflow_args+=(--fail-on-any-bmc-launch-reason-events)
  fi
  if [[ "$CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS" == "1" ]]; then
    workflow_args+=(--fail-on-any-lec-launch-reason-events)
  fi
  if [[ -n "$CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS" ]]; then
    workflow_args+=(--max-bmc-launch-reason-event-rows "$CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS")
  fi
  if [[ -n "$CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS" ]]; then
    workflow_args+=(--max-lec-launch-reason-event-rows "$CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS")
  fi
  if [[ -n "$CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
    workflow_args+=(--bmc-launch-reason-key-allowlist-file "$CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE")
  fi
  if [[ -n "$CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
    workflow_args+=(--lec-launch-reason-key-allowlist-file "$CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE")
  fi
else
  echo "unsupported mode: $MODE" >&2
  exit 1
fi

exec "$RUN_FORMAL_ALL" "$@" "${workflow_args[@]}"
