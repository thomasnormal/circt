#!/usr/bin/env bash
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Orchestrate OpenTitan FPV BMC policy workflows for checked-in profile packs.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_opentitan_fpv_bmc_policy_profiles.sh [options] <update|check> [run_formal_all args...]

Modes:
  update                  Update baselines for selected profile packs
  check                   Enforce drift gates for selected profile packs

Options:
  --run-workflow PATH     Workflow wrapper path
                          (default: utils/run_opentitan_fpv_bmc_policy_workflow.sh)
  --profiles-file FILE    Profile-pack TSV path
                          (default: utils/opentitan_fpv_policy/profile_packs.tsv)
  --opentitan-root DIR    OpenTitan checkout root (default: ~/opentitan)
  --out-dir DIR           Per-profile output root
                          (default: ./opentitan-fpv-bmc-policy-profiles-<mode>)
  --profile NAME          Profile name to run (repeatable; default: all)
  --workflow-baseline-dir DIR
                          Forwarded to workflow as --baseline-dir
  --workflow-presets-file FILE
                          Forwarded to workflow as --presets-file
  --workflow-verilog-cache-mode MODE
                          Forwarded to workflow as
                          --opentitan-fpv-bmc-verilog-cache-mode
                          (`off|read|readwrite|auto`)
  --workflow-verilog-cache-dir DIR
                          Forwarded to workflow as
                          --opentitan-fpv-bmc-verilog-cache-dir
  --workflow-enable-objective-parity
                          Forwarded to workflow as
                          --enable-objective-parity
  --workflow-objective-parity-reason-policy POLICY
                          Forwarded to workflow as
                          --objective-parity-reason-policy
                          (`ignore|projected|all`)
  --workflow-check-bmc-launch-reason-key-allowlist-file FILE
                          Forwarded to workflow as
                          --check-bmc-launch-reason-key-allowlist-file
  --workflow-check-lec-launch-reason-key-allowlist-file FILE
                          Forwarded to workflow as
                          --check-lec-launch-reason-key-allowlist-file
  --workflow-check-max-bmc-launch-reason-event-rows N
                          Forwarded to workflow as
                          --check-max-bmc-launch-reason-event-rows
  --workflow-check-max-lec-launch-reason-event-rows N
                          Forwarded to workflow as
                          --check-max-lec-launch-reason-event-rows
  --workflow-check-bmc-launch-reason-event-budget-file FILE
                          Forwarded to workflow as
                          --check-bmc-launch-reason-event-budget-file
  --workflow-check-lec-launch-reason-event-budget-file FILE
                          Forwarded to workflow as
                          --check-lec-launch-reason-event-budget-file
  --workflow-check-fail-on-any-bmc-launch-reason-events
                          Forwarded to workflow as
                          --check-fail-on-any-bmc-launch-reason-events
  --workflow-check-fail-on-any-lec-launch-reason-events
                          Forwarded to workflow as
                          --check-fail-on-any-lec-launch-reason-events
  --no-strict-gate        check mode: forward --no-strict-gate to workflow
  -h, --help              Show this help

Profile TSV schema:
  Required columns:
    profile_name,fpv_cfg,baseline_prefix
  Optional columns:
    select_cfgs,target_filter,allow_unfiltered,max_targets,description
    verilog_cache_mode,verilog_cache_dir
    objective_parity,objective_parity_reason_policy
    check_bmc_launch_reason_key_allowlist_file
    check_lec_launch_reason_key_allowlist_file
    check_max_bmc_launch_reason_event_rows
    check_max_lec_launch_reason_event_rows
    check_bmc_launch_reason_event_budget_file
    check_lec_launch_reason_event_budget_file
    check_fail_on_any_bmc_launch_reason_events
    check_fail_on_any_lec_launch_reason_events

Notes:
  - Each profile executes one workflow invocation with:
    --with-opentitan-fpv-bmc and --include-lane-regex '^opentitan/FPV_BMC$'
  - Additional args after mode are forwarded to each workflow invocation.
USAGE
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s" "$value"
}

split_tsv_fields() {
  local text="$1"
  TSV_FIELDS=()
  while [[ "$text" == *$'\t'* ]]; do
    TSV_FIELDS+=("${text%%$'\t'*}")
    text="${text#*$'\t'}"
  done
  TSV_FIELDS+=("$text")
}

parse_bool_like() {
  local raw
  raw="$(trim "$1")"
  case "${raw,,}" in
    ""|0|false|no) printf "0" ;;
    1|true|yes) printf "1" ;;
    *)
      echo "invalid boolean value: $raw (expected 0/1/true/false/yes/no)" >&2
      exit 1
      ;;
  esac
}

parse_nonnegative_int_like() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "$raw" ]]; then
    printf "0"
    return
  fi
  if ! [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "invalid non-negative integer: $raw" >&2
    exit 1
  fi
  printf "%s" "$raw"
}

parse_optional_nonnegative_int_like() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "$raw" ]]; then
    printf ""
    return
  fi
  if ! [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "invalid non-negative integer: $raw" >&2
    exit 1
  fi
  printf "%s" "$raw"
}

parse_optional_bool_like() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "$raw" ]]; then
    printf ""
    return
  fi
  parse_bool_like "$raw"
}

parse_cache_mode_like() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "$raw" ]]; then
    printf ""
    return
  fi
  case "${raw,,}" in
    off|read|readwrite|auto)
      printf "%s" "${raw,,}" ;;
    *)
      echo "invalid cache mode value: $raw (expected off|read|readwrite|auto)" >&2
      exit 1 ;;
  esac
}

parse_reason_policy_like() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "$raw" ]]; then
    printf ""
    return
  fi
  case "${raw,,}" in
    ignore|projected|all)
      printf "%s" "${raw,,}" ;;
    *)
      echo "invalid objective parity reason policy: $raw (expected ignore|projected|all)" >&2
      exit 1 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WORKFLOW="${SCRIPT_DIR}/run_opentitan_fpv_bmc_policy_workflow.sh"
PROFILES_FILE="${SCRIPT_DIR}/opentitan_fpv_policy/profile_packs.tsv"
OPENTITAN_ROOT="${HOME}/opentitan"
OUT_DIR=""
MODE=""
NO_STRICT_GATE=0
WORKFLOW_BASELINE_DIR=""
WORKFLOW_PRESETS_FILE=""
WORKFLOW_VERILOG_CACHE_MODE=""
WORKFLOW_VERILOG_CACHE_DIR=""
WORKFLOW_ENABLE_OBJECTIVE_PARITY=0
WORKFLOW_OBJECTIVE_PARITY_REASON_POLICY=""
WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE=""
WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE=""
WORKFLOW_CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS=""
WORKFLOW_CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS=""
WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE=""
WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE=""
WORKFLOW_CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS=0
WORKFLOW_CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS=0
declare -a PROFILE_FILTERS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-workflow)
      RUN_WORKFLOW="$2"; shift 2 ;;
    --profiles-file)
      PROFILES_FILE="$2"; shift 2 ;;
    --opentitan-root)
      OPENTITAN_ROOT="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --profile)
      PROFILE_FILTERS+=("$2"); shift 2 ;;
    --workflow-baseline-dir)
      WORKFLOW_BASELINE_DIR="$2"; shift 2 ;;
    --workflow-presets-file)
      WORKFLOW_PRESETS_FILE="$2"; shift 2 ;;
    --workflow-verilog-cache-mode)
      WORKFLOW_VERILOG_CACHE_MODE="$(parse_cache_mode_like "$2")"; shift 2 ;;
    --workflow-verilog-cache-dir)
      WORKFLOW_VERILOG_CACHE_DIR="$2"; shift 2 ;;
    --workflow-enable-objective-parity)
      WORKFLOW_ENABLE_OBJECTIVE_PARITY=1; shift ;;
    --workflow-objective-parity-reason-policy)
      WORKFLOW_OBJECTIVE_PARITY_REASON_POLICY="$(parse_reason_policy_like "$2")"; shift 2 ;;
    --workflow-check-bmc-launch-reason-key-allowlist-file)
      WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE="$2"; shift 2 ;;
    --workflow-check-lec-launch-reason-key-allowlist-file)
      WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE="$2"; shift 2 ;;
    --workflow-check-max-bmc-launch-reason-event-rows)
      WORKFLOW_CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS="$(parse_optional_nonnegative_int_like "$2")"; shift 2 ;;
    --workflow-check-max-lec-launch-reason-event-rows)
      WORKFLOW_CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS="$(parse_optional_nonnegative_int_like "$2")"; shift 2 ;;
    --workflow-check-bmc-launch-reason-event-budget-file)
      WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE="$2"; shift 2 ;;
    --workflow-check-lec-launch-reason-event-budget-file)
      WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE="$2"; shift 2 ;;
    --workflow-check-fail-on-any-bmc-launch-reason-events)
      WORKFLOW_CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS=1; shift ;;
    --workflow-check-fail-on-any-lec-launch-reason-events)
      WORKFLOW_CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS=1; shift ;;
    --no-strict-gate)
      NO_STRICT_GATE=1; shift ;;
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
if [[ ! -x "$RUN_WORKFLOW" ]]; then
  echo "workflow script not executable: $RUN_WORKFLOW" >&2
  exit 1
fi
if [[ ! -r "$PROFILES_FILE" ]]; then
  echo "profiles file not readable: $PROFILES_FILE" >&2
  exit 1
fi
if [[ ! -d "$OPENTITAN_ROOT" ]]; then
  echo "OpenTitan root not found: $OPENTITAN_ROOT" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_VERILOG_CACHE_DIR" && -z "$WORKFLOW_VERILOG_CACHE_MODE" ]]; then
  echo "--workflow-verilog-cache-dir requires --workflow-verilog-cache-mode" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_OBJECTIVE_PARITY_REASON_POLICY" && "$WORKFLOW_ENABLE_OBJECTIVE_PARITY" != "1" ]]; then
  echo "--workflow-objective-parity-reason-policy requires --workflow-enable-objective-parity" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" && ! -r "$WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
  echo "BMC launch reason allowlist file not readable: $WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" && ! -r "$WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" ]]; then
  echo "LEC launch reason allowlist file not readable: $WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE" && ! -r "$WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE" ]]; then
  echo "BMC launch reason-event budget file not readable: $WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE" >&2
  exit 1
fi
if [[ -n "$WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE" && ! -r "$WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE" ]]; then
  echo "LEC launch reason-event budget file not readable: $WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE" >&2
  exit 1
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${PWD}/opentitan-fpv-bmc-policy-profiles-${MODE}"
fi
mkdir -p "$OUT_DIR"
PROFILES_DIR="$(cd "$(dirname "$PROFILES_FILE")" && pwd)"

declare -A filter_map=()
for profile in "${PROFILE_FILTERS[@]}"; do
  profile="$(trim "$profile")"
  [[ -z "$profile" ]] && continue
  filter_map["$profile"]=1
done

declare -A col_index=()
declare -A selected_profile_seen=()
header_seen=0
line_no=0
ran_any=0

while IFS= read -r line || [[ -n "$line" ]]; do
  line_no=$((line_no + 1))
  line_trimmed="$(trim "$line")"
  if [[ -z "$line_trimmed" || "${line_trimmed:0:1}" == "#" ]]; then
    continue
  fi
  split_tsv_fields "$line"
  cols=("${TSV_FIELDS[@]}")
  if [[ "$header_seen" == "0" ]]; then
    for i in "${!cols[@]}"; do
      name="$(trim "${cols[$i]}")"
      [[ -z "$name" ]] && continue
      col_index["$name"]="$i"
    done
    for required in profile_name fpv_cfg baseline_prefix; do
      if [[ -z "${col_index[$required]+x}" ]]; then
        echo "profiles file missing required column '$required': $PROFILES_FILE" >&2
        exit 1
      fi
    done
    header_seen=1
    continue
  fi

  get_col() {
    local name="$1"
    if [[ -z "${col_index[$name]+x}" ]]; then
      printf ""
      return
    fi
    local idx="${col_index[$name]}"
    if (( idx >= ${#cols[@]} )); then
      printf ""
      return
    fi
    trim "${cols[$idx]}"
  }

  profile_name="$(get_col profile_name)"
  [[ -z "$profile_name" ]] && continue
  if [[ -n "${filter_map[$profile_name]+x}" ]]; then
    selected_profile_seen["$profile_name"]=1
  elif (( ${#filter_map[@]} > 0 )); then
    continue
  fi
  if [[ -n "${selected_profile_seen["__dup__:${profile_name}"]+x}" ]]; then
    echo "duplicate profile_name '$profile_name' in $PROFILES_FILE row $line_no" >&2
    exit 1
  fi
  selected_profile_seen["__dup__:${profile_name}"]=1

  fpv_cfg="$(get_col fpv_cfg)"
  baseline_prefix="$(get_col baseline_prefix)"
  select_cfgs="$(get_col select_cfgs)"
  target_filter="$(get_col target_filter)"
  allow_unfiltered="$(parse_bool_like "$(get_col allow_unfiltered)")"
  max_targets="$(parse_nonnegative_int_like "$(get_col max_targets)")"
  profile_verilog_cache_mode="$(parse_cache_mode_like "$(get_col verilog_cache_mode)")"
  profile_verilog_cache_dir="$(trim "$(get_col verilog_cache_dir)")"
  profile_objective_parity="$(parse_optional_bool_like "$(get_col objective_parity)")"
  profile_objective_parity_reason_policy="$(parse_reason_policy_like "$(get_col objective_parity_reason_policy)")"
  profile_check_bmc_launch_reason_key_allowlist_file="$(trim "$(get_col check_bmc_launch_reason_key_allowlist_file)")"
  profile_check_lec_launch_reason_key_allowlist_file="$(trim "$(get_col check_lec_launch_reason_key_allowlist_file)")"
  profile_check_max_bmc_launch_reason_event_rows="$(parse_optional_nonnegative_int_like "$(get_col check_max_bmc_launch_reason_event_rows)")"
  profile_check_max_lec_launch_reason_event_rows="$(parse_optional_nonnegative_int_like "$(get_col check_max_lec_launch_reason_event_rows)")"
  profile_check_bmc_launch_reason_event_budget_file="$(trim "$(get_col check_bmc_launch_reason_event_budget_file)")"
  profile_check_lec_launch_reason_event_budget_file="$(trim "$(get_col check_lec_launch_reason_event_budget_file)")"
  profile_check_fail_on_any_bmc_launch_reason_events="$(parse_optional_bool_like "$(get_col check_fail_on_any_bmc_launch_reason_events)")"
  profile_check_fail_on_any_lec_launch_reason_events="$(parse_optional_bool_like "$(get_col check_fail_on_any_lec_launch_reason_events)")"

  effective_verilog_cache_mode="$WORKFLOW_VERILOG_CACHE_MODE"
  if [[ -n "$profile_verilog_cache_mode" ]]; then
    effective_verilog_cache_mode="$profile_verilog_cache_mode"
  fi
  effective_verilog_cache_dir="$WORKFLOW_VERILOG_CACHE_DIR"
  if [[ -n "$profile_verilog_cache_dir" ]]; then
    effective_verilog_cache_dir="$profile_verilog_cache_dir"
  fi
  if [[ -n "$effective_verilog_cache_dir" && -z "$effective_verilog_cache_mode" ]]; then
    echo "verilog_cache_dir requires verilog_cache_mode for profile '$profile_name' in $PROFILES_FILE row $line_no" >&2
    exit 1
  fi
  effective_objective_parity="$WORKFLOW_ENABLE_OBJECTIVE_PARITY"
  if [[ -n "$profile_objective_parity" ]]; then
    effective_objective_parity="$profile_objective_parity"
  fi
  effective_objective_parity_reason_policy="$WORKFLOW_OBJECTIVE_PARITY_REASON_POLICY"
  if [[ -n "$profile_objective_parity_reason_policy" ]]; then
    effective_objective_parity_reason_policy="$profile_objective_parity_reason_policy"
  fi
  if [[ "$effective_objective_parity" != "1" && -n "$effective_objective_parity_reason_policy" ]]; then
    echo "objective_parity_reason_policy requires objective_parity=1 for profile '$profile_name' in $PROFILES_FILE row $line_no" >&2
    exit 1
  fi
  lane_regex='^opentitan/FPV_BMC$'
  if [[ "$effective_objective_parity" == "1" ]]; then
    lane_regex='^opentitan/(FPV_BMC|FPV_OBJECTIVE_PARITY)$'
  fi
  effective_check_bmc_launch_reason_key_allowlist_file="$WORKFLOW_CHECK_BMC_LAUNCH_REASON_KEY_ALLOWLIST_FILE"
  if [[ -n "$profile_check_bmc_launch_reason_key_allowlist_file" ]]; then
    effective_check_bmc_launch_reason_key_allowlist_file="$profile_check_bmc_launch_reason_key_allowlist_file"
  fi
  effective_check_lec_launch_reason_key_allowlist_file="$WORKFLOW_CHECK_LEC_LAUNCH_REASON_KEY_ALLOWLIST_FILE"
  if [[ -n "$profile_check_lec_launch_reason_key_allowlist_file" ]]; then
    effective_check_lec_launch_reason_key_allowlist_file="$profile_check_lec_launch_reason_key_allowlist_file"
  fi
  if [[ -n "$effective_check_bmc_launch_reason_key_allowlist_file" && "${effective_check_bmc_launch_reason_key_allowlist_file:0:1}" != "/" ]]; then
    effective_check_bmc_launch_reason_key_allowlist_file="${PROFILES_DIR}/${effective_check_bmc_launch_reason_key_allowlist_file}"
  fi
  if [[ -n "$effective_check_lec_launch_reason_key_allowlist_file" && "${effective_check_lec_launch_reason_key_allowlist_file:0:1}" != "/" ]]; then
    effective_check_lec_launch_reason_key_allowlist_file="${PROFILES_DIR}/${effective_check_lec_launch_reason_key_allowlist_file}"
  fi
  if [[ -n "$effective_check_bmc_launch_reason_key_allowlist_file" && ! -r "$effective_check_bmc_launch_reason_key_allowlist_file" ]]; then
    echo "BMC launch reason allowlist file not readable for profile '$profile_name': $effective_check_bmc_launch_reason_key_allowlist_file" >&2
    exit 1
  fi
  if [[ -n "$effective_check_lec_launch_reason_key_allowlist_file" && ! -r "$effective_check_lec_launch_reason_key_allowlist_file" ]]; then
    echo "LEC launch reason allowlist file not readable for profile '$profile_name': $effective_check_lec_launch_reason_key_allowlist_file" >&2
    exit 1
  fi
  effective_check_max_bmc_launch_reason_event_rows="$WORKFLOW_CHECK_MAX_BMC_LAUNCH_REASON_EVENT_ROWS"
  if [[ -n "$profile_check_max_bmc_launch_reason_event_rows" ]]; then
    effective_check_max_bmc_launch_reason_event_rows="$profile_check_max_bmc_launch_reason_event_rows"
  fi
  effective_check_max_lec_launch_reason_event_rows="$WORKFLOW_CHECK_MAX_LEC_LAUNCH_REASON_EVENT_ROWS"
  if [[ -n "$profile_check_max_lec_launch_reason_event_rows" ]]; then
    effective_check_max_lec_launch_reason_event_rows="$profile_check_max_lec_launch_reason_event_rows"
  fi
  effective_check_bmc_launch_reason_event_budget_file="$WORKFLOW_CHECK_BMC_LAUNCH_REASON_EVENT_BUDGET_FILE"
  if [[ -n "$profile_check_bmc_launch_reason_event_budget_file" ]]; then
    effective_check_bmc_launch_reason_event_budget_file="$profile_check_bmc_launch_reason_event_budget_file"
  fi
  effective_check_lec_launch_reason_event_budget_file="$WORKFLOW_CHECK_LEC_LAUNCH_REASON_EVENT_BUDGET_FILE"
  if [[ -n "$profile_check_lec_launch_reason_event_budget_file" ]]; then
    effective_check_lec_launch_reason_event_budget_file="$profile_check_lec_launch_reason_event_budget_file"
  fi
  if [[ -n "$effective_check_bmc_launch_reason_event_budget_file" && "${effective_check_bmc_launch_reason_event_budget_file:0:1}" != "/" ]]; then
    effective_check_bmc_launch_reason_event_budget_file="${PROFILES_DIR}/${effective_check_bmc_launch_reason_event_budget_file}"
  fi
  if [[ -n "$effective_check_lec_launch_reason_event_budget_file" && "${effective_check_lec_launch_reason_event_budget_file:0:1}" != "/" ]]; then
    effective_check_lec_launch_reason_event_budget_file="${PROFILES_DIR}/${effective_check_lec_launch_reason_event_budget_file}"
  fi
  if [[ -n "$effective_check_bmc_launch_reason_event_budget_file" && ! -r "$effective_check_bmc_launch_reason_event_budget_file" ]]; then
    echo "BMC launch reason-event budget file not readable for profile '$profile_name': $effective_check_bmc_launch_reason_event_budget_file" >&2
    exit 1
  fi
  if [[ -n "$effective_check_lec_launch_reason_event_budget_file" && ! -r "$effective_check_lec_launch_reason_event_budget_file" ]]; then
    echo "LEC launch reason-event budget file not readable for profile '$profile_name': $effective_check_lec_launch_reason_event_budget_file" >&2
    exit 1
  fi
  effective_check_fail_on_any_bmc_launch_reason_events="$WORKFLOW_CHECK_FAIL_ON_ANY_BMC_LAUNCH_REASON_EVENTS"
  if [[ -n "$profile_check_fail_on_any_bmc_launch_reason_events" ]]; then
    effective_check_fail_on_any_bmc_launch_reason_events="$profile_check_fail_on_any_bmc_launch_reason_events"
  fi
  effective_check_fail_on_any_lec_launch_reason_events="$WORKFLOW_CHECK_FAIL_ON_ANY_LEC_LAUNCH_REASON_EVENTS"
  if [[ -n "$profile_check_fail_on_any_lec_launch_reason_events" ]]; then
    effective_check_fail_on_any_lec_launch_reason_events="$profile_check_fail_on_any_lec_launch_reason_events"
  fi

  if [[ -z "$fpv_cfg" ]]; then
    echo "empty fpv_cfg for profile '$profile_name' in $PROFILES_FILE row $line_no" >&2
    exit 1
  fi
  if [[ -z "$baseline_prefix" || ! "$baseline_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid baseline_prefix for profile '$profile_name' in $PROFILES_FILE row $line_no: '$baseline_prefix'" >&2
    exit 1
  fi

  fpv_cfg_path="$fpv_cfg"
  if [[ "${fpv_cfg_path:0:1}" != "/" ]]; then
    fpv_cfg_path="${OPENTITAN_ROOT}/${fpv_cfg_path}"
  fi
  if [[ ! -r "$fpv_cfg_path" ]]; then
    echo "OpenTitan FPV cfg not readable for profile '$profile_name': $fpv_cfg_path" >&2
    exit 1
  fi

  cmd=("$RUN_WORKFLOW")
  if [[ -n "$WORKFLOW_BASELINE_DIR" ]]; then
    cmd+=(--baseline-dir "$WORKFLOW_BASELINE_DIR")
  fi
  if [[ -n "$WORKFLOW_PRESETS_FILE" ]]; then
    cmd+=(--presets-file "$WORKFLOW_PRESETS_FILE")
  fi
  if [[ -n "$effective_verilog_cache_mode" ]]; then
    cmd+=(--opentitan-fpv-bmc-verilog-cache-mode "$effective_verilog_cache_mode")
  fi
  if [[ -n "$effective_verilog_cache_dir" ]]; then
    cmd+=(--opentitan-fpv-bmc-verilog-cache-dir "$effective_verilog_cache_dir")
  fi
  if [[ "$effective_objective_parity" == "1" ]]; then
    cmd+=(--enable-objective-parity)
    if [[ -n "$effective_objective_parity_reason_policy" ]]; then
      cmd+=(--objective-parity-reason-policy "$effective_objective_parity_reason_policy")
    fi
  fi
  if [[ -n "$effective_check_bmc_launch_reason_key_allowlist_file" ]]; then
    cmd+=(--check-bmc-launch-reason-key-allowlist-file "$effective_check_bmc_launch_reason_key_allowlist_file")
  fi
  if [[ -n "$effective_check_lec_launch_reason_key_allowlist_file" ]]; then
    cmd+=(--check-lec-launch-reason-key-allowlist-file "$effective_check_lec_launch_reason_key_allowlist_file")
  fi
  if [[ -n "$effective_check_max_bmc_launch_reason_event_rows" ]]; then
    cmd+=(--check-max-bmc-launch-reason-event-rows "$effective_check_max_bmc_launch_reason_event_rows")
  fi
  if [[ -n "$effective_check_max_lec_launch_reason_event_rows" ]]; then
    cmd+=(--check-max-lec-launch-reason-event-rows "$effective_check_max_lec_launch_reason_event_rows")
  fi
  if [[ -n "$effective_check_bmc_launch_reason_event_budget_file" ]]; then
    cmd+=(--check-bmc-launch-reason-event-budget-file "$effective_check_bmc_launch_reason_event_budget_file")
  fi
  if [[ -n "$effective_check_lec_launch_reason_event_budget_file" ]]; then
    cmd+=(--check-lec-launch-reason-event-budget-file "$effective_check_lec_launch_reason_event_budget_file")
  fi
  if [[ "$effective_check_fail_on_any_bmc_launch_reason_events" == "1" ]]; then
    cmd+=(--check-fail-on-any-bmc-launch-reason-events)
  fi
  if [[ "$effective_check_fail_on_any_lec_launch_reason_events" == "1" ]]; then
    cmd+=(--check-fail-on-any-lec-launch-reason-events)
  fi
  if [[ "$NO_STRICT_GATE" == "1" ]]; then
    cmd+=(--no-strict-gate)
  fi
  cmd+=(--baseline-prefix "$baseline_prefix")
  cmd+=("$MODE")
  cmd+=(
    --opentitan "$OPENTITAN_ROOT"
    --out-dir "${OUT_DIR}/${profile_name}"
    --include-lane-regex "$lane_regex"
    --opentitan-fpv-cfg "$fpv_cfg_path"
  )
  if [[ "$allow_unfiltered" == "1" ]]; then
    cmd+=(--opentitan-fpv-allow-unfiltered)
  fi
  if [[ "$max_targets" != "0" ]]; then
    cmd+=(--opentitan-fpv-max-targets "$max_targets")
  fi
  if [[ -n "$target_filter" ]]; then
    cmd+=(--opentitan-fpv-target-filter "$target_filter")
  fi
  if [[ -n "$select_cfgs" ]]; then
    IFS=',' read -r -a select_tokens <<< "$select_cfgs"
    for token in "${select_tokens[@]}"; do
      token="$(trim "$token")"
      [[ -z "$token" ]] && continue
      cmd+=(--select-cfgs "$token")
    done
  fi
  cmd+=("$@")

  echo "[opentitan-fpv-bmc-policy-profiles] profile=${profile_name} mode=${MODE}" >&2
  "${cmd[@]}"
  ran_any=1
done < "$PROFILES_FILE"

if [[ "$header_seen" == "0" ]]; then
  echo "profiles file missing header row: $PROFILES_FILE" >&2
  exit 1
fi
if (( ${#filter_map[@]} > 0 )); then
  for profile in "${!filter_map[@]}"; do
    if [[ -z "${selected_profile_seen[$profile]+x}" ]]; then
      echo "requested profile not found in $PROFILES_FILE: $profile" >&2
      exit 1
    fi
  done
fi
if [[ "$ran_any" != "1" ]]; then
  echo "no OpenTitan FPV BMC policy profiles selected from: $PROFILES_FILE" >&2
  exit 1
fi
