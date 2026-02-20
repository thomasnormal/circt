#!/usr/bin/env bash
# Unified multi-suite regression orchestrator (CIRCT/Xcelium).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MANIFEST="${SCRIPT_DIR}/../docs/unified_regression_manifest.tsv"
DEFAULT_ADAPTER_CATALOG="${SCRIPT_DIR}/../docs/unified_regression_adapter_catalog.tsv"

MANIFEST="${UNIFIED_REGRESSION_MANIFEST:-$DEFAULT_MANIFEST}"
ADAPTER_CATALOG="${UNIFIED_REGRESSION_ADAPTER_CATALOG:-$DEFAULT_ADAPTER_CATALOG}"
PROFILE="${UNIFIED_REGRESSION_PROFILE:-smoke}"
ENGINE="${UNIFIED_REGRESSION_ENGINE:-circt}"
OUT_DIR="${UNIFIED_REGRESSION_OUT_DIR:-${PWD}/unified-regression-results}"
ENGINE_PARITY_FILE=""
RETRY_SUMMARY_FILE=""
SUITE_REGEX=""
DRY_RUN=0
KEEP_GOING=1
RESUME=0
SHARD_COUNT="${UNIFIED_REGRESSION_SHARD_COUNT:-1}"
SHARD_INDEX="${UNIFIED_REGRESSION_SHARD_INDEX:-0}"
LANE_RETRIES="${UNIFIED_REGRESSION_LANE_RETRIES:-0}"
LANE_RETRY_DELAY_MS="${UNIFIED_REGRESSION_LANE_RETRY_DELAY_MS:-0}"
JOBS="${UNIFIED_REGRESSION_JOBS:-1}"

SUMMARY_FILE=""
PLAN_FILE=""
RUN_WORK_DIR=""
lane_seq=0
selected=0
failed=0
skipped_resume=0
stop_scheduling=0

declare -A RESUME_COMPLETED_LANES=()
declare -a RUNNING_PIDS=()
declare -A PID_TO_BASE=()

usage() {
  cat <<'USAGE'
usage: run_regression_unified.sh [options]

Unified orchestrator for AVIP/sv-tests/OpenTitan/Ibex-style regression suites.

Manifest columns (TSV):
  suite_id<TAB>profiles<TAB>circt_cmd<TAB>xcelium_cmd
Optional extension columns:
  suite_root<TAB>circt_adapter<TAB>xcelium_adapter<TAB>adapter_args

Options:
  --manifest FILE           Suite manifest TSV
  --adapter-catalog FILE    Adapter catalog TSV (default: docs/unified_regression_adapter_catalog.tsv)
  --profile MODE            smoke|nightly|full (default: smoke)
  --engine MODE             circt|xcelium|both (default: circt)
  --out-dir DIR             Output directory (default: ./unified-regression-results)
  --engine-parity-file FILE Output parity report path (default: <out-dir>/engine_parity.tsv)
  --retry-summary-file FILE Output retry summary path (default: <out-dir>/retry-summary.tsv)
  --suite-regex REGEX       Run only suite_ids matching REGEX
  --jobs N                  Max concurrent lane executions (default: 1)
  --lane-retries N          Retry failing executable lanes up to N times (default: 0)
  --lane-retry-delay-ms N   Delay before each retry in milliseconds (default: 0)
  --resume                  Resume from existing summary/plan artifacts in --out-dir
  --shard-count N           Total shard count for suite distribution (default: 1)
  --shard-index N           Zero-based shard index in [0, shard-count) (default: 0)
  --dry-run                 Do not execute commands; emit plan only
  --keep-going              Continue after failures/unconfigured lanes (default)
  --no-keep-going           Stop on first failure
  -h, --help                Show help
USAGE
}

is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_pos_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

matches_profile() {
  local row_profiles="$1"
  local token=""
  local normalized="${row_profiles// /}"
  if [[ -z "$normalized" ]]; then
    return 1
  fi
  IFS=',' read -r -a tokens <<< "$normalized"
  for token in "${tokens[@]}"; do
    case "$token" in
      all|"$PROFILE")
        return 0
        ;;
    esac
  done
  return 1
}

in_selected_shard() {
  local suite_id="$1"
  local hash=""
  local slot="0"

  if [[ "$SHARD_COUNT" -eq 1 ]]; then
    return 0
  fi

  hash="$(printf '%s' "$suite_id" | cksum | awk '{print $1}')"
  slot="$((hash % SHARD_COUNT))"
  [[ "$slot" -eq "$SHARD_INDEX" ]]
}

load_resume_state() {
  local summary_file="$1"
  local suite_id=""
  local engine=""
  local status=""

  while IFS=$'\t' read -r suite_id engine status _rest; do
    if [[ "$suite_id" == "suite_id" || -z "$suite_id" || -z "$engine" ]]; then
      continue
    fi
    if [[ "$status" == "PASS" || "$status" == "UNCONFIGURED" ]]; then
      RESUME_COMPLETED_LANES["${suite_id}::${engine}"]=1
    fi
  done < "$summary_file"
}

is_lane_done_by_resume() {
  local suite_id="$1"
  local engine_kind="$2"
  if [[ "$RESUME" -ne 1 ]]; then
    return 1
  fi
  [[ -n "${RESUME_COMPLETED_LANES[${suite_id}::${engine_kind}]+x}" ]]
}

resolve_lane_command() {
  local engine_kind="$1"
  local explicit_cmd_raw="$2"
  local suite_root_raw="$3"
  local adapter_id_raw="$4"
  local adapter_args_raw="$5"

  local explicit_cmd
  local adapter_id
  local suite_root
  local adapter_args
  local adapter_prefix=""
  local resolved=""

  explicit_cmd="$(trim_whitespace "$explicit_cmd_raw")"
  adapter_id="$(trim_whitespace "$adapter_id_raw")"
  suite_root="$(trim_whitespace "$suite_root_raw")"
  adapter_args="$(trim_whitespace "$adapter_args_raw")"

  if [[ -n "$explicit_cmd" && "$explicit_cmd" != "-" ]]; then
    printf '%s\n' "$explicit_cmd"
    return 0
  fi

  if [[ -z "$adapter_id" || "$adapter_id" == "-" ]]; then
    if [[ -z "$explicit_cmd" ]]; then
      printf '%s\n' "-"
    else
      printf '%s\n' "$explicit_cmd"
    fi
    return 0
  fi

  if [[ ! -f "$ADAPTER_CATALOG" ]]; then
    echo "adapter catalog not found: $ADAPTER_CATALOG" >&2
    return 2
  fi
  if [[ ! -r "$ADAPTER_CATALOG" ]]; then
    echo "adapter catalog not readable: $ADAPTER_CATALOG" >&2
    return 2
  fi

  adapter_prefix="$(awk -F'\t' -v id="$adapter_id" -v eng="$engine_kind" '
    NF >= 3 && $1 == id && $2 == eng { print $3; exit }
  ' "$ADAPTER_CATALOG")"
  adapter_prefix="$(trim_whitespace "$adapter_prefix")"

  if [[ -z "$adapter_prefix" ]]; then
    echo "missing adapter entry in catalog: adapter_id=${adapter_id} engine=${engine_kind} catalog=${ADAPTER_CATALOG}" >&2
    return 2
  fi

  if [[ -z "$suite_root" || "$suite_root" == "-" ]]; then
    echo "adapter lane requires suite_root column: adapter_id=${adapter_id} engine=${engine_kind}" >&2
    return 2
  fi

  resolved="${adapter_prefix} $(printf '%q' "$suite_root")"
  if [[ -n "$adapter_args" && "$adapter_args" != "-" ]]; then
    resolved+=" ${adapter_args}"
  fi

  printf '%s\n' "$resolved"
  return 0
}

run_lane_capture() {
  local suite_id="$1"
  local engine_kind="$2"
  local cmd="$3"
  local plan_row_file="$4"
  local summary_row_file="$5"
  local retry_row_file="$6"

  local log_file="${OUT_DIR}/logs/${suite_id}.${engine_kind}.log"
  local start_sec="$(date +%s)"
  local end_sec=""
  local elapsed_sec=""
  local status=""
  local exit_code="0"
  local attempts="1"
  local retries_used="0"
  local max_attempts="$((LANE_RETRIES + 1))"

  printf "%s\t%s\t%s\n" "$suite_id" "$engine_kind" "$cmd" > "$plan_row_file"

  if [[ -z "$cmd" || "$cmd" == "-" ]]; then
    status="UNCONFIGURED"
    exit_code="125"
    elapsed_sec="0"
  elif [[ "$DRY_RUN" -eq 1 ]]; then
    status="DRYRUN"
    exit_code="0"
    elapsed_sec="0"
  else
    local lane_rc="0"
    local delay_sec=""
    while true; do
      set +e
      bash -lc "$cmd" >"$log_file" 2>&1
      lane_rc="$?"
      set -e

      if [[ "$lane_rc" -eq 0 ]]; then
        status="PASS"
        exit_code="0"
        break
      fi

      status="FAIL"
      exit_code="$lane_rc"
      if [[ "$attempts" -ge "$max_attempts" ]]; then
        break
      fi

      attempts="$((attempts + 1))"
      retries_used="$((attempts - 1))"
      if [[ "$LANE_RETRY_DELAY_MS" -gt 0 ]]; then
        printf -v delay_sec '%d.%03d' "$((LANE_RETRY_DELAY_MS / 1000))" "$((LANE_RETRY_DELAY_MS % 1000))"
        sleep "$delay_sec"
      fi
    done

    retries_used="$((attempts - 1))"
    end_sec="$(date +%s)"
    elapsed_sec="$((end_sec - start_sec))"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$suite_id" "$engine_kind" "$status" "$exit_code" "$elapsed_sec" "$log_file" > "$summary_row_file"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$suite_id" "$engine_kind" "$attempts" "$retries_used" "$status" "$exit_code" > "$retry_row_file"

  if [[ "$status" == "FAIL" || "$status" == "UNCONFIGURED" ]]; then
    return 1
  fi
  return 0
}

append_lane_artifacts() {
  local base="$1"
  local plan_row_file="${base}.plan"
  local summary_row_file="${base}.summary"
  local retry_row_file="${base}.retry"

  if [[ -f "$plan_row_file" ]]; then
    cat "$plan_row_file" >> "$PLAN_FILE"
  fi
  if [[ -f "$summary_row_file" ]]; then
    cat "$summary_row_file" >> "$SUMMARY_FILE"
  fi
  if [[ -f "$retry_row_file" ]]; then
    cat "$retry_row_file" >> "$RETRY_SUMMARY_FILE"
  fi
}

remove_running_pid() {
  local pid_to_remove="$1"
  local running_pid=""
  local updated_pids=()
  for running_pid in "${RUNNING_PIDS[@]}"; do
    if [[ "$running_pid" != "$pid_to_remove" ]]; then
      updated_pids+=("$running_pid")
    fi
  done
  RUNNING_PIDS=("${updated_pids[@]}")
}

reap_lane_pid() {
  local finished_pid="$1"
  local wrc=0
  local base="${PID_TO_BASE[$finished_pid]}"

  set +e
  wait "$finished_pid"
  wrc="$?"
  set -e

  append_lane_artifacts "$base"
  if [[ "$wrc" -ne 0 ]]; then
    failed="$((failed + 1))"
    if [[ "$KEEP_GOING" -ne 1 ]]; then
      stop_scheduling=1
    fi
  fi

  remove_running_pid "$finished_pid"
  unset "PID_TO_BASE[$finished_pid]"
}

wait_for_any_lane_completion() {
  local finished_pid=""

  while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    for finished_pid in "${RUNNING_PIDS[@]}"; do
      if ! kill -0 "$finished_pid" 2>/dev/null; then
        reap_lane_pid "$finished_pid"
        return 0
      fi
    done
    sleep 0.05
  done

  return 0
}

schedule_lane() {
  local suite_id="$1"
  local engine_kind="$2"
  local cmd="$3"
  local base=""

  if is_lane_done_by_resume "$suite_id" "$engine_kind"; then
    skipped_resume="$((skipped_resume + 1))"
    return 0
  fi

  lane_seq="$((lane_seq + 1))"
  base="${RUN_WORK_DIR}/lane_${lane_seq}"

  if [[ "$JOBS" -le 1 ]]; then
    if ! run_lane_capture "$suite_id" "$engine_kind" "$cmd" "${base}.plan" "${base}.summary" "${base}.retry"; then
      append_lane_artifacts "$base"
      failed="$((failed + 1))"
      if [[ "$KEEP_GOING" -ne 1 ]]; then
        stop_scheduling=1
        return 1
      fi
      return 0
    fi
    append_lane_artifacts "$base"
    return 0
  fi

  (
    run_lane_capture "$suite_id" "$engine_kind" "$cmd" "${base}.plan" "${base}.summary" "${base}.retry"
  ) &
  local pid="$!"
  RUNNING_PIDS+=("$pid")
  PID_TO_BASE["$pid"]="$base"

  while [[ "${#RUNNING_PIDS[@]}" -ge "$JOBS" ]]; do
    wait_for_any_lane_completion
    if [[ "$stop_scheduling" -eq 1 ]]; then
      return 1
    fi
  done

  return 0
}

emit_engine_parity_report() {
  local summary_file="$1"
  local parity_file="$2"

  printf "suite_id\tcirct_status\tcirct_exit_code\txcelium_status\txcelium_exit_code\tparity\treason\n" > "$parity_file"

  awk -F'\t' '
    NR == 1 { next }
    {
      suites[$1] = 1
      if ($2 == "circt") {
        circt_status[$1] = $3
        circt_exit[$1] = $4
      } else if ($2 == "xcelium") {
        xcelium_status[$1] = $3
        xcelium_exit[$1] = $4
      }
    }
    END {
      for (suite in suites) {
        cs = (suite in circt_status) ? circt_status[suite] : ""
        ce = (suite in circt_exit) ? circt_exit[suite] : ""
        xs = (suite in xcelium_status) ? xcelium_status[suite] : ""
        xe = (suite in xcelium_exit) ? xcelium_exit[suite] : ""

        parity = "INCOMPLETE"
        reason = "missing_engine_lane"
        if (cs != "" && xs != "" && cs != "UNCONFIGURED" && xs != "UNCONFIGURED") {
          if (cs == xs && ce == xe) {
            parity = "MATCH"
            reason = "-"
          } else {
            parity = "DIFF"
            reason = ""
            if (cs != xs) {
              reason = "status_mismatch"
            }
            if (ce != xe) {
              if (reason != "")
                reason = reason ",exit_code_mismatch"
              else
                reason = "exit_code_mismatch"
            }
          }
        }

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", suite, cs, ce, xs, xe, parity, reason
      }
    }
  ' "$summary_file" | sort -k1,1 >> "$parity_file"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --adapter-catalog)
      ADAPTER_CATALOG="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --engine-parity-file)
      ENGINE_PARITY_FILE="$2"
      shift 2
      ;;
    --retry-summary-file)
      RETRY_SUMMARY_FILE="$2"
      shift 2
      ;;
    --suite-regex)
      SUITE_REGEX="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --lane-retries)
      LANE_RETRIES="$2"
      shift 2
      ;;
    --lane-retry-delay-ms)
      LANE_RETRY_DELAY_MS="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --shard-count)
      SHARD_COUNT="$2"
      shift 2
      ;;
    --shard-index)
      SHARD_INDEX="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --no-keep-going)
      KEEP_GOING=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$PROFILE" in
  smoke|nightly|full) ;;
  *)
    echo "invalid --profile: $PROFILE (expected smoke|nightly|full)" >&2
    exit 1
    ;;
esac

case "$ENGINE" in
  circt|xcelium|both) ;;
  *)
    echo "invalid --engine: $ENGINE (expected circt|xcelium|both)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi
if [[ ! -r "$MANIFEST" ]]; then
  echo "manifest not readable: $MANIFEST" >&2
  exit 1
fi
if ! is_pos_int "$JOBS"; then
  echo "--jobs must be a positive integer: $JOBS" >&2
  exit 1
fi
if ! is_nonneg_int "$LANE_RETRIES"; then
  echo "--lane-retries must be a non-negative integer: $LANE_RETRIES" >&2
  exit 1
fi
if ! is_nonneg_int "$LANE_RETRY_DELAY_MS"; then
  echo "--lane-retry-delay-ms must be a non-negative integer: $LANE_RETRY_DELAY_MS" >&2
  exit 1
fi
if ! is_pos_int "$SHARD_COUNT"; then
  echo "--shard-count must be a positive integer: $SHARD_COUNT" >&2
  exit 1
fi
if ! is_nonneg_int "$SHARD_INDEX"; then
  echo "--shard-index must be a non-negative integer: $SHARD_INDEX" >&2
  exit 1
fi
if [[ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]]; then
  echo "--shard-index must be in [0, --shard-count): $SHARD_INDEX (count=$SHARD_COUNT)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/logs"
SUMMARY_FILE="$OUT_DIR/summary.tsv"
PLAN_FILE="$OUT_DIR/plan.tsv"
RUN_WORK_DIR="${OUT_DIR}/.work"
rm -rf "$RUN_WORK_DIR"
mkdir -p "$RUN_WORK_DIR"

if [[ -z "$ENGINE_PARITY_FILE" ]]; then
  ENGINE_PARITY_FILE="$OUT_DIR/engine_parity.tsv"
fi
if [[ -z "$RETRY_SUMMARY_FILE" ]]; then
  RETRY_SUMMARY_FILE="$OUT_DIR/retry-summary.tsv"
fi

if [[ "$RESUME" -eq 1 && -f "$SUMMARY_FILE" ]]; then
  load_resume_state "$SUMMARY_FILE"
else
  printf "suite_id\tengine\tstatus\texit_code\telapsed_sec\tlog\n" > "$SUMMARY_FILE"
fi
if [[ "$RESUME" -ne 1 || ! -f "$PLAN_FILE" ]]; then
  printf "suite_id\tengine\tcommand\n" > "$PLAN_FILE"
fi
if [[ "$RESUME" -ne 1 || ! -f "$RETRY_SUMMARY_FILE" ]]; then
  printf "suite_id\tengine\tattempts\tretries_used\tstatus\texit_code\n" > "$RETRY_SUMMARY_FILE"
fi

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  if [[ "$stop_scheduling" -eq 1 ]]; then
    break
  fi

  line="${raw_line%$'\r'}"
  if [[ -z "$line" || "${line:0:1}" == "#" ]]; then
    continue
  fi

  suite_id=""
  row_profiles=""
  circt_cmd=""
  xcelium_cmd=""
  suite_root=""
  circt_adapter=""
  xcelium_adapter=""
  adapter_args=""
  IFS=$'\t' read -r suite_id row_profiles circt_cmd xcelium_cmd suite_root circt_adapter xcelium_adapter adapter_args _rest <<< "$line"

  if [[ -z "$suite_id" ]]; then
    continue
  fi
  if ! matches_profile "$row_profiles"; then
    continue
  fi
  if [[ -n "$SUITE_REGEX" ]] && ! [[ "$suite_id" =~ $SUITE_REGEX ]]; then
    continue
  fi
  if ! in_selected_shard "$suite_id"; then
    continue
  fi

  if [[ -z "$(trim_whitespace "$xcelium_adapter")" || "$(trim_whitespace "$xcelium_adapter")" == "-" ]]; then
    xcelium_adapter="$circt_adapter"
  fi

  if ! circt_cmd_resolved="$(resolve_lane_command "circt" "$circt_cmd" "$suite_root" "$circt_adapter" "$adapter_args")"; then
    exit 1
  fi
  if ! xcelium_cmd_resolved="$(resolve_lane_command "xcelium" "$xcelium_cmd" "$suite_root" "$xcelium_adapter" "$adapter_args")"; then
    exit 1
  fi

  selected="$((selected + 1))"

  case "$ENGINE" in
    circt)
      if ! schedule_lane "$suite_id" "circt" "$circt_cmd_resolved"; then
        break
      fi
      ;;
    xcelium)
      if ! schedule_lane "$suite_id" "xcelium" "$xcelium_cmd_resolved"; then
        break
      fi
      ;;
    both)
      if ! schedule_lane "$suite_id" "circt" "$circt_cmd_resolved"; then
        break
      fi
      if ! schedule_lane "$suite_id" "xcelium" "$xcelium_cmd_resolved"; then
        break
      fi
      ;;
  esac
done < "$MANIFEST"

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_any_lane_completion
done

if [[ "$selected" -eq 0 ]]; then
  echo "no suites selected from manifest: $MANIFEST (profile=$PROFILE${SUITE_REGEX:+, suite_regex=$SUITE_REGEX}, shard=${SHARD_INDEX}/${SHARD_COUNT})" >&2
  exit 1
fi

if [[ "$ENGINE" == "both" ]]; then
  emit_engine_parity_report "$SUMMARY_FILE" "$ENGINE_PARITY_FILE"
fi

echo "[run-regression-unified] manifest=$MANIFEST profile=$PROFILE engine=$ENGINE dry_run=$DRY_RUN"
echo "[run-regression-unified] adapter_catalog=$ADAPTER_CATALOG"
if [[ "$ENGINE" == "both" ]]; then
  echo "[run-regression-unified] engine_parity=$ENGINE_PARITY_FILE"
fi
echo "[run-regression-unified] summary=$SUMMARY_FILE plan=$PLAN_FILE retry_summary=$RETRY_SUMMARY_FILE selected=$selected failures=$failed skipped_resume=$skipped_resume shard=${SHARD_INDEX}/${SHARD_COUNT} lane_retries=$LANE_RETRIES jobs=$JOBS"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

exit 0
