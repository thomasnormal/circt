#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_formal_cadence.sh [options] [-- run_formal_all_args...]

Runs `utils/run_formal_all.sh` on a fixed cadence with per-iteration output
directories and fail-fast behavior.

Options:
  --out-root DIR         Root directory for cadence outputs
                         (default: formal-cadence-results-YYYYMMDD)
  --interval-secs N      Sleep interval between runs in seconds (default: 21600)
  --iterations N         Number of iterations to run (0=infinite, default: 0)
  --retain-runs N        Keep only newest N run-* directories (0=keep all,
                         default: 0)
  --on-fail-hook PATH    Executable hook invoked on iteration failure
  --run-formal-all PATH  Path to run_formal_all.sh
                         (default: utils/run_formal_all.sh)
  --strict-gate          Enable strict gate checks (default)
  --no-strict-gate       Disable strict gate checks
  --help                 Show this help

Examples:
  utils/run_formal_cadence.sh --interval-secs 21600 --iterations 4
  utils/run_formal_cadence.sh --out-root /tmp/formal-cadence -- --with-opentitan --opentitan ~/opentitan
USAGE
}

DATE_STR="$(date +%Y%m%d)"
OUT_ROOT="${PWD}/formal-cadence-results-${DATE_STR}"
INTERVAL_SECS=21600
ITERATIONS=0
RETAIN_RUNS=0
ON_FAIL_HOOK=""
RUN_FORMAL_ALL="utils/run_formal_all.sh"
STRICT_GATE=1

declare -a FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --interval-secs)
      INTERVAL_SECS="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    --retain-runs)
      RETAIN_RUNS="$2"; shift 2 ;;
    --on-fail-hook)
      ON_FAIL_HOOK="$2"; shift 2 ;;
    --run-formal-all)
      RUN_FORMAL_ALL="$2"; shift 2 ;;
    --strict-gate)
      STRICT_GATE=1; shift ;;
    --no-strict-gate)
      STRICT_GATE=0; shift ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FORWARD_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$INTERVAL_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --interval-secs: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "invalid --iterations: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$RETAIN_RUNS" =~ ^[0-9]+$ ]]; then
  echo "invalid --retain-runs: expected non-negative integer" >&2
  exit 1
fi
if [[ ! -x "$RUN_FORMAL_ALL" ]]; then
  echo "run_formal_all script not executable: $RUN_FORMAL_ALL" >&2
  exit 1
fi
if [[ -n "$ON_FAIL_HOOK" && ! -x "$ON_FAIL_HOOK" ]]; then
  echo "on-fail hook not executable: $ON_FAIL_HOOK" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
CADENCE_LOG="$OUT_ROOT/cadence.log"
STATE_FILE="$OUT_ROOT/cadence.state"

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
echo "interval_secs=$INTERVAL_SECS" >> "$STATE_FILE"
echo "iterations_target=$ITERATIONS" >> "$STATE_FILE"
echo "retain_runs=$RETAIN_RUNS" >> "$STATE_FILE"
echo "on_fail_hook=$ON_FAIL_HOOK" >> "$STATE_FILE"
echo "strict_gate=$STRICT_GATE" >> "$STATE_FILE"
echo "run_formal_all=$RUN_FORMAL_ALL" >> "$STATE_FILE"

prune_old_runs() {
  local root="$1"
  local retain="$2"
  if [[ "$retain" -le 0 ]]; then
    return 0
  fi

  local -a run_dirs=()
  local run_name
  while IFS= read -r run_name; do
    run_dirs+=("$run_name")
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'run-[0-9][0-9][0-9][0-9]-*' -printf '%f\n' | LC_ALL=C sort)

  local count="${#run_dirs[@]}"
  if [[ "$count" -le "$retain" ]]; then
    return 0
  fi

  local prune_count=$((count - retain))
  local i
  for ((i = 0; i < prune_count; i++)); do
    local prune_dir="$root/${run_dirs[$i]}"
    if [[ -d "$prune_dir" ]]; then
      rm -rf "$prune_dir"
      echo "pruned_run_dir=$prune_dir" | tee -a "$CADENCE_LOG"
    fi
  done
}

invoke_fail_hook() {
  local iteration="$1"
  local exit_code="$2"
  local run_dir="$3"
  if [[ -z "$ON_FAIL_HOOK" ]]; then
    return 0
  fi

  local hook_ts
  hook_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$hook_ts] invoking on-fail hook: $ON_FAIL_HOOK" | tee -a "$CADENCE_LOG"

  set +e
  "$ON_FAIL_HOOK" \
    "$iteration" \
    "$exit_code" \
    "$run_dir" \
    "$OUT_ROOT" \
    "$CADENCE_LOG" \
    "$STATE_FILE" >> "$CADENCE_LOG" 2>&1
  local hook_ec=$?
  set -e

  if [[ "$hook_ec" -ne 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] on-fail hook failed with exit code $hook_ec" \
      | tee -a "$CADENCE_LOG"
  fi
}

iteration=0
while true; do
  iteration=$((iteration + 1))
  run_stamp="$(date +%Y%m%d-%H%M%S)"
  run_dir="$OUT_ROOT/run-$(printf '%04d' "$iteration")-${run_stamp}"
  mkdir -p "$run_dir"
  ln -sfn "$run_dir" "$OUT_ROOT/latest"

  cmd=("$RUN_FORMAL_ALL" "--out-dir" "$run_dir")
  if [[ "$STRICT_GATE" == "1" ]]; then
    cmd+=("--strict-gate")
  fi
  cmd+=("${FORWARD_ARGS[@]}")

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iteration=$iteration run_dir=$run_dir" \
    | tee -a "$CADENCE_LOG"
  echo "command: ${cmd[*]}" | tee -a "$CADENCE_LOG"

  set +e
  "${cmd[@]}" >> "$CADENCE_LOG" 2>&1
  ec=$?
  set -e

  echo "exit_code=$ec" >> "$STATE_FILE"
  echo "last_iteration=$iteration" >> "$STATE_FILE"
  echo "last_run_dir=$run_dir" >> "$STATE_FILE"

  if [[ "$ec" -ne 0 ]]; then
    invoke_fail_hook "$iteration" "$ec" "$run_dir"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failing fast: iteration $iteration failed" \
      | tee -a "$CADENCE_LOG"
    exit "$ec"
  fi

  prune_old_runs "$OUT_ROOT" "$RETAIN_RUNS"

  if [[ "$ITERATIONS" -ne 0 && "$iteration" -ge "$ITERATIONS" ]]; then
    echo "completed_iterations=$iteration" | tee -a "$CADENCE_LOG"
    exit 0
  fi

  if [[ "$INTERVAL_SECS" -gt 0 ]]; then
    sleep "$INTERVAL_SECS"
  fi
done
