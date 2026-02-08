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
if [[ ! -x "$RUN_FORMAL_ALL" ]]; then
  echo "run_formal_all script not executable: $RUN_FORMAL_ALL" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
CADENCE_LOG="$OUT_ROOT/cadence.log"
STATE_FILE="$OUT_ROOT/cadence.state"

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
echo "interval_secs=$INTERVAL_SECS" >> "$STATE_FILE"
echo "iterations_target=$ITERATIONS" >> "$STATE_FILE"
echo "strict_gate=$STRICT_GATE" >> "$STATE_FILE"
echo "run_formal_all=$RUN_FORMAL_ALL" >> "$STATE_FILE"

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
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failing fast: iteration $iteration failed" \
      | tee -a "$CADENCE_LOG"
    exit "$ec"
  fi

  if [[ "$ITERATIONS" -ne 0 && "$iteration" -ge "$ITERATIONS" ]]; then
    echo "completed_iterations=$iteration" | tee -a "$CADENCE_LOG"
    exit 0
  fi

  if [[ "$INTERVAL_SECS" -gt 0 ]]; then
    sleep "$INTERVAL_SECS"
  fi
done
