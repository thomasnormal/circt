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
  --retain-hours N       Prune run-* directories older than N hours
                         (-1=disabled, default: -1)
  --on-fail-hook PATH    Executable hook invoked on iteration failure
  --on-fail-webhook URL  HTTP webhook POSTed on iteration failure
                         (repeatable; can specify multiple endpoints)
  --webhook-retries N    Retry count per webhook endpoint (default: 0)
  --webhook-backoff-mode MODE
                         Retry backoff mode: fixed | exponential (default: fixed)
  --webhook-backoff-secs N
                         Sleep between webhook retries in seconds (default: 5)
  --webhook-backoff-max-secs N
                         Max backoff sleep in seconds (default: 300)
  --webhook-jitter-secs N
                         Add random jitter [0,N] seconds to webhook retry sleeps
                         (default: 0)
  --webhook-timeout-secs N
                         Per-webhook HTTP timeout in seconds (default: 15)
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
RETAIN_HOURS=-1
ON_FAIL_HOOK=""
WEBHOOK_RETRIES=0
WEBHOOK_BACKOFF_MODE="fixed"
WEBHOOK_BACKOFF_SECS=5
WEBHOOK_BACKOFF_MAX_SECS=300
WEBHOOK_JITTER_SECS=0
WEBHOOK_TIMEOUT_SECS=15
RUN_FORMAL_ALL="utils/run_formal_all.sh"
STRICT_GATE=1

declare -a FORWARD_ARGS=()
declare -a ON_FAIL_WEBHOOKS=()

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
    --retain-hours)
      RETAIN_HOURS="$2"; shift 2 ;;
    --on-fail-hook)
      ON_FAIL_HOOK="$2"; shift 2 ;;
    --on-fail-webhook)
      ON_FAIL_WEBHOOKS+=("$2"); shift 2 ;;
    --webhook-retries)
      WEBHOOK_RETRIES="$2"; shift 2 ;;
    --webhook-backoff-mode)
      WEBHOOK_BACKOFF_MODE="$2"; shift 2 ;;
    --webhook-backoff-secs)
      WEBHOOK_BACKOFF_SECS="$2"; shift 2 ;;
    --webhook-backoff-max-secs)
      WEBHOOK_BACKOFF_MAX_SECS="$2"; shift 2 ;;
    --webhook-jitter-secs)
      WEBHOOK_JITTER_SECS="$2"; shift 2 ;;
    --webhook-timeout-secs)
      WEBHOOK_TIMEOUT_SECS="$2"; shift 2 ;;
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
if ! [[ "$RETAIN_HOURS" =~ ^-?[0-9]+$ ]]; then
  echo "invalid --retain-hours: expected integer >= -1" >&2
  exit 1
fi
if [[ "$RETAIN_HOURS" -lt -1 ]]; then
  echo "invalid --retain-hours: expected integer >= -1" >&2
  exit 1
fi
if ! [[ "$WEBHOOK_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "invalid --webhook-retries: expected non-negative integer" >&2
  exit 1
fi
if [[ "$WEBHOOK_BACKOFF_MODE" != "fixed" && "$WEBHOOK_BACKOFF_MODE" != "exponential" ]]; then
  echo "invalid --webhook-backoff-mode: expected fixed or exponential" >&2
  exit 1
fi
if ! [[ "$WEBHOOK_BACKOFF_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --webhook-backoff-secs: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$WEBHOOK_BACKOFF_MAX_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --webhook-backoff-max-secs: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$WEBHOOK_JITTER_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --webhook-jitter-secs: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$WEBHOOK_TIMEOUT_SECS" =~ ^[0-9]+$ ]] || [[ "$WEBHOOK_TIMEOUT_SECS" == "0" ]]; then
  echo "invalid --webhook-timeout-secs: expected positive integer" >&2
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
if [[ "${#ON_FAIL_WEBHOOKS[@]}" -gt 0 ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "--on-fail-webhook requires curl in PATH" >&2
    exit 1
  fi
fi
for webhook_url in "${ON_FAIL_WEBHOOKS[@]}"; do
  if [[ -z "$webhook_url" ]]; then
    echo "invalid --on-fail-webhook: expected non-empty URL" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
CADENCE_LOG="$OUT_ROOT/cadence.log"
STATE_FILE="$OUT_ROOT/cadence.state"

on_fail_webhooks_csv=""
if [[ "${#ON_FAIL_WEBHOOKS[@]}" -gt 0 ]]; then
  on_fail_webhooks_csv="$(IFS=,; echo "${ON_FAIL_WEBHOOKS[*]}")"
fi

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
echo "interval_secs=$INTERVAL_SECS" >> "$STATE_FILE"
echo "iterations_target=$ITERATIONS" >> "$STATE_FILE"
echo "retain_runs=$RETAIN_RUNS" >> "$STATE_FILE"
echo "retain_hours=$RETAIN_HOURS" >> "$STATE_FILE"
echo "on_fail_hook=$ON_FAIL_HOOK" >> "$STATE_FILE"
echo "on_fail_webhooks=$on_fail_webhooks_csv" >> "$STATE_FILE"
echo "webhook_retries=$WEBHOOK_RETRIES" >> "$STATE_FILE"
echo "webhook_backoff_mode=$WEBHOOK_BACKOFF_MODE" >> "$STATE_FILE"
echo "webhook_backoff_secs=$WEBHOOK_BACKOFF_SECS" >> "$STATE_FILE"
echo "webhook_backoff_max_secs=$WEBHOOK_BACKOFF_MAX_SECS" >> "$STATE_FILE"
echo "webhook_jitter_secs=$WEBHOOK_JITTER_SECS" >> "$STATE_FILE"
echo "webhook_timeout_secs=$WEBHOOK_TIMEOUT_SECS" >> "$STATE_FILE"
echo "strict_gate=$STRICT_GATE" >> "$STATE_FILE"
echo "run_formal_all=$RUN_FORMAL_ALL" >> "$STATE_FILE"

prune_old_runs() {
  local root="$1"
  local retain="$2"
  local retain_hours="$3"

  local -a run_dirs=()
  local run_name
  while IFS= read -r run_name; do
    run_dirs+=("$run_name")
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'run-[0-9][0-9][0-9][0-9]-*' -printf '%f\n' | LC_ALL=C sort)

  if [[ "$retain_hours" -ge 0 ]]; then
    local cutoff_mins=$((retain_hours * 60))
    for run_name in "${run_dirs[@]}"; do
      local run_dir="$root/$run_name"
      if find "$run_dir" -maxdepth 0 -mmin "+$cutoff_mins" -print -quit | grep -q .; then
        rm -rf "$run_dir"
        echo "pruned_run_dir=$run_dir" | tee -a "$CADENCE_LOG"
      fi
    done
    run_dirs=()
    while IFS= read -r run_name; do
      run_dirs+=("$run_name")
    done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'run-[0-9][0-9][0-9][0-9]-*' -printf '%f\n' | LC_ALL=C sort)
  fi

  if [[ "$retain" -le 0 ]]; then
    return 0
  fi

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

post_fail_webhook() {
  local webhook_url="$1"
  local payload="$2"
  local webhook_ts="$3"
  local attempt=0
  local max_attempts=$((WEBHOOK_RETRIES + 1))
  while true; do
    attempt=$((attempt + 1))
    echo "[$webhook_ts] posting on-fail webhook attempt $attempt/$max_attempts: $webhook_url" \
      | tee -a "$CADENCE_LOG"
    set +e
    curl -fsS --max-time "$WEBHOOK_TIMEOUT_SECS" -X POST \
      -H "Content-Type: application/json" --data "$payload" \
      "$webhook_url" >> "$CADENCE_LOG" 2>&1
    local curl_ec=$?
    set -e
    if [[ "$curl_ec" -eq 0 ]]; then
      return 0
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] on-fail webhook failed with exit code $curl_ec after $attempt attempts: $webhook_url" \
        | tee -a "$CADENCE_LOG"
      return 1
    fi
    local retry_index="$attempt"
    local sleep_secs="$WEBHOOK_BACKOFF_SECS"
    if [[ "$WEBHOOK_BACKOFF_MODE" == "exponential" && "$retry_index" -gt 1 ]]; then
      local exp_mul=1
      local step=1
      while [[ "$step" -lt "$retry_index" ]]; do
        exp_mul=$((exp_mul * 2))
        step=$((step + 1))
      done
      sleep_secs=$((WEBHOOK_BACKOFF_SECS * exp_mul))
    fi
    if [[ "$sleep_secs" -gt "$WEBHOOK_BACKOFF_MAX_SECS" ]]; then
      sleep_secs="$WEBHOOK_BACKOFF_MAX_SECS"
    fi
    if [[ "$WEBHOOK_JITTER_SECS" -gt 0 ]]; then
      sleep_secs=$((sleep_secs + (RANDOM % (WEBHOOK_JITTER_SECS + 1))))
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] on-fail webhook attempt failed with exit code $curl_ec (retrying): $webhook_url" \
      | tee -a "$CADENCE_LOG"
    if [[ "$sleep_secs" -gt 0 ]]; then
      echo "webhook_retry_sleep_secs=$sleep_secs" | tee -a "$CADENCE_LOG"
      sleep "$sleep_secs"
    fi
  done
}

invoke_fail_webhooks() {
  local iteration="$1"
  local exit_code="$2"
  local run_dir="$3"
  if [[ "${#ON_FAIL_WEBHOOKS[@]}" -eq 0 ]]; then
    return 0
  fi

  local webhook_ts payload
  webhook_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  payload="$(
    ITERATION="$iteration" \
    EXIT_CODE="$exit_code" \
    RUN_DIR="$run_dir" \
    OUT_ROOT="$OUT_ROOT" \
    CADENCE_LOG_PATH="$CADENCE_LOG" \
    CADENCE_STATE_PATH="$STATE_FILE" \
    WEBHOOK_COUNT="${#ON_FAIL_WEBHOOKS[@]}" \
    EVENT_TS="$webhook_ts" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "event": "formal_cadence_failure",
            "timestamp_utc": os.environ["EVENT_TS"],
            "iteration": int(os.environ["ITERATION"]),
            "exit_code": int(os.environ["EXIT_CODE"]),
            "run_dir": os.environ["RUN_DIR"],
            "out_root": os.environ["OUT_ROOT"],
            "cadence_log": os.environ["CADENCE_LOG_PATH"],
            "cadence_state": os.environ["CADENCE_STATE_PATH"],
            "webhook_count": int(os.environ["WEBHOOK_COUNT"]),
        },
        sort_keys=True,
    )
)
PY
  )"

  local webhook_url
  local failures=0
  for webhook_url in "${ON_FAIL_WEBHOOKS[@]}"; do
    if ! post_fail_webhook "$webhook_url" "$payload" "$webhook_ts"; then
      failures=$((failures + 1))
    fi
  done
  if [[ "$failures" -gt 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] webhook delivery failures: $failures/${#ON_FAIL_WEBHOOKS[@]}" \
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
    invoke_fail_webhooks "$iteration" "$ec" "$run_dir"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failing fast: iteration $iteration failed" \
      | tee -a "$CADENCE_LOG"
    exit "$ec"
  fi

  prune_old_runs "$OUT_ROOT" "$RETAIN_RUNS" "$RETAIN_HOURS"

  if [[ "$ITERATIONS" -ne 0 && "$iteration" -ge "$ITERATIONS" ]]; then
    echo "completed_iterations=$iteration" | tee -a "$CADENCE_LOG"
    exit 0
  fi

  if [[ "$INTERVAL_SECS" -gt 0 ]]; then
    sleep "$INTERVAL_SECS"
  fi
done
