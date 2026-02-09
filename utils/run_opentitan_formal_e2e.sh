#!/usr/bin/env bash
# Run OpenTitan end-to-end parity checks (no smoke mode).
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_opentitan_formal_e2e.sh [options]

Options:
  --opentitan-root DIR     OpenTitan checkout root (default: ~/opentitan)
  --out-dir DIR            Output directory for logs/results (default: ./opentitan-formal-e2e)
  --results-file FILE      TSV output file (default: <out-dir>/results.tsv)
  --circt-verilog PATH     circt-verilog binary (default: build/bin/circt-verilog)
  --sim-targets LIST       Comma-separated OpenTitan sim targets
                           (default: gpio,uart,usbdev,i2c)
  --verilog-targets LIST   Comma-separated OpenTitan parse targets
                           (default: gpio,uart,spi_device,usbdev,i2c,dma,keymgr_dpe)
  --sim-timeout SECS       Per-target wall timeout for sim runs (default: 180)
  --impl-filter REGEX      Regex filter for OpenTitan AES S-Box LEC implementations
  --include-masked         Include masked AES S-Box implementations in LEC
  --allow-xprop-only       Accept XPROP_ONLY LEC rows (otherwise they fail parity)
  --lec-assume-known-inputs
                           Run LEC with known-input assumptions enabled
  --lec-x-optimistic       Force optimistic X equivalence in LEC
  --lec-strict-x           Force strict (non-optimistic) X equivalence in LEC
  --skip-sim               Skip OpenTitan sim lane
  --skip-verilog           Skip OpenTitan parse lane
  --skip-lec               Skip OpenTitan LEC lane
  --sim-script PATH        Override sim runner script
  --verilog-script PATH    Override parse runner script
  --lec-script PATH        Override LEC runner script
  -h, --help               Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENTITAN_ROOT="${HOME}/opentitan"
OUT_DIR="${PWD}/opentitan-formal-e2e"
RESULTS_FILE=""
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
SIM_TARGETS="gpio,uart,usbdev,i2c"
VERILOG_TARGETS="gpio,uart,spi_device,usbdev,i2c,dma,keymgr_dpe"
SIM_TIMEOUT=180
IMPL_FILTER=""
INCLUDE_MASKED=0
ALLOW_XPROP_ONLY=0
LEC_ASSUME_KNOWN_INPUTS=0
LEC_X_OPTIMISTIC_MODE="auto"
LEC_X_MODE_FLAG_COUNT=0
RUN_SIM=1
RUN_VERILOG=1
RUN_LEC=1
SIM_SCRIPT="${SCRIPT_DIR}/run_opentitan_circt_sim.sh"
VERILOG_SCRIPT="${SCRIPT_DIR}/run_opentitan_circt_verilog.sh"
LEC_SCRIPT="${SCRIPT_DIR}/run_opentitan_circt_lec.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opentitan-root) OPENTITAN_ROOT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --circt-verilog) CIRCT_VERILOG="$2"; shift 2 ;;
    --sim-targets) SIM_TARGETS="$2"; shift 2 ;;
    --verilog-targets) VERILOG_TARGETS="$2"; shift 2 ;;
    --sim-timeout) SIM_TIMEOUT="$2"; shift 2 ;;
    --impl-filter) IMPL_FILTER="$2"; shift 2 ;;
    --include-masked) INCLUDE_MASKED=1; shift ;;
    --allow-xprop-only) ALLOW_XPROP_ONLY=1; shift ;;
    --lec-assume-known-inputs) LEC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --lec-x-optimistic) LEC_X_OPTIMISTIC_MODE="on"; LEC_X_MODE_FLAG_COUNT=$((LEC_X_MODE_FLAG_COUNT + 1)); shift ;;
    --lec-strict-x) LEC_X_OPTIMISTIC_MODE="off"; LEC_X_MODE_FLAG_COUNT=$((LEC_X_MODE_FLAG_COUNT + 1)); shift ;;
    --skip-sim) RUN_SIM=0; shift ;;
    --skip-verilog) RUN_VERILOG=0; shift ;;
    --skip-lec) RUN_LEC=0; shift ;;
    --sim-script) SIM_SCRIPT="$2"; shift 2 ;;
    --verilog-script) VERILOG_SCRIPT="$2"; shift 2 ;;
    --lec-script) LEC_SCRIPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$LEC_X_OPTIMISTIC_MODE" != "auto" ]] && [[ "$LEC_X_OPTIMISTIC_MODE" != "on" ]] && [[ "$LEC_X_OPTIMISTIC_MODE" != "off" ]]; then
  echo "Invalid LEC X mode: $LEC_X_OPTIMISTIC_MODE" >&2
  exit 1
fi
if [[ "$LEC_X_MODE_FLAG_COUNT" -gt 1 ]]; then
  echo "Use only one of --lec-x-optimistic or --lec-strict-x." >&2
  exit 1
fi

if [[ ! -d "$OPENTITAN_ROOT" ]]; then
  echo "OpenTitan root not found: $OPENTITAN_ROOT" >&2
  exit 1
fi

for required_script in "$SIM_SCRIPT" "$VERILOG_SCRIPT" "$LEC_SCRIPT"; do
  if [[ ! -x "$required_script" ]]; then
    echo "Required executable not found: $required_script" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
if [[ -z "$RESULTS_FILE" ]]; then
  RESULTS_FILE="${OUT_DIR}/results.tsv"
fi
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"
printf "kind\ttarget\tstatus\tdetail\tartifact\n" > "$RESULTS_FILE"

failures=0
passes=0

append_result() {
  local kind="$1"
  local target="$2"
  local status="$3"
  local detail="$4"
  local artifact="$5"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$kind" "$target" "$status" "$detail" "$artifact" >> "$RESULTS_FILE"
}

split_csv() {
  local csv="$1"
  local item=""
  local IFS=','
  read -r -a _csv_items <<< "$csv"
  for item in "${_csv_items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "$item" ]]; then
      printf "%s\n" "$item"
    fi
  done
}

if [[ "$RUN_SIM" == "1" ]]; then
  while IFS= read -r target; do
    [[ -z "$target" ]] && continue
    log_file="${LOG_DIR}/sim-${target}.log"
    if env CIRCT_VERILOG="$CIRCT_VERILOG" OPENTITAN_DIR="$OPENTITAN_ROOT" \
      "$SIM_SCRIPT" "$target" "--timeout=${SIM_TIMEOUT}" > "$log_file" 2>&1; then
      append_result "SIM" "$target" "PASS" "end_to_end" "$log_file"
      passes=$((passes + 1))
    else
      append_result "SIM" "$target" "FAIL" "end_to_end" "$log_file"
      failures=$((failures + 1))
    fi
  done < <(split_csv "$SIM_TARGETS")
fi

if [[ "$RUN_VERILOG" == "1" ]]; then
  while IFS= read -r target; do
    [[ -z "$target" ]] && continue
    log_file="${LOG_DIR}/verilog-${target}.log"
    if env CIRCT_VERILOG="$CIRCT_VERILOG" OPENTITAN_DIR="$OPENTITAN_ROOT" \
      "$VERILOG_SCRIPT" "$target" --ir-hw > "$log_file" 2>&1; then
      append_result "VERILOG" "$target" "PASS" "parse" "$log_file"
      passes=$((passes + 1))
    else
      append_result "VERILOG" "$target" "FAIL" "parse" "$log_file"
      failures=$((failures + 1))
    fi
  done < <(split_csv "$VERILOG_TARGETS")
fi

if [[ "$RUN_LEC" == "1" ]]; then
  lec_results="${OUT_DIR}/lec-results.tsv"
  lec_log="${LOG_DIR}/lec.log"
  lec_cmd=(
    "$LEC_SCRIPT"
    --opentitan-root "$OPENTITAN_ROOT"
    --results-file "$lec_results"
  )
  if [[ -n "$IMPL_FILTER" ]]; then
    lec_cmd+=(--impl-filter "$IMPL_FILTER")
  fi
  if [[ "$INCLUDE_MASKED" == "1" ]]; then
    lec_cmd+=(--include-masked)
  fi
  lec_env=(CIRCT_VERILOG="$CIRCT_VERILOG" LEC_SMOKE_ONLY=0)
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    lec_env+=(LEC_ASSUME_KNOWN_INPUTS=1)
  fi
  if [[ "$LEC_X_OPTIMISTIC_MODE" == "on" ]]; then
    lec_env+=(LEC_X_OPTIMISTIC=1)
  elif [[ "$LEC_X_OPTIMISTIC_MODE" == "off" ]]; then
    lec_env+=(LEC_X_OPTIMISTIC=0)
  fi
  if env "${lec_env[@]}" "${lec_cmd[@]}" > "$lec_log" 2>&1; then
    :
  fi
  if [[ ! -f "$lec_results" ]]; then
    append_result "LEC" "aes_sbox" "FAIL" "missing_results" "$lec_log"
    failures=$((failures + 1))
  else
    while IFS=$'\t' read -r status impl detail suite mode; do
      [[ -z "$status" ]] && continue
      case "$status" in
        PASS)
          append_result "LEC" "$impl" "PASS" "equivalent" "$detail"
          passes=$((passes + 1))
          ;;
        XFAIL)
          if [[ "$ALLOW_XPROP_ONLY" == "1" ]]; then
            append_result "LEC" "$impl" "PASS" "xprop_only_accepted" "$detail"
            passes=$((passes + 1))
          else
            append_result "LEC" "$impl" "FAIL" "xprop_only" "$detail"
            failures=$((failures + 1))
          fi
          ;;
        FAIL|*)
          append_result "LEC" "$impl" "FAIL" "non_equivalent" "$detail"
          failures=$((failures + 1))
          ;;
      esac
    done < "$lec_results"
  fi
fi

echo "OpenTitan E2E summary: pass=${passes} fail=${failures}"
echo "Results: $RESULTS_FILE"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
