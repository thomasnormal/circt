#!/usr/bin/env bash
# Run multiple mutation-coverage lanes and aggregate lane-level status.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_matrix.sh [options]

Required:
  --lanes-tsv FILE          Lane config TSV:
                              lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<TAB>activate_cmd<TAB>propagate_cmd<TAB>coverage_threshold<TAB>[generate_count]<TAB>[mutations_top]<TAB>[mutations_seed]<TAB>[mutations_yosys]<TAB>[reuse_pair_file]<TAB>[reuse_summary_file]<TAB>[mutations_modes]<TAB>[global_propagate_cmd]<TAB>[global_propagate_circt_lec]

Optional:
  --out-dir DIR             Matrix output dir (default: ./mutation-matrix-results)
  --results-file FILE       Lane summary TSV (default: <out-dir>/results.tsv)
  --create-mutated-script FILE
                            Passed through to run_mutation_cover.sh
  --jobs-per-lane N         Passed through to run_mutation_cover.sh --jobs (default: 1)
  --default-reuse-pair-file FILE
                            Default --reuse-pair-file for lanes that do not set reuse_pair_file
  --default-reuse-summary-file FILE
                            Default --reuse-summary-file for lanes that do not set reuse_summary_file
  --default-mutations-modes CSV
                            Default --mutations-modes for generated-mutation lanes
  --default-formal-global-propagate-cmd CMD
                            Default --formal-global-propagate-cmd for lanes
                            without lane-specific global_propagate_cmd
  --default-formal-global-propagate-circt-lec PATH
                            Default --formal-global-propagate-circt-lec for
                            lanes without lane-specific global_propagate_circt_lec
  --reuse-cache-dir DIR     Passed through to run_mutation_cover.sh --reuse-cache-dir
  --reuse-compat-mode MODE  Passed through to run_mutation_cover.sh reuse compatibility policy
                            (off|warn|strict, default: warn)
  --lane-jobs N             Number of concurrent lanes (default: 1)
  --stop-on-fail            Stop at first failed lane (requires --lane-jobs=1)
  -h, --help                Show help

Notes:
  - Use '-' for activate_cmd or propagate_cmd to disable that stage.
  - coverage_threshold may be '-' to skip threshold gating for a lane.
  - mutations_file may be '-' when generate_count (>0) is provided.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANES_TSV=""
OUT_DIR="${PWD}/mutation-matrix-results"
RESULTS_FILE=""
CREATE_MUTATED_SCRIPT=""
JOBS_PER_LANE=1
DEFAULT_REUSE_PAIR_FILE=""
DEFAULT_REUSE_SUMMARY_FILE=""
DEFAULT_MUTATIONS_MODES=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC=""
REUSE_CACHE_DIR=""
REUSE_COMPAT_MODE="warn"
LANE_JOBS=1
STOP_ON_FAIL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes-tsv) LANES_TSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --create-mutated-script) CREATE_MUTATED_SCRIPT="$2"; shift 2 ;;
    --jobs-per-lane) JOBS_PER_LANE="$2"; shift 2 ;;
    --default-reuse-pair-file) DEFAULT_REUSE_PAIR_FILE="$2"; shift 2 ;;
    --default-reuse-summary-file) DEFAULT_REUSE_SUMMARY_FILE="$2"; shift 2 ;;
    --default-mutations-modes) DEFAULT_MUTATIONS_MODES="$2"; shift 2 ;;
    --default-formal-global-propagate-cmd) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-lec) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC="$2"; shift 2 ;;
    --reuse-cache-dir) REUSE_CACHE_DIR="$2"; shift 2 ;;
    --reuse-compat-mode) REUSE_COMPAT_MODE="$2"; shift 2 ;;
    --lane-jobs) LANE_JOBS="$2"; shift 2 ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LANES_TSV" ]]; then
  echo "Missing required --lanes-tsv." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$LANES_TSV" ]]; then
  echo "Lane file not found: $LANES_TSV" >&2
  exit 1
fi
if [[ ! "$JOBS_PER_LANE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs-per-lane value: $JOBS_PER_LANE" >&2
  exit 1
fi
if [[ ! "$LANE_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --lane-jobs value: $LANE_JOBS" >&2
  exit 1
fi
if [[ -n "$DEFAULT_REUSE_PAIR_FILE" && ! -f "$DEFAULT_REUSE_PAIR_FILE" ]]; then
  echo "Default reuse pair file not found: $DEFAULT_REUSE_PAIR_FILE" >&2
  exit 1
fi
if [[ -n "$DEFAULT_REUSE_SUMMARY_FILE" && ! -f "$DEFAULT_REUSE_SUMMARY_FILE" ]]; then
  echo "Default reuse summary file not found: $DEFAULT_REUSE_SUMMARY_FILE" >&2
  exit 1
fi
if [[ ! "$REUSE_COMPAT_MODE" =~ ^(off|warn|strict)$ ]]; then
  echo "Invalid --reuse-compat-mode value: $REUSE_COMPAT_MODE (expected off|warn|strict)." >&2
  exit 1
fi
if [[ "$STOP_ON_FAIL" -eq 1 && "$LANE_JOBS" -gt 1 ]]; then
  echo "--stop-on-fail requires --lane-jobs=1 for deterministic stop semantics." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
RESULTS_FILE="${RESULTS_FILE:-${OUT_DIR}/results.tsv}"

declare -a LANE_ID
declare -a DESIGN
declare -a MUTATIONS_FILE
declare -a TESTS_MANIFEST
declare -a ACTIVATE_CMD
declare -a PROPAGATE_CMD
declare -a THRESHOLD
declare -a GENERATE_COUNT
declare -a MUTATIONS_TOP
declare -a MUTATIONS_SEED
declare -a MUTATIONS_YOSYS
declare -a REUSE_PAIR_FILE
declare -a REUSE_SUMMARY_FILE
declare -a MUTATIONS_MODES
declare -a GLOBAL_PROPAGATE_CMD
declare -a GLOBAL_PROPAGATE_CIRCT_LEC
declare -a EXECUTED_INDICES

parse_failures=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  IFS=$'\t' read -r lane_id design mutations_file tests_manifest activate_cmd propagate_cmd threshold generate_count mutations_top mutations_seed mutations_yosys reuse_pair_file reuse_summary_file mutations_modes global_propagate_cmd global_propagate_circt_lec _ <<< "$line"
  if [[ -z "$lane_id" || -z "$design" || -z "$mutations_file" || -z "$tests_manifest" ]]; then
    echo "Malformed lane config line: $line" >&2
    parse_failures=$((parse_failures + 1))
    continue
  fi

  LANE_ID+=("$lane_id")
  DESIGN+=("$design")
  MUTATIONS_FILE+=("$mutations_file")
  TESTS_MANIFEST+=("$tests_manifest")
  ACTIVATE_CMD+=("${activate_cmd:-}")
  PROPAGATE_CMD+=("${propagate_cmd:-}")
  THRESHOLD+=("${threshold:-}")
  GENERATE_COUNT+=("${generate_count:--}")
  MUTATIONS_TOP+=("${mutations_top:--}")
  MUTATIONS_SEED+=("${mutations_seed:--}")
  MUTATIONS_YOSYS+=("${mutations_yosys:--}")
  REUSE_PAIR_FILE+=("${reuse_pair_file:--}")
  REUSE_SUMMARY_FILE+=("${reuse_summary_file:--}")
  MUTATIONS_MODES+=("${mutations_modes:--}")
  GLOBAL_PROPAGATE_CMD+=("${global_propagate_cmd:--}")
  GLOBAL_PROPAGATE_CIRCT_LEC+=("${global_propagate_circt_lec:--}")
done < "$LANES_TSV"

if [[ "${#LANE_ID[@]}" -eq 0 ]]; then
  echo "No valid lanes loaded from: $LANES_TSV" >&2
  exit 1
fi

run_lane() {
  local i="$1"
  local lane_id="${LANE_ID[$i]}"
  local lane_dir="${OUT_DIR}/${lane_id}"
  local lane_log="${lane_dir}/lane.log"
  local lane_metrics="${lane_dir}/metrics.tsv"
  local lane_json="${lane_dir}/summary.json"
  local lane_status_file="${lane_dir}/lane_status.tsv"
  local coverage="0.00"
  local gate="UNKNOWN"
  local lane_status="FAIL"
  local rc=1
  local lane_reuse_pair_file=""
  local lane_reuse_summary_file=""
  local lane_mutations_modes=""
  local lane_global_propagate_cmd=""
  local lane_global_propagate_circt_lec=""

  mkdir -p "$lane_dir"

  cmd=(
    "${SCRIPT_DIR}/run_mutation_cover.sh"
    --design "${DESIGN[$i]}"
    --tests-manifest "${TESTS_MANIFEST[$i]}"
    --work-dir "$lane_dir"
    --metrics-file "$lane_metrics"
    --summary-json-file "$lane_json"
    --jobs "$JOBS_PER_LANE"
    --reuse-compat-mode "$REUSE_COMPAT_MODE"
  )
  if [[ -n "$REUSE_CACHE_DIR" ]]; then
    cmd+=(--reuse-cache-dir "$REUSE_CACHE_DIR")
  fi

  if [[ "${MUTATIONS_FILE[$i]}" != "-" ]]; then
    cmd+=(--mutations-file "${MUTATIONS_FILE[$i]}")
  elif [[ "${GENERATE_COUNT[$i]}" != "-" && -n "${GENERATE_COUNT[$i]}" ]]; then
    cmd+=(--generate-mutations "${GENERATE_COUNT[$i]}")
  else
    gate="CONFIG_ERROR"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
    return 0
  fi

  lane_reuse_pair_file="${REUSE_PAIR_FILE[$i]}"
  if [[ "$lane_reuse_pair_file" == "-" || -z "$lane_reuse_pair_file" ]]; then
    lane_reuse_pair_file="$DEFAULT_REUSE_PAIR_FILE"
  fi
  if [[ -n "$lane_reuse_pair_file" ]]; then
    if [[ ! -f "$lane_reuse_pair_file" ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--reuse-pair-file "$lane_reuse_pair_file")
  fi

  lane_reuse_summary_file="${REUSE_SUMMARY_FILE[$i]}"
  if [[ "$lane_reuse_summary_file" == "-" || -z "$lane_reuse_summary_file" ]]; then
    lane_reuse_summary_file="$DEFAULT_REUSE_SUMMARY_FILE"
  fi
  if [[ -n "$lane_reuse_summary_file" ]]; then
    if [[ ! -f "$lane_reuse_summary_file" ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--reuse-summary-file "$lane_reuse_summary_file")
  fi

  if [[ "${GENERATE_COUNT[$i]}" != "-" && -n "${GENERATE_COUNT[$i]}" ]]; then
    lane_mutations_modes="${MUTATIONS_MODES[$i]}"
    if [[ "$lane_mutations_modes" == "-" || -z "$lane_mutations_modes" ]]; then
      lane_mutations_modes="$DEFAULT_MUTATIONS_MODES"
    fi
    if [[ "${MUTATIONS_TOP[$i]}" != "-" && -n "${MUTATIONS_TOP[$i]}" ]]; then
      cmd+=(--mutations-top "${MUTATIONS_TOP[$i]}")
    fi
    if [[ "${MUTATIONS_SEED[$i]}" != "-" && -n "${MUTATIONS_SEED[$i]}" ]]; then
      cmd+=(--mutations-seed "${MUTATIONS_SEED[$i]}")
    fi
    if [[ "${MUTATIONS_YOSYS[$i]}" != "-" && -n "${MUTATIONS_YOSYS[$i]}" ]]; then
      cmd+=(--mutations-yosys "${MUTATIONS_YOSYS[$i]}")
    fi
    if [[ -n "$lane_mutations_modes" ]]; then
      cmd+=(--mutations-modes "$lane_mutations_modes")
    fi
  fi

  if [[ -n "$CREATE_MUTATED_SCRIPT" ]]; then
    cmd+=(--create-mutated-script "$CREATE_MUTATED_SCRIPT")
  fi
  if [[ -n "${ACTIVATE_CMD[$i]}" && "${ACTIVATE_CMD[$i]}" != "-" ]]; then
    cmd+=(--formal-activate-cmd "${ACTIVATE_CMD[$i]}")
  fi
  if [[ -n "${PROPAGATE_CMD[$i]}" && "${PROPAGATE_CMD[$i]}" != "-" ]]; then
    cmd+=(--formal-propagate-cmd "${PROPAGATE_CMD[$i]}")
  fi
  lane_global_propagate_cmd="${GLOBAL_PROPAGATE_CMD[$i]}"
  if [[ "$lane_global_propagate_cmd" == "-" || -z "$lane_global_propagate_cmd" ]]; then
    lane_global_propagate_cmd="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD"
  fi
  if [[ -n "$lane_global_propagate_cmd" ]]; then
    cmd+=(--formal-global-propagate-cmd "$lane_global_propagate_cmd")
  fi
  lane_global_propagate_circt_lec="${GLOBAL_PROPAGATE_CIRCT_LEC[$i]}"
  if [[ "$lane_global_propagate_circt_lec" == "-" || -z "$lane_global_propagate_circt_lec" ]]; then
    lane_global_propagate_circt_lec="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC"
  fi
  if [[ -n "$lane_global_propagate_circt_lec" ]]; then
    cmd+=(--formal-global-propagate-circt-lec "$lane_global_propagate_circt_lec")
  fi
  if [[ -n "${THRESHOLD[$i]}" && "${THRESHOLD[$i]}" != "-" ]]; then
    cmd+=(--coverage-threshold "${THRESHOLD[$i]}")
  fi

  rc=0
  set +e
  "${cmd[@]}" >"$lane_log" 2>&1
  rc=$?
  set -e

  if [[ -f "$lane_metrics" ]]; then
    cov_v="$(awk -F$'\t' '$1=="mutation_coverage_percent"{print $2}' "$lane_metrics" | head -n1)"
    [[ -n "$cov_v" ]] && coverage="$cov_v"
  fi
  if [[ -f "$lane_log" ]]; then
    gate_v="$(awk -F': ' '/^Gate status:/{print $2}' "$lane_log" | tail -n1)"
    [[ -n "$gate_v" ]] && gate="$gate_v"
  fi
  if [[ "$rc" -eq 0 ]]; then
    lane_status="PASS"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
}

if [[ "$LANE_JOBS" -le 1 ]]; then
  for i in "${!LANE_ID[@]}"; do
    run_lane "$i"
    EXECUTED_INDICES+=("$i")
    if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
      lane_status_file="${OUT_DIR}/${LANE_ID[$i]}/lane_status.tsv"
      lane_status="$(awk -F$'\t' 'NR==1{print $2}' "$lane_status_file")"
      if [[ "$lane_status" != "PASS" ]]; then
        break
      fi
    fi
  done
else
  active_jobs=0
  for i in "${!LANE_ID[@]}"; do
    run_lane "$i" &
    EXECUTED_INDICES+=("$i")
    active_jobs=$((active_jobs + 1))
    if [[ "$active_jobs" -ge "$LANE_JOBS" ]]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
  while [[ "$active_jobs" -gt 0 ]]; do
    wait -n
    active_jobs=$((active_jobs - 1))
  done
fi

printf "lane_id\tstatus\texit_code\tcoverage_percent\tgate_status\tlane_dir\tmetrics_file\tsummary_json\n" > "$RESULTS_FILE"
failures="$parse_failures"
passes=0

for i in "${EXECUTED_INDICES[@]}"; do
  lane_status_file="${OUT_DIR}/${LANE_ID[$i]}/lane_status.tsv"
  if [[ ! -f "$lane_status_file" ]]; then
    failures=$((failures + 1))
    printf "%s\tFAIL\t1\t0.00\tMISSING_STATUS\t%s\t%s\t%s\n" \
      "${LANE_ID[$i]}" "${OUT_DIR}/${LANE_ID[$i]}" \
      "${OUT_DIR}/${LANE_ID[$i]}/metrics.tsv" "${OUT_DIR}/${LANE_ID[$i]}/summary.json" >> "$RESULTS_FILE"
    continue
  fi
  cat "$lane_status_file" >> "$RESULTS_FILE"
  lane_status="$(awk -F$'\t' 'NR==1{print $2}' "$lane_status_file")"
  if [[ "$lane_status" == "PASS" ]]; then
    passes=$((passes + 1))
  else
    failures=$((failures + 1))
  fi
done

echo "Mutation matrix summary: pass=${passes} fail=${failures}"
echo "Results: $RESULTS_FILE"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
