#!/usr/bin/env bash
# Run multiple mutation-coverage lanes and aggregate lane-level status.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_matrix.sh [options]

Required:
  --lanes-tsv FILE          Lane config TSV:
                              lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<TAB>activate_cmd<TAB>propagate_cmd<TAB>coverage_threshold<TAB>[generate_count]<TAB>[mutations_top]<TAB>[mutations_seed]<TAB>[mutations_yosys]

Optional:
  --out-dir DIR             Matrix output dir (default: ./mutation-matrix-results)
  --results-file FILE       Lane summary TSV (default: <out-dir>/results.tsv)
  --create-mutated-script FILE
                            Passed through to run_mutation_cover.sh
  --jobs-per-lane N         Passed through to run_mutation_cover.sh --jobs (default: 1)
  --stop-on-fail            Stop at first failed lane
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
STOP_ON_FAIL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes-tsv) LANES_TSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --create-mutated-script) CREATE_MUTATED_SCRIPT="$2"; shift 2 ;;
    --jobs-per-lane) JOBS_PER_LANE="$2"; shift 2 ;;
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

mkdir -p "$OUT_DIR"
RESULTS_FILE="${RESULTS_FILE:-${OUT_DIR}/results.tsv}"
printf "lane_id\tstatus\texit_code\tcoverage_percent\tgate_status\tlane_dir\tmetrics_file\tsummary_json\n" > "$RESULTS_FILE"

failures=0
passes=0

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  IFS=$'\t' read -r lane_id design mutations_file tests_manifest activate_cmd propagate_cmd threshold generate_count mutations_top mutations_seed mutations_yosys _ <<< "$line"
  if [[ -z "$lane_id" || -z "$design" || -z "$mutations_file" || -z "$tests_manifest" ]]; then
    echo "Malformed lane config line: $line" >&2
    failures=$((failures + 1))
    continue
  fi

  lane_dir="${OUT_DIR}/${lane_id}"
  mkdir -p "$lane_dir"
  lane_log="${lane_dir}/lane.log"
  lane_metrics="${lane_dir}/metrics.tsv"
  lane_json="${lane_dir}/summary.json"

  cmd=(
    "${SCRIPT_DIR}/run_mutation_cover.sh"
    --design "$design"
    --tests-manifest "$tests_manifest"
    --work-dir "$lane_dir"
    --metrics-file "$lane_metrics"
    --summary-json-file "$lane_json"
    --jobs "$JOBS_PER_LANE"
  )
  gen_count="${generate_count:--}"
  if [[ "$mutations_file" != "-" ]]; then
    cmd+=(--mutations-file "$mutations_file")
  elif [[ "$gen_count" != "-" && -n "$gen_count" ]]; then
    cmd+=(--generate-mutations "$gen_count")
  else
    echo "Lane '${lane_id}' missing mutation source (mutations_file or generate_count)." >&2
    failures=$((failures + 1))
    if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
      echo "Mutation matrix summary: pass=${passes} fail=${failures}"
      echo "Results: $RESULTS_FILE"
      exit 1
    fi
    continue
  fi
  if [[ "$gen_count" != "-" && -n "$gen_count" ]]; then
    if [[ "${mutations_top:--}" != "-" && -n "${mutations_top:-}" ]]; then
      cmd+=(--mutations-top "$mutations_top")
    fi
    if [[ "${mutations_seed:--}" != "-" && -n "${mutations_seed:-}" ]]; then
      cmd+=(--mutations-seed "$mutations_seed")
    fi
    if [[ "${mutations_yosys:--}" != "-" && -n "${mutations_yosys:-}" ]]; then
      cmd+=(--mutations-yosys "$mutations_yosys")
    fi
  fi
  if [[ -n "$CREATE_MUTATED_SCRIPT" ]]; then
    cmd+=(--create-mutated-script "$CREATE_MUTATED_SCRIPT")
  fi

  if [[ -n "${activate_cmd:-}" && "$activate_cmd" != "-" ]]; then
    cmd+=(--formal-activate-cmd "$activate_cmd")
  fi
  if [[ -n "${propagate_cmd:-}" && "$propagate_cmd" != "-" ]]; then
    cmd+=(--formal-propagate-cmd "$propagate_cmd")
  fi
  if [[ -n "${threshold:-}" && "$threshold" != "-" ]]; then
    cmd+=(--coverage-threshold "$threshold")
  fi

  rc=0
  set +e
  "${cmd[@]}" >"$lane_log" 2>&1
  rc=$?
  set -e

  coverage="0.00"
  gate="UNKNOWN"
  if [[ -f "$lane_metrics" ]]; then
    cov_v="$(awk -F$'\t' '$1=="mutation_coverage_percent"{print $2}' "$lane_metrics" | head -n1)"
    [[ -n "$cov_v" ]] && coverage="$cov_v"
  fi
  if [[ -f "$lane_log" ]]; then
    gate_v="$(awk -F': ' '/^Gate status:/{print $2}' "$lane_log" | tail -n1)"
    [[ -n "$gate_v" ]] && gate="$gate_v"
  fi

  lane_status="PASS"
  if [[ "$rc" -ne 0 ]]; then
    lane_status="FAIL"
    failures=$((failures + 1))
  else
    passes=$((passes + 1))
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" >> "$RESULTS_FILE"

  if [[ "$lane_status" == "FAIL" && "$STOP_ON_FAIL" -eq 1 ]]; then
    echo "Mutation matrix summary: pass=${passes} fail=${failures}"
    echo "Results: $RESULTS_FILE"
    exit 1
  fi
done < "$LANES_TSV"

echo "Mutation matrix summary: pass=${passes} fail=${failures}"
echo "Results: $RESULTS_FILE"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
