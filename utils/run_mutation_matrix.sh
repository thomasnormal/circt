#!/usr/bin/env bash
# Run multiple mutation-coverage lanes and aggregate lane-level status.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_matrix.sh [options]

Required:
  --lanes-tsv FILE          Lane config TSV:
                              lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<TAB>activate_cmd<TAB>propagate_cmd<TAB>coverage_threshold<TAB>[generate_count]<TAB>[mutations_top]<TAB>[mutations_seed]<TAB>[mutations_yosys]<TAB>[reuse_pair_file]<TAB>[reuse_summary_file]<TAB>[mutations_modes]<TAB>[global_propagate_cmd]<TAB>[global_propagate_circt_lec]<TAB>[global_propagate_circt_bmc]<TAB>[global_propagate_bmc_args]<TAB>[global_propagate_bmc_bound]<TAB>[global_propagate_bmc_module]<TAB>[global_propagate_bmc_run_smtlib]<TAB>[global_propagate_bmc_z3]<TAB>[global_propagate_bmc_assume_known_inputs]<TAB>[global_propagate_bmc_ignore_asserts_until]<TAB>[global_propagate_circt_lec_args]<TAB>[global_propagate_c1]<TAB>[global_propagate_c2]<TAB>[global_propagate_z3]<TAB>[global_propagate_assume_known_inputs]<TAB>[global_propagate_accept_xprop_only]<TAB>[mutations_cfg]<TAB>[mutations_select]<TAB>[mutations_profiles]<TAB>[mutations_mode_counts]<TAB>[global_propagate_circt_chain]<TAB>[bmc_orig_cache_max_entries]<TAB>[bmc_orig_cache_max_bytes]

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
                            (concrete: inv,const0,const1,cnot0,cnot1;
                            families: arith,control,balanced,all)
  --default-mutations-mode-counts CSV
                            Default --mutations-mode-counts for generated-mutation lanes
  --default-mutations-profiles CSV
                            Default --mutations-profiles for generated-mutation lanes
  --default-mutations-cfg CSV
                            Default --mutations-cfg for generated-mutation lanes
  --default-mutations-select CSV
                            Default --mutations-select for generated-mutation lanes
  --default-formal-global-propagate-cmd CMD
                            Default --formal-global-propagate-cmd for lanes
                            without lane-specific global_propagate_cmd
  --default-formal-global-propagate-circt-lec PATH
                            Default --formal-global-propagate-circt-lec for
                            lanes without lane-specific global_propagate_circt_lec
  --default-formal-global-propagate-circt-lec-args ARGS
                            Default --formal-global-propagate-circt-lec-args
                            for lanes without lane-specific
                            global_propagate_circt_lec_args
  --default-formal-global-propagate-c1 NAME
                            Default --formal-global-propagate-c1 for lanes
                            without lane-specific global_propagate_c1
  --default-formal-global-propagate-c2 NAME
                            Default --formal-global-propagate-c2 for lanes
                            without lane-specific global_propagate_c2
  --default-formal-global-propagate-z3 PATH
                            Default --formal-global-propagate-z3 for lanes
                            without lane-specific global_propagate_z3
  --default-formal-global-propagate-assume-known-inputs
                            Enable default
                            --formal-global-propagate-assume-known-inputs
                            for lanes using circt-lec global filtering
  --default-formal-global-propagate-accept-xprop-only
                            Enable default
                            --formal-global-propagate-accept-xprop-only
                            for lanes using circt-lec global filtering
  --default-formal-global-propagate-circt-bmc PATH
                            Default --formal-global-propagate-circt-bmc for
                            lanes without lane-specific global_propagate_circt_bmc
  --default-formal-global-propagate-circt-chain MODE
                            Default --formal-global-propagate-circt-chain for
                            lanes without lane-specific global_propagate_circt_chain
                            (lec-then-bmc|bmc-then-lec|consensus|auto)
  --default-formal-global-propagate-circt-bmc-args ARGS
                            Default --formal-global-propagate-circt-bmc-args
                            for lanes without lane-specific global_propagate_bmc_args
  --default-formal-global-propagate-bmc-bound N
                            Default --formal-global-propagate-bmc-bound for
                            lanes without lane-specific global_propagate_bmc_bound
  --default-formal-global-propagate-bmc-module NAME
                            Default --formal-global-propagate-bmc-module for
                            lanes without lane-specific global_propagate_bmc_module
  --default-formal-global-propagate-bmc-run-smtlib
                            Enable default --formal-global-propagate-bmc-run-smtlib
                            for lanes using circt-bmc global filtering
  --default-formal-global-propagate-bmc-z3 PATH
                            Default --formal-global-propagate-bmc-z3 for
                            lanes without lane-specific global_propagate_bmc_z3
  --default-formal-global-propagate-bmc-assume-known-inputs
                            Enable default
                            --formal-global-propagate-bmc-assume-known-inputs
                            for lanes using circt-bmc global filtering
  --default-formal-global-propagate-bmc-ignore-asserts-until N
                            Default --formal-global-propagate-bmc-ignore-asserts-until
                            for lanes without lane-specific
                            global_propagate_bmc_ignore_asserts_until
  --default-bmc-orig-cache-max-entries N
                            Default --bmc-orig-cache-max-entries for lanes
                            without lane-specific bmc_orig_cache_max_entries
  --default-bmc-orig-cache-max-bytes N
                            Default --bmc-orig-cache-max-bytes for lanes
                            without lane-specific bmc_orig_cache_max_bytes
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
DEFAULT_MUTATIONS_MODE_COUNTS=""
DEFAULT_MUTATIONS_PROFILES=""
DEFAULT_MUTATIONS_CFG=""
DEFAULT_MUTATIONS_SELECT=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_C1=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_C2=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_Z3=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS=0
DEFAULT_FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY=0
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_MODULE=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB=0
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_Z3=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS=0
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL=""
DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES=""
DEFAULT_BMC_ORIG_CACHE_MAX_BYTES=""
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
    --default-mutations-mode-counts) DEFAULT_MUTATIONS_MODE_COUNTS="$2"; shift 2 ;;
    --default-mutations-profiles) DEFAULT_MUTATIONS_PROFILES="$2"; shift 2 ;;
    --default-mutations-cfg) DEFAULT_MUTATIONS_CFG="$2"; shift 2 ;;
    --default-mutations-select) DEFAULT_MUTATIONS_SELECT="$2"; shift 2 ;;
    --default-formal-global-propagate-cmd) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-lec) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-lec-args) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS="$2"; shift 2 ;;
    --default-formal-global-propagate-c1) DEFAULT_FORMAL_GLOBAL_PROPAGATE_C1="$2"; shift 2 ;;
    --default-formal-global-propagate-c2) DEFAULT_FORMAL_GLOBAL_PROPAGATE_C2="$2"; shift 2 ;;
    --default-formal-global-propagate-z3) DEFAULT_FORMAL_GLOBAL_PROPAGATE_Z3="$2"; shift 2 ;;
    --default-formal-global-propagate-assume-known-inputs) DEFAULT_FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS=1; shift ;;
    --default-formal-global-propagate-accept-xprop-only) DEFAULT_FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY=1; shift ;;
    --default-formal-global-propagate-circt-bmc) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-chain) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-bmc-args) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS="$2"; shift 2 ;;
    --default-formal-global-propagate-bmc-bound) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND="$2"; shift 2 ;;
    --default-formal-global-propagate-bmc-module) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_MODULE="$2"; shift 2 ;;
    --default-formal-global-propagate-bmc-run-smtlib) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB=1; shift ;;
    --default-formal-global-propagate-bmc-z3) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_Z3="$2"; shift 2 ;;
    --default-formal-global-propagate-bmc-assume-known-inputs) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --default-formal-global-propagate-bmc-ignore-asserts-until) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL="$2"; shift 2 ;;
    --default-bmc-orig-cache-max-entries) DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES="$2"; shift 2 ;;
    --default-bmc-orig-cache-max-bytes) DEFAULT_BMC_ORIG_CACHE_MAX_BYTES="$2"; shift 2 ;;
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
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --default-formal-global-propagate-bmc-bound value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" >&2
  exit 1
fi
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-formal-global-propagate-bmc-ignore-asserts-until value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" >&2
  exit 1
fi
if [[ -n "$DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES" ]] && ! [[ "$DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-bmc-orig-cache-max-entries value: $DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES" >&2
  exit 1
fi
if [[ -n "$DEFAULT_BMC_ORIG_CACHE_MAX_BYTES" ]] && ! [[ "$DEFAULT_BMC_ORIG_CACHE_MAX_BYTES" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-bmc-orig-cache-max-bytes value: $DEFAULT_BMC_ORIG_CACHE_MAX_BYTES" >&2
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
declare -a MUTATIONS_MODE_COUNTS
declare -a MUTATIONS_PROFILES
declare -a MUTATIONS_CFG
declare -a MUTATIONS_SELECT
declare -a GLOBAL_PROPAGATE_CMD
declare -a GLOBAL_PROPAGATE_CIRCT_LEC
declare -a GLOBAL_PROPAGATE_CIRCT_LEC_ARGS
declare -a GLOBAL_PROPAGATE_C1
declare -a GLOBAL_PROPAGATE_C2
declare -a GLOBAL_PROPAGATE_Z3
declare -a GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS
declare -a GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY
declare -a GLOBAL_PROPAGATE_CIRCT_BMC
declare -a GLOBAL_PROPAGATE_CIRCT_CHAIN
declare -a GLOBAL_PROPAGATE_BMC_ARGS
declare -a GLOBAL_PROPAGATE_BMC_BOUND
declare -a GLOBAL_PROPAGATE_BMC_MODULE
declare -a GLOBAL_PROPAGATE_BMC_RUN_SMTLIB
declare -a GLOBAL_PROPAGATE_BMC_Z3
declare -a GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS
declare -a GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL
declare -a BMC_ORIG_CACHE_MAX_ENTRIES
declare -a BMC_ORIG_CACHE_MAX_BYTES
declare -a EXECUTED_INDICES

parse_failures=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  IFS=$'\t' read -r lane_id design mutations_file tests_manifest activate_cmd propagate_cmd threshold generate_count mutations_top mutations_seed mutations_yosys reuse_pair_file reuse_summary_file mutations_modes global_propagate_cmd global_propagate_circt_lec global_propagate_circt_bmc global_propagate_bmc_args global_propagate_bmc_bound global_propagate_bmc_module global_propagate_bmc_run_smtlib global_propagate_bmc_z3 global_propagate_bmc_assume_known_inputs global_propagate_bmc_ignore_asserts_until global_propagate_circt_lec_args global_propagate_c1 global_propagate_c2 global_propagate_z3 global_propagate_assume_known_inputs global_propagate_accept_xprop_only mutations_cfg mutations_select mutations_profiles mutations_mode_counts global_propagate_circt_chain bmc_orig_cache_max_entries bmc_orig_cache_max_bytes _ <<< "$line"
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
  MUTATIONS_MODE_COUNTS+=("${mutations_mode_counts:--}")
  MUTATIONS_PROFILES+=("${mutations_profiles:--}")
  MUTATIONS_CFG+=("${mutations_cfg:--}")
  MUTATIONS_SELECT+=("${mutations_select:--}")
  GLOBAL_PROPAGATE_CMD+=("${global_propagate_cmd:--}")
  GLOBAL_PROPAGATE_CIRCT_LEC+=("${global_propagate_circt_lec:--}")
  GLOBAL_PROPAGATE_CIRCT_LEC_ARGS+=("${global_propagate_circt_lec_args:--}")
  GLOBAL_PROPAGATE_C1+=("${global_propagate_c1:--}")
  GLOBAL_PROPAGATE_C2+=("${global_propagate_c2:--}")
  GLOBAL_PROPAGATE_Z3+=("${global_propagate_z3:--}")
  GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS+=("${global_propagate_assume_known_inputs:--}")
  GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY+=("${global_propagate_accept_xprop_only:--}")
  GLOBAL_PROPAGATE_CIRCT_BMC+=("${global_propagate_circt_bmc:--}")
  GLOBAL_PROPAGATE_CIRCT_CHAIN+=("${global_propagate_circt_chain:--}")
  GLOBAL_PROPAGATE_BMC_ARGS+=("${global_propagate_bmc_args:--}")
  GLOBAL_PROPAGATE_BMC_BOUND+=("${global_propagate_bmc_bound:--}")
  GLOBAL_PROPAGATE_BMC_MODULE+=("${global_propagate_bmc_module:--}")
  GLOBAL_PROPAGATE_BMC_RUN_SMTLIB+=("${global_propagate_bmc_run_smtlib:--}")
  GLOBAL_PROPAGATE_BMC_Z3+=("${global_propagate_bmc_z3:--}")
  GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS+=("${global_propagate_bmc_assume_known_inputs:--}")
  GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL+=("${global_propagate_bmc_ignore_asserts_until:--}")
  BMC_ORIG_CACHE_MAX_ENTRIES+=("${bmc_orig_cache_max_entries:--}")
  BMC_ORIG_CACHE_MAX_BYTES+=("${bmc_orig_cache_max_bytes:--}")
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
  local lane_mutations_mode_counts=""
  local lane_mutations_profiles=""
  local lane_mutations_cfg=""
  local lane_mutations_select=""
  local lane_global_propagate_cmd=""
  local lane_global_propagate_circt_lec=""
  local lane_global_propagate_circt_lec_args=""
  local lane_global_propagate_c1=""
  local lane_global_propagate_c2=""
  local lane_global_propagate_z3=""
  local lane_global_propagate_assume_known_inputs=""
  local lane_global_propagate_accept_xprop_only=""
  local lane_global_propagate_circt_bmc=""
  local lane_global_propagate_circt_chain=""
  local lane_global_propagate_bmc_args=""
  local lane_global_propagate_bmc_bound=""
  local lane_global_propagate_bmc_module=""
  local lane_global_propagate_bmc_run_smtlib=""
  local lane_global_propagate_bmc_z3=""
  local lane_global_propagate_bmc_assume_known_inputs=""
  local lane_global_propagate_bmc_ignore_asserts_until=""
  local lane_bmc_orig_cache_max_entries=""
  local lane_bmc_orig_cache_max_bytes=""

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
    lane_mutations_mode_counts="${MUTATIONS_MODE_COUNTS[$i]}"
    if [[ "$lane_mutations_mode_counts" == "-" || -z "$lane_mutations_mode_counts" ]]; then
      lane_mutations_mode_counts="$DEFAULT_MUTATIONS_MODE_COUNTS"
    fi
    lane_mutations_profiles="${MUTATIONS_PROFILES[$i]}"
    if [[ "$lane_mutations_profiles" == "-" || -z "$lane_mutations_profiles" ]]; then
      lane_mutations_profiles="$DEFAULT_MUTATIONS_PROFILES"
    fi
    lane_mutations_cfg="${MUTATIONS_CFG[$i]}"
    if [[ "$lane_mutations_cfg" == "-" || -z "$lane_mutations_cfg" ]]; then
      lane_mutations_cfg="$DEFAULT_MUTATIONS_CFG"
    fi
    lane_mutations_select="${MUTATIONS_SELECT[$i]}"
    if [[ "$lane_mutations_select" == "-" || -z "$lane_mutations_select" ]]; then
      lane_mutations_select="$DEFAULT_MUTATIONS_SELECT"
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
    if [[ -n "$lane_mutations_mode_counts" ]]; then
      cmd+=(--mutations-mode-counts "$lane_mutations_mode_counts")
    fi
    if [[ -n "$lane_mutations_profiles" ]]; then
      cmd+=(--mutations-profiles "$lane_mutations_profiles")
    fi
    if [[ -n "$lane_mutations_cfg" ]]; then
      cmd+=(--mutations-cfg "$lane_mutations_cfg")
    fi
    if [[ -n "$lane_mutations_select" ]]; then
      cmd+=(--mutations-select "$lane_mutations_select")
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
  lane_global_propagate_circt_lec_args="${GLOBAL_PROPAGATE_CIRCT_LEC_ARGS[$i]}"
  if [[ "$lane_global_propagate_circt_lec_args" == "-" || -z "$lane_global_propagate_circt_lec_args" ]]; then
    lane_global_propagate_circt_lec_args="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS"
  fi
  if [[ -n "$lane_global_propagate_circt_lec_args" ]]; then
    cmd+=(--formal-global-propagate-circt-lec-args "$lane_global_propagate_circt_lec_args")
  fi
  lane_global_propagate_c1="${GLOBAL_PROPAGATE_C1[$i]}"
  if [[ "$lane_global_propagate_c1" == "-" || -z "$lane_global_propagate_c1" ]]; then
    lane_global_propagate_c1="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_C1"
  fi
  if [[ -n "$lane_global_propagate_c1" ]]; then
    cmd+=(--formal-global-propagate-c1 "$lane_global_propagate_c1")
  fi
  lane_global_propagate_c2="${GLOBAL_PROPAGATE_C2[$i]}"
  if [[ "$lane_global_propagate_c2" == "-" || -z "$lane_global_propagate_c2" ]]; then
    lane_global_propagate_c2="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_C2"
  fi
  if [[ -n "$lane_global_propagate_c2" ]]; then
    cmd+=(--formal-global-propagate-c2 "$lane_global_propagate_c2")
  fi
  lane_global_propagate_z3="${GLOBAL_PROPAGATE_Z3[$i]}"
  if [[ "$lane_global_propagate_z3" == "-" || -z "$lane_global_propagate_z3" ]]; then
    lane_global_propagate_z3="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_Z3"
  fi
  if [[ -n "$lane_global_propagate_z3" ]]; then
    cmd+=(--formal-global-propagate-z3 "$lane_global_propagate_z3")
  fi
  lane_global_propagate_assume_known_inputs="${GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS[$i]}"
  if [[ "$lane_global_propagate_assume_known_inputs" == "-" || -z "$lane_global_propagate_assume_known_inputs" ]]; then
    lane_global_propagate_assume_known_inputs="$([[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS" -eq 1 ]] && printf "1" || printf "")"
  fi
  if [[ "$lane_global_propagate_assume_known_inputs" == "1" || "$lane_global_propagate_assume_known_inputs" == "true" || "$lane_global_propagate_assume_known_inputs" == "yes" ]]; then
    cmd+=(--formal-global-propagate-assume-known-inputs)
  fi
  lane_global_propagate_accept_xprop_only="${GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY[$i]}"
  if [[ "$lane_global_propagate_accept_xprop_only" == "-" || -z "$lane_global_propagate_accept_xprop_only" ]]; then
    lane_global_propagate_accept_xprop_only="$([[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY" -eq 1 ]] && printf "1" || printf "")"
  fi
  if [[ "$lane_global_propagate_accept_xprop_only" == "1" || "$lane_global_propagate_accept_xprop_only" == "true" || "$lane_global_propagate_accept_xprop_only" == "yes" ]]; then
    cmd+=(--formal-global-propagate-accept-xprop-only)
  fi

  lane_global_propagate_circt_bmc="${GLOBAL_PROPAGATE_CIRCT_BMC[$i]}"
  if [[ "$lane_global_propagate_circt_bmc" == "-" || -z "$lane_global_propagate_circt_bmc" ]]; then
    lane_global_propagate_circt_bmc="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC"
  fi
  if [[ -n "$lane_global_propagate_circt_bmc" ]]; then
    cmd+=(--formal-global-propagate-circt-bmc "$lane_global_propagate_circt_bmc")
  fi
  lane_global_propagate_circt_chain="${GLOBAL_PROPAGATE_CIRCT_CHAIN[$i]}"
  if [[ "$lane_global_propagate_circt_chain" == "-" || -z "$lane_global_propagate_circt_chain" ]]; then
    lane_global_propagate_circt_chain="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN"
  fi
  if [[ -n "$lane_global_propagate_circt_chain" ]]; then
    cmd+=(--formal-global-propagate-circt-chain "$lane_global_propagate_circt_chain")
  fi

  lane_global_propagate_bmc_args="${GLOBAL_PROPAGATE_BMC_ARGS[$i]}"
  if [[ "$lane_global_propagate_bmc_args" == "-" || -z "$lane_global_propagate_bmc_args" ]]; then
    lane_global_propagate_bmc_args="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS"
  fi
  if [[ -n "$lane_global_propagate_bmc_args" ]]; then
    cmd+=(--formal-global-propagate-circt-bmc-args "$lane_global_propagate_bmc_args")
  fi

  lane_global_propagate_bmc_bound="${GLOBAL_PROPAGATE_BMC_BOUND[$i]}"
  if [[ "$lane_global_propagate_bmc_bound" == "-" || -z "$lane_global_propagate_bmc_bound" ]]; then
    lane_global_propagate_bmc_bound="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND"
  fi
  if [[ -n "$lane_global_propagate_bmc_bound" ]]; then
    if ! [[ "$lane_global_propagate_bmc_bound" =~ ^[1-9][0-9]*$ ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--formal-global-propagate-bmc-bound "$lane_global_propagate_bmc_bound")
  fi

  lane_global_propagate_bmc_module="${GLOBAL_PROPAGATE_BMC_MODULE[$i]}"
  if [[ "$lane_global_propagate_bmc_module" == "-" || -z "$lane_global_propagate_bmc_module" ]]; then
    lane_global_propagate_bmc_module="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_MODULE"
  fi
  if [[ -n "$lane_global_propagate_bmc_module" ]]; then
    cmd+=(--formal-global-propagate-bmc-module "$lane_global_propagate_bmc_module")
  fi

  lane_global_propagate_bmc_run_smtlib="${GLOBAL_PROPAGATE_BMC_RUN_SMTLIB[$i]}"
  if [[ "$lane_global_propagate_bmc_run_smtlib" == "-" || -z "$lane_global_propagate_bmc_run_smtlib" ]]; then
    lane_global_propagate_bmc_run_smtlib="$([[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB" -eq 1 ]] && printf "1" || printf "")"
  fi
  if [[ "$lane_global_propagate_bmc_run_smtlib" == "1" || "$lane_global_propagate_bmc_run_smtlib" == "true" || "$lane_global_propagate_bmc_run_smtlib" == "yes" ]]; then
    cmd+=(--formal-global-propagate-bmc-run-smtlib)
  fi

  lane_global_propagate_bmc_z3="${GLOBAL_PROPAGATE_BMC_Z3[$i]}"
  if [[ "$lane_global_propagate_bmc_z3" == "-" || -z "$lane_global_propagate_bmc_z3" ]]; then
    lane_global_propagate_bmc_z3="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_Z3"
  fi
  if [[ -n "$lane_global_propagate_bmc_z3" ]]; then
    cmd+=(--formal-global-propagate-bmc-z3 "$lane_global_propagate_bmc_z3")
  fi

  lane_global_propagate_bmc_assume_known_inputs="${GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS[$i]}"
  if [[ "$lane_global_propagate_bmc_assume_known_inputs" == "-" || -z "$lane_global_propagate_bmc_assume_known_inputs" ]]; then
    lane_global_propagate_bmc_assume_known_inputs="$([[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS" -eq 1 ]] && printf "1" || printf "")"
  fi
  if [[ "$lane_global_propagate_bmc_assume_known_inputs" == "1" || "$lane_global_propagate_bmc_assume_known_inputs" == "true" || "$lane_global_propagate_bmc_assume_known_inputs" == "yes" ]]; then
    cmd+=(--formal-global-propagate-bmc-assume-known-inputs)
  fi

  lane_global_propagate_bmc_ignore_asserts_until="${GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL[$i]}"
  if [[ "$lane_global_propagate_bmc_ignore_asserts_until" == "-" || -z "$lane_global_propagate_bmc_ignore_asserts_until" ]]; then
    lane_global_propagate_bmc_ignore_asserts_until="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL"
  fi
  if [[ -n "$lane_global_propagate_bmc_ignore_asserts_until" ]]; then
    if ! [[ "$lane_global_propagate_bmc_ignore_asserts_until" =~ ^[0-9]+$ ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--formal-global-propagate-bmc-ignore-asserts-until "$lane_global_propagate_bmc_ignore_asserts_until")
  fi

  lane_bmc_orig_cache_max_entries="${BMC_ORIG_CACHE_MAX_ENTRIES[$i]}"
  if [[ "$lane_bmc_orig_cache_max_entries" == "-" || -z "$lane_bmc_orig_cache_max_entries" ]]; then
    lane_bmc_orig_cache_max_entries="$DEFAULT_BMC_ORIG_CACHE_MAX_ENTRIES"
  fi
  if [[ -n "$lane_bmc_orig_cache_max_entries" ]]; then
    if ! [[ "$lane_bmc_orig_cache_max_entries" =~ ^[0-9]+$ ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--bmc-orig-cache-max-entries "$lane_bmc_orig_cache_max_entries")
  fi

  lane_bmc_orig_cache_max_bytes="${BMC_ORIG_CACHE_MAX_BYTES[$i]}"
  if [[ "$lane_bmc_orig_cache_max_bytes" == "-" || -z "$lane_bmc_orig_cache_max_bytes" ]]; then
    lane_bmc_orig_cache_max_bytes="$DEFAULT_BMC_ORIG_CACHE_MAX_BYTES"
  fi
  if [[ -n "$lane_bmc_orig_cache_max_bytes" ]]; then
    if ! [[ "$lane_bmc_orig_cache_max_bytes" =~ ^[0-9]+$ ]]; then
      gate="CONFIG_ERROR"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" > "$lane_status_file"
      return 0
    fi
    cmd+=(--bmc-orig-cache-max-bytes "$lane_bmc_orig_cache_max_bytes")
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
