#!/usr/bin/env bash
# Run multiple mutation-coverage lanes and aggregate lane-level status.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_matrix.sh [options]

Required:
  --lanes-tsv FILE          Lane config TSV:
                              lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<TAB>activate_cmd<TAB>propagate_cmd<TAB>coverage_threshold<TAB>[generate_count]<TAB>[mutations_top]<TAB>[mutations_seed]<TAB>[mutations_yosys]<TAB>[reuse_pair_file]<TAB>[reuse_summary_file]<TAB>[mutations_modes]<TAB>[global_propagate_cmd]<TAB>[global_propagate_circt_lec]<TAB>[global_propagate_circt_bmc]<TAB>[global_propagate_bmc_args]<TAB>[global_propagate_bmc_bound]<TAB>[global_propagate_bmc_module]<TAB>[global_propagate_bmc_run_smtlib]<TAB>[global_propagate_bmc_z3]<TAB>[global_propagate_bmc_assume_known_inputs]<TAB>[global_propagate_bmc_ignore_asserts_until]<TAB>[global_propagate_circt_lec_args]<TAB>[global_propagate_c1]<TAB>[global_propagate_c2]<TAB>[global_propagate_z3]<TAB>[global_propagate_assume_known_inputs]<TAB>[global_propagate_accept_xprop_only]<TAB>[mutations_cfg]<TAB>[mutations_select]<TAB>[mutations_profiles]<TAB>[mutations_mode_counts]<TAB>[global_propagate_circt_chain]<TAB>[bmc_orig_cache_max_entries]<TAB>[bmc_orig_cache_max_bytes]<TAB>[bmc_orig_cache_max_age_seconds]<TAB>[bmc_orig_cache_eviction_policy]<TAB>[skip_baseline]<TAB>[fail_on_undetected]<TAB>[fail_on_errors]<TAB>[global_propagate_timeout_seconds]<TAB>[global_propagate_lec_timeout_seconds]<TAB>[global_propagate_bmc_timeout_seconds]<TAB>[mutations_mode_weights]

Optional:
  --out-dir DIR             Matrix output dir (default: ./mutation-matrix-results)
  --results-file FILE       Lane summary TSV (default: <out-dir>/results.tsv)
  --gate-summary-file FILE  Gate-status count TSV (default: <out-dir>/gate_summary.tsv)
  --provenance-summary-file FILE
                            Provenance aggregate TSV
                            (default: <out-dir>/provenance_summary.tsv)
  --baseline-results-file FILE
                            Baseline results TSV for provenance strict-gate
                            tuple checks
  --fail-on-new-contract-fingerprint-case-ids
                            Fail when new lane_id::contract_fingerprint tuples
                            appear vs --baseline-results-file
  --fail-on-new-mutation-source-fingerprint-case-ids
                            Fail when new lane_id::mutation_source_fingerprint
                            tuples appear vs --baseline-results-file
  --fail-on-new-contract-fingerprint-identities
                            Fail when new contract fingerprint identities
                            appear vs --baseline-results-file
  --fail-on-new-mutation-source-fingerprint-identities
                            Fail when new mutation-source fingerprint identities
                            appear vs --baseline-results-file
  --fail-on-contract-fingerprint-divergence
                            Fail when current-run lane contract fingerprint
                            identities diverge (cardinality > 1)
  --fail-on-mutation-source-fingerprint-divergence
                            Fail when current-run lane mutation-source
                            fingerprint identities diverge (cardinality > 1)
  --strict-provenance-gate
                            Enable all provenance tuple/identity drift checks
  --provenance-gate-report-json FILE
                            Structured provenance gate report JSON.
                            Defaults to <out-dir>/provenance_gate_report.json
                            when provenance gating/reporting is enabled.
  --provenance-gate-report-tsv FILE
                            Structured provenance gate report TSV.
                            Defaults to <out-dir>/provenance_gate_report.tsv
                            when provenance gating/reporting is enabled.
  --create-mutated-script FILE
                            Passed through to run_mutation_cover.sh
  --jobs-per-lane N         Passed through to run_mutation_cover.sh --jobs (default: 1)
  --skip-baseline           Passed through to run_mutation_cover.sh --skip-baseline
  --fail-on-undetected      Passed through to run_mutation_cover.sh --fail-on-undetected
  --fail-on-errors          Passed through to run_mutation_cover.sh --fail-on-errors
  --default-reuse-pair-file FILE
                            Default --reuse-pair-file for lanes that do not set reuse_pair_file
  --default-reuse-summary-file FILE
                            Default --reuse-summary-file for lanes that do not set reuse_summary_file
  --default-mutations-modes CSV
                            Default --mutations-modes for generated-mutation lanes
                            (concrete: inv,const0,const1,cnot0,cnot1;
                            families: arith,control,balanced,all,
                            stuck,invert,connect)
  --default-mutations-mode-counts CSV
                            Default --mutations-mode-counts for generated-mutation lanes
  --default-mutations-mode-weights CSV
                            Default --mutations-mode-weights for generated-mutation lanes
  --default-mutations-profiles CSV
                            Default --mutations-profiles for generated-mutation lanes
  --default-mutations-cfg CSV
                            Default --mutations-cfg for generated-mutation lanes
  --default-mutations-select CSV
                            Default --mutations-select for generated-mutation lanes
  --default-mutations-seed N
                            Default --mutations-seed for generated-mutation lanes
  --default-mutations-yosys PATH
                            Default --mutations-yosys for generated-mutation lanes
  --default-formal-global-propagate-cmd CMD
                            Default --formal-global-propagate-cmd for lanes
                            without lane-specific global_propagate_cmd
  --default-formal-global-propagate-timeout-seconds N
                            Default --formal-global-propagate-timeout-seconds
                            for lanes without lane-specific
                            global_propagate_timeout_seconds
  --default-formal-global-propagate-lec-timeout-seconds N
                            Default
                            --formal-global-propagate-lec-timeout-seconds
                            for lanes without lane-specific
                            global_propagate_lec_timeout_seconds
  --default-formal-global-propagate-bmc-timeout-seconds N
                            Default
                            --formal-global-propagate-bmc-timeout-seconds
                            for lanes without lane-specific
                            global_propagate_bmc_timeout_seconds
  --default-formal-global-propagate-circt-lec [PATH]
                            Default --formal-global-propagate-circt-lec for
                            lanes without lane-specific global_propagate_circt_lec.
                            If PATH is omitted or set to 'auto', discovery uses
                            run_mutation_cover.sh defaults.
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
  --default-formal-global-propagate-circt-bmc [PATH]
                            Default --formal-global-propagate-circt-bmc for
                            lanes without lane-specific global_propagate_circt_bmc.
                            If PATH is omitted or set to 'auto', discovery uses
                            run_mutation_cover.sh defaults.
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
  --default-bmc-orig-cache-max-age-seconds N
                            Default --bmc-orig-cache-max-age-seconds for lanes
                            without lane-specific bmc_orig_cache_max_age_seconds
  --default-bmc-orig-cache-eviction-policy MODE
                            Default --bmc-orig-cache-eviction-policy for
                            lanes without lane-specific
                            bmc_orig_cache_eviction_policy
                            (lru|fifo|cost-lru)
  --reuse-cache-dir DIR     Passed through to run_mutation_cover.sh --reuse-cache-dir
  --reuse-compat-mode MODE  Passed through to run_mutation_cover.sh reuse compatibility policy
                            (off|warn|strict, default: warn)
  --include-lane-regex REGEX
                            Execute only lane_ids matching any provided ERE
                            (repeatable)
  --exclude-lane-regex REGEX
                            Exclude lane_ids matching any provided ERE
                            (repeatable, applied after include filters)
  --lane-jobs N             Number of concurrent lanes (default: 1)
  --lane-schedule-policy MODE
                            Lane scheduling policy:
                            fifo|cache-aware (default: fifo)
                            cache-aware prioritizes one lane per generated-cache
                            key before scheduling same-key followers
  --stop-on-fail            Stop at first failed lane (requires --lane-jobs=1)
  -h, --help                Show help

Notes:
  - Use '-' for activate_cmd or propagate_cmd to disable that stage.
  - coverage_threshold may be '-' to skip threshold gating for a lane.
  - mutations_file may be '-' when generate_count (>0) is provided.
  - lane TSV boolean override columns
    (`skip_baseline`,`fail_on_undetected`,`fail_on_errors`) accept
    `1|0|true|false|yes|no|-`.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANES_TSV=""
OUT_DIR="${PWD}/mutation-matrix-results"
RESULTS_FILE=""
GATE_SUMMARY_FILE=""
PROVENANCE_SUMMARY_FILE=""
PROVENANCE_GATE_REPORT_JSON=""
PROVENANCE_GATE_REPORT_TSV=""
BASELINE_RESULTS_FILE=""
CREATE_MUTATED_SCRIPT=""
JOBS_PER_LANE=1
SKIP_BASELINE=0
FAIL_ON_UNDETECTED=0
FAIL_ON_ERRORS=0
DEFAULT_REUSE_PAIR_FILE=""
DEFAULT_REUSE_SUMMARY_FILE=""
DEFAULT_MUTATIONS_MODES=""
DEFAULT_MUTATIONS_MODE_COUNTS=""
DEFAULT_MUTATIONS_MODE_WEIGHTS=""
DEFAULT_MUTATIONS_PROFILES=""
DEFAULT_MUTATIONS_CFG=""
DEFAULT_MUTATIONS_SELECT=""
DEFAULT_MUTATIONS_SEED=""
DEFAULT_MUTATIONS_YOSYS=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS=""
DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS=""
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
DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS=""
DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY=""
REUSE_CACHE_DIR=""
REUSE_COMPAT_MODE="warn"
FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS=0
FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS=0
FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES=0
FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES=0
FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE=0
FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE=0
STRICT_PROVENANCE_GATE=0
INCLUDE_LANE_REGEX=()
EXCLUDE_LANE_REGEX=()
LANE_JOBS=1
LANE_SCHEDULE_POLICY="fifo"
STOP_ON_FAIL=0

VALIDATION_ERROR=""
PARSED_ALLOCATION_TOTAL=0
PARSED_ALLOCATION_ENABLED=0

trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

set_validation_error() {
  VALIDATION_ERROR="$1"
}

is_known_mutation_mode() {
  local mode_name="$1"
  case "$mode_name" in
    inv|const0|const1|cnot0|cnot1|arith|control|balanced|all|stuck|invert|connect)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_known_mutation_profile() {
  local profile_name="$1"
  case "$profile_name" in
    arith-depth|control-depth|balanced-depth|fault-basic|fault-stuck|fault-connect|cover|none)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_mode_list_csv() {
  local label="$1"
  local csv="$2"
  local mode=""
  local -a modes=()

  csv="$(trim_ws "$csv")"
  if [[ -z "$csv" ]]; then
    return 0
  fi
  IFS=',' read -r -a modes <<< "$csv"
  for mode in "${modes[@]}"; do
    mode="$(trim_ws "$mode")"
    [[ -z "$mode" ]] && continue
    if ! is_known_mutation_mode "$mode"; then
      set_validation_error "Invalid ${label} mode: $mode (expected inv|const0|const1|cnot0|cnot1|arith|control|balanced|all|stuck|invert|connect)."
      return 1
    fi
  done
  return 0
}

validate_profile_list_csv() {
  local label="$1"
  local csv="$2"
  local profile=""
  local -a profiles=()

  csv="$(trim_ws "$csv")"
  if [[ -z "$csv" ]]; then
    return 0
  fi
  IFS=',' read -r -a profiles <<< "$csv"
  for profile in "${profiles[@]}"; do
    profile="$(trim_ws "$profile")"
    [[ -z "$profile" ]] && continue
    if ! is_known_mutation_profile "$profile"; then
      set_validation_error "Invalid ${label} profile: $profile (expected arith-depth|control-depth|balanced-depth|fault-basic|fault-stuck|fault-connect|cover|none)."
      return 1
    fi
  done
  return 0
}

parse_mode_allocation_csv() {
  local label="$1"
  local csv="$2"
  local value_name="$3"
  local entry=""
  local mode_name=""
  local mode_value=""
  local -a entries=()

  PARSED_ALLOCATION_TOTAL=0
  PARSED_ALLOCATION_ENABLED=0
  csv="$(trim_ws "$csv")"
  if [[ -z "$csv" ]]; then
    return 0
  fi

  IFS=',' read -r -a entries <<< "$csv"
  for entry in "${entries[@]}"; do
    entry="$(trim_ws "$entry")"
    [[ -z "$entry" ]] && continue
    mode_name="${entry%%=*}"
    mode_value="${entry#*=}"
    mode_name="$(trim_ws "$mode_name")"
    mode_value="$(trim_ws "$mode_value")"
    if [[ -z "$mode_name" || "$mode_value" == "$entry" ]]; then
      set_validation_error "Invalid ${label} entry: $entry (expected NAME=${value_name})."
      return 1
    fi
    if ! is_known_mutation_mode "$mode_name"; then
      set_validation_error "Invalid ${label} mode: $mode_name (expected inv|const0|const1|cnot0|cnot1|arith|control|balanced|all|stuck|invert|connect)."
      return 1
    fi
    if [[ ! "$mode_value" =~ ^[1-9][0-9]*$ ]]; then
      set_validation_error "Invalid ${label} value for ${mode_name}: ${mode_value} (expected positive integer)."
      return 1
    fi
    PARSED_ALLOCATION_TOTAL=$((PARSED_ALLOCATION_TOTAL + mode_value))
    PARSED_ALLOCATION_ENABLED=1
  done

  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes-tsv) LANES_TSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --gate-summary-file) GATE_SUMMARY_FILE="$2"; shift 2 ;;
    --provenance-summary-file) PROVENANCE_SUMMARY_FILE="$2"; shift 2 ;;
    --baseline-results-file) BASELINE_RESULTS_FILE="$2"; shift 2 ;;
    --fail-on-new-contract-fingerprint-case-ids) FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS=1; shift ;;
    --fail-on-new-mutation-source-fingerprint-case-ids) FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS=1; shift ;;
    --fail-on-new-contract-fingerprint-identities) FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES=1; shift ;;
    --fail-on-new-mutation-source-fingerprint-identities) FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES=1; shift ;;
    --fail-on-contract-fingerprint-divergence) FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE=1; shift ;;
    --fail-on-mutation-source-fingerprint-divergence) FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE=1; shift ;;
    --strict-provenance-gate) STRICT_PROVENANCE_GATE=1; shift ;;
    --provenance-gate-report-json) PROVENANCE_GATE_REPORT_JSON="$2"; shift 2 ;;
    --provenance-gate-report-tsv) PROVENANCE_GATE_REPORT_TSV="$2"; shift 2 ;;
    --create-mutated-script) CREATE_MUTATED_SCRIPT="$2"; shift 2 ;;
    --jobs-per-lane) JOBS_PER_LANE="$2"; shift 2 ;;
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --fail-on-undetected) FAIL_ON_UNDETECTED=1; shift ;;
    --fail-on-errors) FAIL_ON_ERRORS=1; shift ;;
    --default-reuse-pair-file) DEFAULT_REUSE_PAIR_FILE="$2"; shift 2 ;;
    --default-reuse-summary-file) DEFAULT_REUSE_SUMMARY_FILE="$2"; shift 2 ;;
    --default-mutations-modes) DEFAULT_MUTATIONS_MODES="$2"; shift 2 ;;
    --default-mutations-mode-counts) DEFAULT_MUTATIONS_MODE_COUNTS="$2"; shift 2 ;;
    --default-mutations-mode-weights) DEFAULT_MUTATIONS_MODE_WEIGHTS="$2"; shift 2 ;;
    --default-mutations-profiles) DEFAULT_MUTATIONS_PROFILES="$2"; shift 2 ;;
    --default-mutations-cfg) DEFAULT_MUTATIONS_CFG="$2"; shift 2 ;;
    --default-mutations-select) DEFAULT_MUTATIONS_SELECT="$2"; shift 2 ;;
    --default-mutations-seed) DEFAULT_MUTATIONS_SEED="$2"; shift 2 ;;
    --default-mutations-yosys) DEFAULT_MUTATIONS_YOSYS="$2"; shift 2 ;;
    --default-formal-global-propagate-cmd) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --default-formal-global-propagate-timeout-seconds) DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --default-formal-global-propagate-lec-timeout-seconds) DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --default-formal-global-propagate-bmc-timeout-seconds) DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --default-formal-global-propagate-circt-lec)
      if [[ "$#" -gt 1 && "${2:0:2}" != "--" ]]; then
        DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC="$2"
        shift 2
      else
        DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC="auto"
        shift
      fi
      ;;
    --default-formal-global-propagate-circt-lec-args) DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS="$2"; shift 2 ;;
    --default-formal-global-propagate-c1) DEFAULT_FORMAL_GLOBAL_PROPAGATE_C1="$2"; shift 2 ;;
    --default-formal-global-propagate-c2) DEFAULT_FORMAL_GLOBAL_PROPAGATE_C2="$2"; shift 2 ;;
    --default-formal-global-propagate-z3) DEFAULT_FORMAL_GLOBAL_PROPAGATE_Z3="$2"; shift 2 ;;
    --default-formal-global-propagate-assume-known-inputs) DEFAULT_FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS=1; shift ;;
    --default-formal-global-propagate-accept-xprop-only) DEFAULT_FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY=1; shift ;;
    --default-formal-global-propagate-circt-bmc)
      if [[ "$#" -gt 1 && "${2:0:2}" != "--" ]]; then
        DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC="$2"
        shift 2
      else
        DEFAULT_FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC="auto"
        shift
      fi
      ;;
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
    --default-bmc-orig-cache-max-age-seconds) DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS="$2"; shift 2 ;;
    --default-bmc-orig-cache-eviction-policy) DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY="$2"; shift 2 ;;
    --reuse-cache-dir) REUSE_CACHE_DIR="$2"; shift 2 ;;
    --reuse-compat-mode) REUSE_COMPAT_MODE="$2"; shift 2 ;;
    --include-lane-regex) INCLUDE_LANE_REGEX+=("$2"); shift 2 ;;
    --exclude-lane-regex) EXCLUDE_LANE_REGEX+=("$2"); shift 2 ;;
    --lane-jobs) LANE_JOBS="$2"; shift 2 ;;
    --lane-schedule-policy) LANE_SCHEDULE_POLICY="$2"; shift 2 ;;
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
if [[ ! "$LANE_SCHEDULE_POLICY" =~ ^(fifo|cache-aware)$ ]]; then
  echo "Invalid --lane-schedule-policy value: $LANE_SCHEDULE_POLICY (expected fifo|cache-aware)." >&2
  exit 1
fi
if [[ -n "$DEFAULT_MUTATIONS_SEED" ]] && ! [[ "$DEFAULT_MUTATIONS_SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-mutations-seed value: $DEFAULT_MUTATIONS_SEED" >&2
  exit 1
fi
if ! validate_mode_list_csv "--default-mutations-modes" "$DEFAULT_MUTATIONS_MODES"; then
  echo "$VALIDATION_ERROR" >&2
  exit 1
fi
if ! validate_profile_list_csv "--default-mutations-profiles" "$DEFAULT_MUTATIONS_PROFILES"; then
  echo "$VALIDATION_ERROR" >&2
  exit 1
fi
if ! parse_mode_allocation_csv "--default-mutations-mode-counts" "$DEFAULT_MUTATIONS_MODE_COUNTS" "COUNT"; then
  echo "$VALIDATION_ERROR" >&2
  exit 1
fi
default_mode_counts_enabled="$PARSED_ALLOCATION_ENABLED"
if ! parse_mode_allocation_csv "--default-mutations-mode-weights" "$DEFAULT_MUTATIONS_MODE_WEIGHTS" "WEIGHT"; then
  echo "$VALIDATION_ERROR" >&2
  exit 1
fi
default_mode_weights_enabled="$PARSED_ALLOCATION_ENABLED"
if [[ "$default_mode_counts_enabled" -eq 1 && "$default_mode_weights_enabled" -eq 1 ]]; then
  echo "Use either --default-mutations-mode-counts or --default-mutations-mode-weights, not both." >&2
  exit 1
fi
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --default-formal-global-propagate-bmc-bound value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" >&2
  exit 1
fi
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-formal-global-propagate-timeout-seconds value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS" >&2
  exit 1
fi
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-formal-global-propagate-lec-timeout-seconds value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS" >&2
  exit 1
fi
if [[ -n "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS" ]] && ! [[ "$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-formal-global-propagate-bmc-timeout-seconds value: $DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS" >&2
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
if [[ -n "$DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS" ]] && ! [[ "$DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --default-bmc-orig-cache-max-age-seconds value: $DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS" >&2
  exit 1
fi
if [[ -n "$DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY" ]] && ! [[ "$DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY" =~ ^(lru|fifo|cost-lru)$ ]]; then
  echo "Invalid --default-bmc-orig-cache-eviction-policy value: $DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY (expected lru|fifo|cost-lru)." >&2
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

for lane_regex in "${INCLUDE_LANE_REGEX[@]}"; do
  if printf "" | grep -Eq "$lane_regex" >/dev/null 2>&1; then
    :
  else
    rc=$?
    if [[ "$rc" -eq 2 ]]; then
      echo "Invalid --include-lane-regex value: $lane_regex" >&2
      exit 1
    fi
  fi
done
for lane_regex in "${EXCLUDE_LANE_REGEX[@]}"; do
  if printf "" | grep -Eq "$lane_regex" >/dev/null 2>&1; then
    :
  else
    rc=$?
    if [[ "$rc" -eq 2 ]]; then
      echo "Invalid --exclude-lane-regex value: $lane_regex" >&2
      exit 1
    fi
  fi
done
if [[ "$STOP_ON_FAIL" -eq 1 && "$LANE_JOBS" -gt 1 ]]; then
  echo "--stop-on-fail requires --lane-jobs=1 for deterministic stop semantics." >&2
  exit 1
fi
if [[ "$STRICT_PROVENANCE_GATE" -eq 1 ]]; then
  FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS=1
  FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS=1
  FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES=1
  FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES=1
fi
if [[ "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES" -eq 1 ]]; then
  if [[ -z "$BASELINE_RESULTS_FILE" ]]; then
    echo "Provenance strict-gate requires --baseline-results-file." >&2
    exit 1
  fi
  if [[ ! -f "$BASELINE_RESULTS_FILE" ]]; then
    echo "Baseline results file not found: $BASELINE_RESULTS_FILE" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
RESULTS_FILE="${RESULTS_FILE:-${OUT_DIR}/results.tsv}"
GATE_SUMMARY_FILE="${GATE_SUMMARY_FILE:-${OUT_DIR}/gate_summary.tsv}"
PROVENANCE_SUMMARY_FILE="${PROVENANCE_SUMMARY_FILE:-${OUT_DIR}/provenance_summary.tsv}"
PROVENANCE_GATE_REPORT_ENABLED=0
if [[ "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES" -eq 1 || "$FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE" -eq 1 || "$FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE" -eq 1 || -n "$PROVENANCE_GATE_REPORT_JSON" || -n "$PROVENANCE_GATE_REPORT_TSV" ]]; then
  PROVENANCE_GATE_REPORT_ENABLED=1
  PROVENANCE_GATE_REPORT_JSON="${PROVENANCE_GATE_REPORT_JSON:-${OUT_DIR}/provenance_gate_report.json}"
  PROVENANCE_GATE_REPORT_TSV="${PROVENANCE_GATE_REPORT_TSV:-${OUT_DIR}/provenance_gate_report.tsv}"
fi

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
declare -a MUTATIONS_MODE_WEIGHTS
declare -a MUTATIONS_PROFILES
declare -a MUTATIONS_CFG
declare -a MUTATIONS_SELECT
declare -a GLOBAL_PROPAGATE_CMD
declare -a GLOBAL_PROPAGATE_TIMEOUT_SECONDS
declare -a GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS
declare -a GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS
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
declare -a BMC_ORIG_CACHE_MAX_AGE_SECONDS
declare -a BMC_ORIG_CACHE_EVICTION_POLICY
declare -a LANE_SKIP_BASELINE
declare -a LANE_FAIL_ON_UNDETECTED
declare -a LANE_FAIL_ON_ERRORS
SELECTED_INDICES=()
EXECUTED_INDICES=()
SCHEDULED_INDICES=()

hash_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return
  fi
  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 | awk '{print $NF}'
    return
  fi
  python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
}

hash_string() {
  local s="$1"
  printf "%s" "$s" | hash_stdin
}

hash_file() {
  local path="$1"
  hash_stdin < "$path"
}

collect_fingerprint_case_ids_from_results() {
  local results_file="$1"
  local fingerprint_column="$2"
  python3 - "$results_file" "$fingerprint_column" <<'PY'
import csv
import sys

results_path = sys.argv[1]
fingerprint_column = sys.argv[2]

with open(results_path, newline="") as f:
  reader = csv.DictReader(f, delimiter="	")
  if reader.fieldnames is None:
    print(f"results file has no header: {results_path}", file=sys.stderr)
    raise SystemExit(2)
  if "lane_id" not in reader.fieldnames:
    print(f"results file missing required lane_id column: {results_path}", file=sys.stderr)
    raise SystemExit(3)
  if fingerprint_column not in reader.fieldnames:
    print(
        f"results file missing required {fingerprint_column} column: {results_path}",
        file=sys.stderr,
    )
    raise SystemExit(4)

  case_ids = set()
  for row in reader:
    lane_id = (row.get("lane_id") or "").strip()
    fingerprint = (row.get(fingerprint_column) or "").strip()
    if not lane_id or not fingerprint or fingerprint == "-":
      continue
    case_ids.add(f"{lane_id}::{fingerprint}")

print(";".join(sorted(case_ids)))
PY
}


collect_fingerprint_identities_from_results() {
  local results_file="$1"
  local fingerprint_column="$2"
  python3 - "$results_file" "$fingerprint_column" <<'PY'
import csv
import sys

results_path = sys.argv[1]
fingerprint_column = sys.argv[2]

with open(results_path, newline="") as f:
  reader = csv.DictReader(f, delimiter="	")
  if reader.fieldnames is None:
    print(f"results file has no header: {results_path}", file=sys.stderr)
    raise SystemExit(2)
  if fingerprint_column not in reader.fieldnames:
    print(
        f"results file missing required {fingerprint_column} column: {results_path}",
        file=sys.stderr,
    )
    raise SystemExit(4)

  identities = set()
  for row in reader:
    fingerprint = (row.get(fingerprint_column) or "").strip()
    if not fingerprint or fingerprint == "-":
      continue
    identities.add(fingerprint)

print(";".join(sorted(identities)))
PY
}

compute_new_case_ids() {
  local baseline_case_ids="$1"
  local current_case_ids="$2"
  python3 - "$baseline_case_ids" "$current_case_ids" <<'PY'
import sys

def parse_case_ids(raw: str):
  values = set()
  for token in raw.split(";"):
    token = token.strip()
    if token:
      values.add(token)
  return values

baseline = parse_case_ids(sys.argv[1])
current = parse_case_ids(sys.argv[2])
new = sorted(current - baseline)

print(len(baseline))
print(len(current))
print(len(new))
print(";".join(new))
PY
}

case_ids_sample() {
  local case_ids_csv="$1"
  local limit="${2:-3}"
  local sample=""
  local i=0
  local -a case_ids=()

  if [[ -z "$case_ids_csv" ]]; then
    printf "none"
    return
  fi

  IFS=';' read -r -a case_ids <<< "$case_ids_csv"
  while [[ "$i" -lt "${#case_ids[@]}" && "$i" -lt "$limit" ]]; do
    if [[ -n "${case_ids[$i]}" ]]; then
      if [[ -n "$sample" ]]; then
        sample+=", "
      fi
      sample+="${case_ids[$i]}"
    fi
    i=$((i + 1))
  done
  if [[ "${#case_ids[@]}" -gt "$limit" ]]; then
    sample+=", ..."
  fi
  printf "%s" "$sample"
}

sample_fingerprint_count_pairs() {
  local map_name="$1"
  local limit="${2:-3}"
  local sample=""
  local count=0
  local fingerprint=""
  declare -n fingerprint_counts="$map_name"

  while IFS= read -r fingerprint; do
    [[ -z "$fingerprint" ]] && continue
    if [[ -n "$sample" ]]; then
      sample+=", "
    fi
    sample+="${fingerprint}:${fingerprint_counts[$fingerprint]}"
    count=$((count + 1))
    if [[ "$count" -ge "$limit" ]]; then
      break
    fi
  done < <(printf "%s\n" "${!fingerprint_counts[@]}" | sort)

  if [[ "${#fingerprint_counts[@]}" -gt "$limit" ]]; then
    sample+=", ..."
  fi
  if [[ -z "$sample" ]]; then
    sample="none"
  fi
  printf "%s" "$sample"
}

sanitize_provenance_gate_field() {
  local value="$1"
  value="${value//$'\t'/ }"
  value="${value//$'\r'/ }"
  value="${value//$'\n'/ }"
  printf "%s" "$value"
}

append_provenance_gate_diagnostic() {
  local diagnostics_file="$1"
  local rule_id="$2"
  local detail="$3"
  local message="$4"

  if [[ -z "$diagnostics_file" ]]; then
    return
  fi

  rule_id="$(sanitize_provenance_gate_field "$rule_id")"
  detail="$(sanitize_provenance_gate_field "$detail")"
  message="$(sanitize_provenance_gate_field "$message")"
  printf "%s\t%s\t%s\n" "$rule_id" "$detail" "$message" >> "$diagnostics_file"
}

write_provenance_gate_report() {
  local status="$1"
  local diagnostics_file="$2"
  local report_json="$3"
  local report_tsv="$4"

  PROVENANCE_GATE_STATUS="$status" \
  PROVENANCE_GATE_DIAGNOSTICS_FILE="$diagnostics_file" \
  PROVENANCE_GATE_REPORT_JSON="$report_json" \
  PROVENANCE_GATE_REPORT_TSV="$report_tsv" \
  PROVENANCE_GATE_FAILURES="$provenance_gate_failures" \
  STRICT_PROVENANCE_GATE="$STRICT_PROVENANCE_GATE" \
  FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS="$FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS" \
  FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS="$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS" \
  FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES="$FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES" \
  FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES="$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES" \
  FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE="$FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE" \
  FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE="$FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE" \
  BASELINE_RESULTS_FILE="$BASELINE_RESULTS_FILE" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
from pathlib import Path

status = os.environ.get("PROVENANCE_GATE_STATUS", "pass").strip().lower()
if status not in {"pass", "fail"}:
  status = "pass"

diagnostics_path = Path(os.environ.get("PROVENANCE_GATE_DIAGNOSTICS_FILE", ""))
report_json = os.environ.get("PROVENANCE_GATE_REPORT_JSON", "").strip()
report_tsv = os.environ.get("PROVENANCE_GATE_REPORT_TSV", "").strip()

diagnostics = []
if diagnostics_path.exists():
  with diagnostics_path.open() as f:
    for raw_line in f:
      line = raw_line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t", 2)
      while len(parts) < 3:
        parts.append("")
      diagnostics.append(
          {
              "rule_id": parts[0],
              "detail": parts[1],
              "message": parts[2],
          }
      )

payload = {
    "schema_version": 1,
    "status": status,
    "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat()
    + "Z",
    "baseline_results_file": os.environ.get("BASELINE_RESULTS_FILE", "").strip(),
    "strict_provenance_gate": os.environ.get("STRICT_PROVENANCE_GATE", "0") == "1",
    "fail_on_new_contract_fingerprint_case_ids": os.environ.get(
        "FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS", "0"
    )
    == "1",
    "fail_on_new_mutation_source_fingerprint_case_ids": os.environ.get(
        "FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS", "0"
    )
    == "1",
    "fail_on_new_contract_fingerprint_identities": os.environ.get(
        "FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES", "0"
    )
    == "1",
    "fail_on_new_mutation_source_fingerprint_identities": os.environ.get(
        "FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES", "0"
    )
    == "1",
    "fail_on_contract_fingerprint_divergence": os.environ.get(
        "FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE", "0"
    )
    == "1",
    "fail_on_mutation_source_fingerprint_divergence": os.environ.get(
        "FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE", "0"
    )
    == "1",
    "provenance_gate_failures": int(os.environ.get("PROVENANCE_GATE_FAILURES", "0")),
    "diagnostic_count": len(diagnostics),
    "diagnostics": diagnostics,
}

if report_json:
  json_path = Path(report_json)
  json_path.parent.mkdir(parents=True, exist_ok=True)
  with json_path.open("w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")

if report_tsv:
  tsv_path = Path(report_tsv)
  tsv_path.parent.mkdir(parents=True, exist_ok=True)
  with tsv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        delimiter="\t",
        fieldnames=["status", "rule_id", "detail", "message"],
    )
    writer.writeheader()
    for diagnostic in diagnostics:
      writer.writerow(
          {
              "status": status,
              "rule_id": diagnostic["rule_id"],
              "detail": diagnostic["detail"],
              "message": diagnostic["message"],
          }
      )
PY
}
parse_failures=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  IFS=$'\t' read -r lane_id design mutations_file tests_manifest activate_cmd propagate_cmd threshold generate_count mutations_top mutations_seed mutations_yosys reuse_pair_file reuse_summary_file mutations_modes global_propagate_cmd global_propagate_circt_lec global_propagate_circt_bmc global_propagate_bmc_args global_propagate_bmc_bound global_propagate_bmc_module global_propagate_bmc_run_smtlib global_propagate_bmc_z3 global_propagate_bmc_assume_known_inputs global_propagate_bmc_ignore_asserts_until global_propagate_circt_lec_args global_propagate_c1 global_propagate_c2 global_propagate_z3 global_propagate_assume_known_inputs global_propagate_accept_xprop_only mutations_cfg mutations_select mutations_profiles mutations_mode_counts global_propagate_circt_chain bmc_orig_cache_max_entries bmc_orig_cache_max_bytes bmc_orig_cache_max_age_seconds bmc_orig_cache_eviction_policy lane_skip_baseline lane_fail_on_undetected lane_fail_on_errors global_propagate_timeout_seconds global_propagate_lec_timeout_seconds global_propagate_bmc_timeout_seconds mutations_mode_weights _ <<< "$line"
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
  MUTATIONS_MODE_WEIGHTS+=("${mutations_mode_weights:--}")
  MUTATIONS_PROFILES+=("${mutations_profiles:--}")
  MUTATIONS_CFG+=("${mutations_cfg:--}")
  MUTATIONS_SELECT+=("${mutations_select:--}")
  GLOBAL_PROPAGATE_CMD+=("${global_propagate_cmd:--}")
  GLOBAL_PROPAGATE_TIMEOUT_SECONDS+=("${global_propagate_timeout_seconds:--}")
  GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS+=("${global_propagate_lec_timeout_seconds:--}")
  GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS+=("${global_propagate_bmc_timeout_seconds:--}")
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
  BMC_ORIG_CACHE_MAX_AGE_SECONDS+=("${bmc_orig_cache_max_age_seconds:--}")
  BMC_ORIG_CACHE_EVICTION_POLICY+=("${bmc_orig_cache_eviction_policy:--}")
  LANE_SKIP_BASELINE+=("${lane_skip_baseline:--}")
  LANE_FAIL_ON_UNDETECTED+=("${lane_fail_on_undetected:--}")
  LANE_FAIL_ON_ERRORS+=("${lane_fail_on_errors:--}")
done < "$LANES_TSV"

if [[ "${#LANE_ID[@]}" -eq 0 ]]; then
  echo "No valid lanes loaded from: $LANES_TSV" >&2
  exit 1
fi

lane_matches_selected_filters() {
  local lane_id="$1"
  local lane_regex=""
  local include_matched=0

  if [[ "${#INCLUDE_LANE_REGEX[@]}" -gt 0 ]]; then
    include_matched=0
    for lane_regex in "${INCLUDE_LANE_REGEX[@]}"; do
      if [[ "$lane_id" =~ $lane_regex ]]; then
        include_matched=1
        break
      fi
    done
    if [[ "$include_matched" -ne 1 ]]; then
      return 1
    fi
  fi

  for lane_regex in "${EXCLUDE_LANE_REGEX[@]}"; do
    if [[ "$lane_id" =~ $lane_regex ]]; then
      return 1
    fi
  done

  return 0
}

for i in "${!LANE_ID[@]}"; do
  if lane_matches_selected_filters "${LANE_ID[$i]}"; then
    SELECTED_INDICES+=("$i")
  fi
done

if [[ "${#SELECTED_INDICES[@]}" -eq 0 ]]; then
  echo "No lanes selected after applying include/exclude regex filters." >&2
  exit 1
fi

lane_cache_schedule_key() {
  local i="$1"
  local key_payload=""
  local lane_generate_count="${GENERATE_COUNT[$i]}"
  local lane_mutations_file="${MUTATIONS_FILE[$i]}"
  local lane_mutations_top="${MUTATIONS_TOP[$i]}"
  local lane_mutations_seed="${MUTATIONS_SEED[$i]}"
  local lane_mutations_yosys="${MUTATIONS_YOSYS[$i]}"
  local lane_mutations_modes="${MUTATIONS_MODES[$i]}"
  local lane_mutations_mode_counts="${MUTATIONS_MODE_COUNTS[$i]}"
  local lane_mutations_mode_weights="${MUTATIONS_MODE_WEIGHTS[$i]}"
  local lane_mutations_profiles="${MUTATIONS_PROFILES[$i]}"
  local lane_mutations_cfg="${MUTATIONS_CFG[$i]}"
  local lane_mutations_select="${MUTATIONS_SELECT[$i]}"

  if [[ -z "$REUSE_CACHE_DIR" ]]; then
    printf "lane:%s\n" "${LANE_ID[$i]}"
    return
  fi
  if [[ "$lane_mutations_file" != "-" ]]; then
    printf "lane:%s\n" "${LANE_ID[$i]}"
    return
  fi
  if [[ "$lane_generate_count" == "-" || -z "$lane_generate_count" ]]; then
    printf "lane:%s\n" "${LANE_ID[$i]}"
    return
  fi

  if [[ "$lane_mutations_top" == "-" ]]; then
    lane_mutations_top=""
  fi
  if [[ "$lane_mutations_seed" == "-" || -z "$lane_mutations_seed" ]]; then
    lane_mutations_seed="$DEFAULT_MUTATIONS_SEED"
  fi
  if [[ "$lane_mutations_seed" == "-" || -z "$lane_mutations_seed" ]]; then
    lane_mutations_seed="1"
  fi
  if [[ "$lane_mutations_yosys" == "-" || -z "$lane_mutations_yosys" ]]; then
    lane_mutations_yosys="$DEFAULT_MUTATIONS_YOSYS"
  fi
  if [[ -z "$lane_mutations_yosys" || "$lane_mutations_yosys" == "-" ]]; then
    lane_mutations_yosys="yosys"
  fi
  if [[ "$lane_mutations_modes" == "-" || -z "$lane_mutations_modes" ]]; then
    lane_mutations_modes="$DEFAULT_MUTATIONS_MODES"
  fi
  if [[ "$lane_mutations_mode_counts" == "-" || -z "$lane_mutations_mode_counts" ]]; then
    lane_mutations_mode_counts="$DEFAULT_MUTATIONS_MODE_COUNTS"
  fi
  if [[ "$lane_mutations_mode_weights" == "-" || -z "$lane_mutations_mode_weights" ]]; then
    lane_mutations_mode_weights="$DEFAULT_MUTATIONS_MODE_WEIGHTS"
  fi
  if [[ "$lane_mutations_profiles" == "-" || -z "$lane_mutations_profiles" ]]; then
    lane_mutations_profiles="$DEFAULT_MUTATIONS_PROFILES"
  fi
  if [[ "$lane_mutations_cfg" == "-" || -z "$lane_mutations_cfg" ]]; then
    lane_mutations_cfg="$DEFAULT_MUTATIONS_CFG"
  fi
  if [[ "$lane_mutations_select" == "-" || -z "$lane_mutations_select" ]]; then
    lane_mutations_select="$DEFAULT_MUTATIONS_SELECT"
  fi

  key_payload="$(
    cat <<EOF
v1
design=${DESIGN[$i]}
count=$lane_generate_count
top=$lane_mutations_top
seed=$lane_mutations_seed
yosys=$lane_mutations_yosys
modes=$lane_mutations_modes
mode_counts=$lane_mutations_mode_counts
mode_weights=$lane_mutations_mode_weights
profiles=$lane_mutations_profiles
cfg=$lane_mutations_cfg
select=$lane_mutations_select
EOF
  )"
  printf "cache:%s\n" "$(hash_string "$key_payload")"
}

schedule_lanes() {
  local i=""
  local key=""
  local unique_keys=0
  local followers=0
  declare -A seen=()
  declare -a leaders=()
  declare -a deferred=()

  if [[ "$LANE_SCHEDULE_POLICY" != "cache-aware" ]]; then
    SCHEDULED_INDICES=("${SELECTED_INDICES[@]}")
    echo "Lane scheduling policy: ${LANE_SCHEDULE_POLICY} (selected=${#SELECTED_INDICES[@]})"
    return
  fi

  for i in "${SELECTED_INDICES[@]}"; do
    key="$(lane_cache_schedule_key "$i")"
    if [[ -z "${seen[$key]+x}" ]]; then
      seen["$key"]=1
      leaders+=("$i")
      unique_keys=$((unique_keys + 1))
    else
      deferred+=("$i")
      followers=$((followers + 1))
    fi
  done

  SCHEDULED_INDICES=("${leaders[@]}" "${deferred[@]}")
  echo "Lane scheduling policy: ${LANE_SCHEDULE_POLICY} (selected=${#SELECTED_INDICES[@]} unique_keys=${unique_keys} deferred_followers=${followers})"
}

schedule_lanes

resolve_bool_override() {
  local lane_value="$1"
  local default_value="$2"

  if [[ "$lane_value" == "-" || -z "$lane_value" ]]; then
    if [[ "$default_value" -eq 1 ]]; then
      printf "1\n"
    else
      printf "0\n"
    fi
    return
  fi
  case "$lane_value" in
    1|true|yes) printf "1\n" ;;
    0|false|no) printf "0\n" ;;
    *) printf "invalid\n" ;;
  esac
}

prequalify_metric_from_log() {
  local log_file="$1"
  local metric_key="$2"
  local default_value="$3"
  local metric_value=""

  metric_value="$(awk -F$'\t' -v k="$metric_key" '$1==k{print $2; exit}' "$log_file" | tr -d '\r')"
  if [[ "$metric_value" =~ ^[0-9]+$ ]]; then
    printf "%s\n" "$metric_value"
  else
    printf "%s\n" "$default_value"
  fi
}

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
  local lane_mutations_mode_weights=""
  local lane_mutations_profiles=""
  local lane_mutations_cfg=""
  local lane_mutations_select=""
  local lane_mutations_seed=""
  local lane_global_propagate_cmd=""
  local lane_global_propagate_timeout_seconds=""
  local lane_global_propagate_lec_timeout_seconds=""
  local lane_global_propagate_bmc_timeout_seconds=""
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
  local lane_bmc_orig_cache_max_age_seconds=""
  local lane_bmc_orig_cache_eviction_policy=""
  local lane_skip_baseline=""
  local lane_fail_on_undetected=""
  local lane_fail_on_errors=""
  local lane_generated_mutations_cache_status="disabled"
  local lane_generated_mutations_cache_hit="0"
  local lane_generated_mutations_cache_miss="0"
  local lane_generated_mutations_cache_saved_runtime_ns="0"
  local lane_generated_mutations_cache_lock_wait_ns="0"
  local lane_generated_mutations_cache_lock_contended="0"
  local config_error_code="-"
  local config_error_reason="-"
  local lane_prequalify_summary_present="0"
  local lane_prequalify_total_mutants="-"
  local lane_prequalify_not_propagated_mutants="0"
  local lane_prequalify_propagated_mutants="0"
  local lane_prequalify_create_mutated_error_mutants="0"
  local lane_prequalify_probe_error_mutants="0"
  local lane_prequalify_cmd_token_not_propagated_mutants="0"
  local lane_prequalify_cmd_token_propagated_mutants="0"
  local lane_prequalify_cmd_rc_not_propagated_mutants="0"
  local lane_prequalify_cmd_rc_propagated_mutants="0"
  local lane_prequalify_cmd_timeout_propagated_mutants="0"
  local lane_prequalify_cmd_error_mutants="0"
  local lane_mode_counts_enabled=0
  local lane_mode_weights_enabled=0
  local lane_mode_counts_total=0
  local lane_contract_payload=""
  local lane_mutation_source_payload=""
  local lane_contract_fingerprint="-"
  local lane_mutation_source_fingerprint="-"
  local lane_mutations_file_hash="missing"
  local -a cmd=()

  lane_write_status() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$lane_id" "$lane_status" "$rc" "$coverage" "$gate" "$lane_dir" "$lane_metrics" "$lane_json" \
      "$lane_generated_mutations_cache_status" "$lane_generated_mutations_cache_hit" \
      "$lane_generated_mutations_cache_miss" "$lane_generated_mutations_cache_saved_runtime_ns" \
      "$lane_generated_mutations_cache_lock_wait_ns" "$lane_generated_mutations_cache_lock_contended" \
      "$config_error_code" "$config_error_reason" "$lane_prequalify_summary_present" \
      "$lane_prequalify_total_mutants" "$lane_prequalify_not_propagated_mutants" \
      "$lane_prequalify_propagated_mutants" \
      "$lane_prequalify_create_mutated_error_mutants" \
      "$lane_prequalify_probe_error_mutants" \
      "$lane_prequalify_cmd_token_not_propagated_mutants" \
      "$lane_prequalify_cmd_token_propagated_mutants" \
      "$lane_prequalify_cmd_rc_not_propagated_mutants" \
      "$lane_prequalify_cmd_rc_propagated_mutants" \
      "$lane_prequalify_cmd_timeout_propagated_mutants" \
      "$lane_prequalify_cmd_error_mutants" "$lane_contract_fingerprint" \
      "$lane_mutation_source_fingerprint" > "$lane_status_file"
  }

  lane_config_error() {
    local code="$1"
    local reason="$2"
    gate="CONFIG_ERROR"
    lane_status="FAIL"
    rc=1
    if [[ -n "$code" ]]; then
      config_error_code="$code"
    else
      config_error_code="-"
    fi
    if [[ -n "$reason" ]]; then
      config_error_reason="$reason"
    else
      config_error_reason="-"
    fi
    lane_write_status
  }

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
  lane_skip_baseline="$(resolve_bool_override "${LANE_SKIP_BASELINE[$i]}" "$SKIP_BASELINE")"
  if [[ "$lane_skip_baseline" == "invalid" ]]; then
    lane_config_error "INVALID_SKIP_BASELINE_OVERRIDE" "Invalid lane skip_baseline override (expected 1|0|true|false|yes|no|-)."
    return 0
  fi
  lane_fail_on_undetected="$(resolve_bool_override "${LANE_FAIL_ON_UNDETECTED[$i]}" "$FAIL_ON_UNDETECTED")"
  if [[ "$lane_fail_on_undetected" == "invalid" ]]; then
    lane_config_error "INVALID_FAIL_ON_UNDETECTED_OVERRIDE" "Invalid lane fail_on_undetected override (expected 1|0|true|false|yes|no|-)."
    return 0
  fi
  lane_fail_on_errors="$(resolve_bool_override "${LANE_FAIL_ON_ERRORS[$i]}" "$FAIL_ON_ERRORS")"
  if [[ "$lane_fail_on_errors" == "invalid" ]]; then
    lane_config_error "INVALID_FAIL_ON_ERRORS_OVERRIDE" "Invalid lane fail_on_errors override (expected 1|0|true|false|yes|no|-)."
    return 0
  fi

  if [[ "$lane_skip_baseline" -eq 1 ]]; then
    cmd+=(--skip-baseline)
  fi
  if [[ "$lane_fail_on_undetected" -eq 1 ]]; then
    cmd+=(--fail-on-undetected)
  fi
  if [[ "$lane_fail_on_errors" -eq 1 ]]; then
    cmd+=(--fail-on-errors)
  fi

  if [[ "${MUTATIONS_FILE[$i]}" != "-" ]]; then
    cmd+=(--mutations-file "${MUTATIONS_FILE[$i]}")
  elif [[ "${GENERATE_COUNT[$i]}" != "-" && -n "${GENERATE_COUNT[$i]}" ]]; then
    cmd+=(--generate-mutations "${GENERATE_COUNT[$i]}")
  else
    lane_config_error "MISSING_MUTATION_SOURCE" "Lane is missing mutation input (provide mutations_file or generate_count)."
    return 0
  fi

  lane_reuse_pair_file="${REUSE_PAIR_FILE[$i]}"
  if [[ "$lane_reuse_pair_file" == "-" || -z "$lane_reuse_pair_file" ]]; then
    lane_reuse_pair_file="$DEFAULT_REUSE_PAIR_FILE"
  fi
  if [[ -n "$lane_reuse_pair_file" ]]; then
    if [[ ! -f "$lane_reuse_pair_file" ]]; then
      lane_config_error "MISSING_REUSE_PAIR_FILE" "Lane reuse_pair_file not found: $lane_reuse_pair_file"
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
      lane_config_error "MISSING_REUSE_SUMMARY_FILE" "Lane reuse_summary_file not found: $lane_reuse_summary_file"
      return 0
    fi
    cmd+=(--reuse-summary-file "$lane_reuse_summary_file")
  fi

  if [[ "${GENERATE_COUNT[$i]}" != "-" && -n "${GENERATE_COUNT[$i]}" ]]; then
    if ! [[ "${GENERATE_COUNT[$i]}" =~ ^[1-9][0-9]*$ ]]; then
      lane_config_error "INVALID_GENERATE_COUNT" "Invalid lane generate_count value: ${GENERATE_COUNT[$i]} (expected positive integer)."
      return 0
    fi
    lane_mutations_modes="${MUTATIONS_MODES[$i]}"
    if [[ "$lane_mutations_modes" == "-" || -z "$lane_mutations_modes" ]]; then
      lane_mutations_modes="$DEFAULT_MUTATIONS_MODES"
    fi
    lane_mutations_mode_counts="${MUTATIONS_MODE_COUNTS[$i]}"
    if [[ "$lane_mutations_mode_counts" == "-" || -z "$lane_mutations_mode_counts" ]]; then
      lane_mutations_mode_counts="$DEFAULT_MUTATIONS_MODE_COUNTS"
    fi
    lane_mutations_mode_weights="${MUTATIONS_MODE_WEIGHTS[$i]}"
    if [[ "$lane_mutations_mode_weights" == "-" || -z "$lane_mutations_mode_weights" ]]; then
      lane_mutations_mode_weights="$DEFAULT_MUTATIONS_MODE_WEIGHTS"
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
    if ! validate_mode_list_csv "lane mutations_modes" "$lane_mutations_modes"; then
      lane_config_error "INVALID_MUTATIONS_MODES" "$VALIDATION_ERROR"
      return 0
    fi
    if ! validate_profile_list_csv "lane mutations_profiles" "$lane_mutations_profiles"; then
      lane_config_error "INVALID_MUTATIONS_PROFILES" "$VALIDATION_ERROR"
      return 0
    fi
    if ! parse_mode_allocation_csv "lane mutations_mode_counts" "$lane_mutations_mode_counts" "COUNT"; then
      lane_config_error "INVALID_MUTATIONS_MODE_COUNTS" "$VALIDATION_ERROR"
      return 0
    fi
    lane_mode_counts_enabled="$PARSED_ALLOCATION_ENABLED"
    lane_mode_counts_total="$PARSED_ALLOCATION_TOTAL"
    if ! parse_mode_allocation_csv "lane mutations_mode_weights" "$lane_mutations_mode_weights" "WEIGHT"; then
      lane_config_error "INVALID_MUTATIONS_MODE_WEIGHTS" "$VALIDATION_ERROR"
      return 0
    fi
    lane_mode_weights_enabled="$PARSED_ALLOCATION_ENABLED"
    if [[ "$lane_mode_counts_enabled" -eq 1 && "$lane_mode_weights_enabled" -eq 1 ]]; then
      lane_config_error "CONFLICT_MUTATIONS_MODE_ALLOCATION" "Lane uses both mutations_mode_counts and mutations_mode_weights; choose one."
      return 0
    fi
    if [[ "$lane_mode_counts_enabled" -eq 1 && "$lane_mode_counts_total" -ne "${GENERATE_COUNT[$i]}" ]]; then
      lane_config_error "MISMATCH_MUTATIONS_MODE_COUNTS_TOTAL" "Lane mutations_mode_counts total (${lane_mode_counts_total}) must equal generate_count (${GENERATE_COUNT[$i]})."
      return 0
    fi
    if [[ "${MUTATIONS_TOP[$i]}" != "-" && -n "${MUTATIONS_TOP[$i]}" ]]; then
      cmd+=(--mutations-top "${MUTATIONS_TOP[$i]}")
    fi
    lane_mutations_seed="${MUTATIONS_SEED[$i]}"
    if [[ "$lane_mutations_seed" == "-" || -z "$lane_mutations_seed" ]]; then
      lane_mutations_seed="$DEFAULT_MUTATIONS_SEED"
    fi
    if [[ -n "$lane_mutations_seed" ]]; then
      if ! [[ "$lane_mutations_seed" =~ ^[0-9]+$ ]]; then
        lane_config_error "INVALID_MUTATIONS_SEED" "Invalid lane mutations_seed value: $lane_mutations_seed"
        return 0
      fi
      cmd+=(--mutations-seed "$lane_mutations_seed")
    fi
    lane_mutations_yosys="${MUTATIONS_YOSYS[$i]}"
    if [[ "$lane_mutations_yosys" == "-" || -z "$lane_mutations_yosys" ]]; then
      lane_mutations_yosys="$DEFAULT_MUTATIONS_YOSYS"
    fi
    if [[ -n "$lane_mutations_yosys" && "$lane_mutations_yosys" != "-" ]]; then
      cmd+=(--mutations-yosys "$lane_mutations_yosys")
    fi
    if [[ -n "$lane_mutations_modes" ]]; then
      cmd+=(--mutations-modes "$lane_mutations_modes")
    fi
    if [[ -n "$lane_mutations_mode_counts" ]]; then
      cmd+=(--mutations-mode-counts "$lane_mutations_mode_counts")
    fi
    if [[ -n "$lane_mutations_mode_weights" ]]; then
      cmd+=(--mutations-mode-weights "$lane_mutations_mode_weights")
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
  lane_global_propagate_timeout_seconds="${GLOBAL_PROPAGATE_TIMEOUT_SECONDS[$i]}"
  if [[ "$lane_global_propagate_timeout_seconds" == "-" || -z "$lane_global_propagate_timeout_seconds" ]]; then
    lane_global_propagate_timeout_seconds="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_TIMEOUT_SECONDS"
  fi
  if [[ -n "$lane_global_propagate_timeout_seconds" ]]; then
    if ! [[ "$lane_global_propagate_timeout_seconds" =~ ^[0-9]+$ ]]; then
      lane_config_error "INVALID_GLOBAL_PROPAGATE_TIMEOUT" "Invalid lane global_propagate_timeout_seconds value: $lane_global_propagate_timeout_seconds"
      return 0
    fi
    cmd+=(--formal-global-propagate-timeout-seconds "$lane_global_propagate_timeout_seconds")
  fi
  lane_global_propagate_lec_timeout_seconds="${GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS[$i]}"
  if [[ "$lane_global_propagate_lec_timeout_seconds" == "-" || -z "$lane_global_propagate_lec_timeout_seconds" ]]; then
    lane_global_propagate_lec_timeout_seconds="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_LEC_TIMEOUT_SECONDS"
  fi
  if [[ -n "$lane_global_propagate_lec_timeout_seconds" ]]; then
    if ! [[ "$lane_global_propagate_lec_timeout_seconds" =~ ^[0-9]+$ ]]; then
      lane_config_error "INVALID_GLOBAL_PROPAGATE_LEC_TIMEOUT" "Invalid lane global_propagate_lec_timeout_seconds value: $lane_global_propagate_lec_timeout_seconds"
      return 0
    fi
    cmd+=(--formal-global-propagate-lec-timeout-seconds "$lane_global_propagate_lec_timeout_seconds")
  fi
  lane_global_propagate_bmc_timeout_seconds="${GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS[$i]}"
  if [[ "$lane_global_propagate_bmc_timeout_seconds" == "-" || -z "$lane_global_propagate_bmc_timeout_seconds" ]]; then
    lane_global_propagate_bmc_timeout_seconds="$DEFAULT_FORMAL_GLOBAL_PROPAGATE_BMC_TIMEOUT_SECONDS"
  fi
  if [[ -n "$lane_global_propagate_bmc_timeout_seconds" ]]; then
    if ! [[ "$lane_global_propagate_bmc_timeout_seconds" =~ ^[0-9]+$ ]]; then
      lane_config_error "INVALID_GLOBAL_PROPAGATE_BMC_TIMEOUT" "Invalid lane global_propagate_bmc_timeout_seconds value: $lane_global_propagate_bmc_timeout_seconds"
      return 0
    fi
    cmd+=(--formal-global-propagate-bmc-timeout-seconds "$lane_global_propagate_bmc_timeout_seconds")
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
      lane_config_error "INVALID_GLOBAL_PROPAGATE_BMC_BOUND" "Invalid lane global_propagate_bmc_bound value: $lane_global_propagate_bmc_bound"
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
      lane_config_error "INVALID_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" "Invalid lane global_propagate_bmc_ignore_asserts_until value: $lane_global_propagate_bmc_ignore_asserts_until"
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
      lane_config_error "INVALID_BMC_ORIG_CACHE_MAX_ENTRIES" "Invalid lane bmc_orig_cache_max_entries value: $lane_bmc_orig_cache_max_entries"
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
      lane_config_error "INVALID_BMC_ORIG_CACHE_MAX_BYTES" "Invalid lane bmc_orig_cache_max_bytes value: $lane_bmc_orig_cache_max_bytes"
      return 0
    fi
    cmd+=(--bmc-orig-cache-max-bytes "$lane_bmc_orig_cache_max_bytes")
  fi

  lane_bmc_orig_cache_max_age_seconds="${BMC_ORIG_CACHE_MAX_AGE_SECONDS[$i]}"
  if [[ "$lane_bmc_orig_cache_max_age_seconds" == "-" || -z "$lane_bmc_orig_cache_max_age_seconds" ]]; then
    lane_bmc_orig_cache_max_age_seconds="$DEFAULT_BMC_ORIG_CACHE_MAX_AGE_SECONDS"
  fi
  if [[ -n "$lane_bmc_orig_cache_max_age_seconds" ]]; then
    if ! [[ "$lane_bmc_orig_cache_max_age_seconds" =~ ^[0-9]+$ ]]; then
      lane_config_error "INVALID_BMC_ORIG_CACHE_MAX_AGE_SECONDS" "Invalid lane bmc_orig_cache_max_age_seconds value: $lane_bmc_orig_cache_max_age_seconds"
      return 0
    fi
    cmd+=(--bmc-orig-cache-max-age-seconds "$lane_bmc_orig_cache_max_age_seconds")
  fi

  lane_bmc_orig_cache_eviction_policy="${BMC_ORIG_CACHE_EVICTION_POLICY[$i]}"
  if [[ "$lane_bmc_orig_cache_eviction_policy" == "-" || -z "$lane_bmc_orig_cache_eviction_policy" ]]; then
    lane_bmc_orig_cache_eviction_policy="$DEFAULT_BMC_ORIG_CACHE_EVICTION_POLICY"
  fi
  if [[ -n "$lane_bmc_orig_cache_eviction_policy" ]]; then
    if ! [[ "$lane_bmc_orig_cache_eviction_policy" =~ ^(lru|fifo|cost-lru)$ ]]; then
      lane_config_error "INVALID_BMC_ORIG_CACHE_EVICTION_POLICY" "Invalid lane bmc_orig_cache_eviction_policy value: $lane_bmc_orig_cache_eviction_policy"
      return 0
    fi
    cmd+=(--bmc-orig-cache-eviction-policy "$lane_bmc_orig_cache_eviction_policy")
  fi

  if [[ -n "${THRESHOLD[$i]}" && "${THRESHOLD[$i]}" != "-" ]]; then
    cmd+=(--coverage-threshold "${THRESHOLD[$i]}")
  fi

  if [[ "${MUTATIONS_FILE[$i]}" != "-" ]]; then
    if [[ -f "${MUTATIONS_FILE[$i]}" ]]; then
      lane_mutations_file_hash="$(hash_file "${MUTATIONS_FILE[$i]}")"
    fi
    lane_mutation_source_payload="$({
      cat <<EOF
v1
kind=file
mutations_file=${MUTATIONS_FILE[$i]}
mutations_file_hash=${lane_mutations_file_hash}
EOF
    })"
  else
    lane_mutation_source_payload="$({
      cat <<EOF
v1
kind=generated
design=${DESIGN[$i]}
generate_count=${GENERATE_COUNT[$i]}
mutations_top=${MUTATIONS_TOP[$i]}
mutations_seed=${lane_mutations_seed}
mutations_yosys=${lane_mutations_yosys}
mutations_modes=${lane_mutations_modes}
mutations_mode_counts=${lane_mutations_mode_counts}
mutations_mode_weights=${lane_mutations_mode_weights}
mutations_profiles=${lane_mutations_profiles}
mutations_cfg=${lane_mutations_cfg}
mutations_select=${lane_mutations_select}
EOF
    })"
  fi
  lane_mutation_source_fingerprint="$(hash_string "$lane_mutation_source_payload")"

  lane_contract_payload="$({
    cat <<EOF
v1
design=${DESIGN[$i]}
tests_manifest=${TESTS_MANIFEST[$i]}
jobs_per_lane=${JOBS_PER_LANE}
skip_baseline=${lane_skip_baseline}
fail_on_undetected=${lane_fail_on_undetected}
fail_on_errors=${lane_fail_on_errors}
create_mutated_script=${CREATE_MUTATED_SCRIPT}
activate_cmd=${ACTIVATE_CMD[$i]}
propagate_cmd=${PROPAGATE_CMD[$i]}
threshold=${THRESHOLD[$i]}
reuse_cache_dir=${REUSE_CACHE_DIR}
reuse_compat_mode=${REUSE_COMPAT_MODE}
reuse_pair_file=${lane_reuse_pair_file}
reuse_summary_file=${lane_reuse_summary_file}
global_propagate_cmd=${lane_global_propagate_cmd}
global_propagate_timeout_seconds=${lane_global_propagate_timeout_seconds}
global_propagate_lec_timeout_seconds=${lane_global_propagate_lec_timeout_seconds}
global_propagate_bmc_timeout_seconds=${lane_global_propagate_bmc_timeout_seconds}
global_propagate_circt_lec=${lane_global_propagate_circt_lec}
global_propagate_circt_lec_args=${lane_global_propagate_circt_lec_args}
global_propagate_c1=${lane_global_propagate_c1}
global_propagate_c2=${lane_global_propagate_c2}
global_propagate_z3=${lane_global_propagate_z3}
global_propagate_assume_known_inputs=${lane_global_propagate_assume_known_inputs}
global_propagate_accept_xprop_only=${lane_global_propagate_accept_xprop_only}
global_propagate_circt_bmc=${lane_global_propagate_circt_bmc}
global_propagate_circt_chain=${lane_global_propagate_circt_chain}
global_propagate_bmc_args=${lane_global_propagate_bmc_args}
global_propagate_bmc_bound=${lane_global_propagate_bmc_bound}
global_propagate_bmc_module=${lane_global_propagate_bmc_module}
global_propagate_bmc_run_smtlib=${lane_global_propagate_bmc_run_smtlib}
global_propagate_bmc_z3=${lane_global_propagate_bmc_z3}
global_propagate_bmc_assume_known_inputs=${lane_global_propagate_bmc_assume_known_inputs}
global_propagate_bmc_ignore_asserts_until=${lane_global_propagate_bmc_ignore_asserts_until}
bmc_orig_cache_max_entries=${lane_bmc_orig_cache_max_entries}
bmc_orig_cache_max_bytes=${lane_bmc_orig_cache_max_bytes}
bmc_orig_cache_max_age_seconds=${lane_bmc_orig_cache_max_age_seconds}
bmc_orig_cache_eviction_policy=${lane_bmc_orig_cache_eviction_policy}
mutation_source_fingerprint=${lane_mutation_source_fingerprint}
EOF
  })"
  lane_contract_fingerprint="$(hash_string "$lane_contract_payload")"

  rc=0
  set +e
  "${cmd[@]}" >"$lane_log" 2>&1
  rc=$?
  set -e

  if [[ -f "$lane_metrics" ]]; then
    cov_v="$(awk -F$'\t' '$1=="mutation_coverage_percent"{print $2}' "$lane_metrics" | head -n1)"
    [[ -n "$cov_v" ]] && coverage="$cov_v"
    gate_v="$(awk -F$'\t' '$1=="gate_status"{print $2}' "$lane_metrics" | head -n1)"
    [[ -n "$gate_v" ]] && gate="$gate_v"
    lane_generated_mutations_cache_status="$(awk -F$'\t' '$1=="generated_mutations_cache_status"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_status="${lane_generated_mutations_cache_status:-disabled}"
    lane_generated_mutations_cache_hit="$(awk -F$'\t' '$1=="generated_mutations_cache_hit"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_hit="${lane_generated_mutations_cache_hit:-0}"
    if [[ ! "$lane_generated_mutations_cache_hit" =~ ^[0-9]+$ ]]; then
      lane_generated_mutations_cache_hit=0
    fi
    lane_generated_mutations_cache_miss="$(awk -F$'\t' '$1=="generated_mutations_cache_miss"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_miss="${lane_generated_mutations_cache_miss:-0}"
    if [[ ! "$lane_generated_mutations_cache_miss" =~ ^[0-9]+$ ]]; then
      lane_generated_mutations_cache_miss=0
    fi
    lane_generated_mutations_cache_saved_runtime_ns="$(awk -F$'\t' '$1=="generated_mutations_cache_saved_runtime_ns"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_saved_runtime_ns="${lane_generated_mutations_cache_saved_runtime_ns:-0}"
    if [[ ! "$lane_generated_mutations_cache_saved_runtime_ns" =~ ^[0-9]+$ ]]; then
      lane_generated_mutations_cache_saved_runtime_ns=0
    fi
    lane_generated_mutations_cache_lock_wait_ns="$(awk -F$'\t' '$1=="generated_mutations_cache_lock_wait_ns"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_lock_wait_ns="${lane_generated_mutations_cache_lock_wait_ns:-0}"
    if [[ ! "$lane_generated_mutations_cache_lock_wait_ns" =~ ^[0-9]+$ ]]; then
      lane_generated_mutations_cache_lock_wait_ns=0
    fi
    lane_generated_mutations_cache_lock_contended="$(awk -F$'\t' '$1=="generated_mutations_cache_lock_contended"{print $2}' "$lane_metrics" | head -n1)"
    lane_generated_mutations_cache_lock_contended="${lane_generated_mutations_cache_lock_contended:-0}"
    if [[ ! "$lane_generated_mutations_cache_lock_contended" =~ ^[0-9]+$ ]]; then
      lane_generated_mutations_cache_lock_contended=0
    fi
  fi
  if [[ -f "$lane_log" ]]; then
    gate_v="$(awk -F': ' '/^Gate status:/{print $2}' "$lane_log" | tail -n1)"
    [[ -n "$gate_v" ]] && gate="$gate_v"
  fi
  if [[ "$rc" -eq 0 ]]; then
    lane_status="PASS"
  fi

  lane_prequalify_log="${lane_dir}/native_global_filter_prequalify.log"
  if [[ -f "$lane_prequalify_log" ]]; then
    lane_prequalify_total_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_total_mutants" "-")"
    if [[ "$lane_prequalify_total_mutants" =~ ^[0-9]+$ ]]; then
      lane_prequalify_summary_present="1"
    fi
    lane_prequalify_not_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_not_propagated_mutants" "0")"
    lane_prequalify_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_propagated_mutants" "0")"
    lane_prequalify_create_mutated_error_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_create_mutated_error_mutants" "0")"
    lane_prequalify_probe_error_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_probe_error_mutants" "0")"
    lane_prequalify_cmd_token_not_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_token_not_propagated_mutants" "0")"
    lane_prequalify_cmd_token_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_token_propagated_mutants" "0")"
    lane_prequalify_cmd_rc_not_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_rc_not_propagated_mutants" "0")"
    lane_prequalify_cmd_rc_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_rc_propagated_mutants" "0")"
    lane_prequalify_cmd_timeout_propagated_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_timeout_propagated_mutants" "0")"
    lane_prequalify_cmd_error_mutants="$(prequalify_metric_from_log "$lane_prequalify_log" "prequalify_cmd_error_mutants" "0")"
  fi

  lane_write_status
}

if [[ "$LANE_JOBS" -le 1 ]]; then
  for i in "${SCHEDULED_INDICES[@]}"; do
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
  for i in "${SCHEDULED_INDICES[@]}"; do
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

printf "lane_id\tstatus\texit_code\tcoverage_percent\tgate_status\tlane_dir\tmetrics_file\tsummary_json\tgenerated_mutations_cache_status\tgenerated_mutations_cache_hit\tgenerated_mutations_cache_miss\tgenerated_mutations_cache_saved_runtime_ns\tgenerated_mutations_cache_lock_wait_ns\tgenerated_mutations_cache_lock_contended\tconfig_error_code\tconfig_error_reason\tprequalify_summary_present\tprequalify_total_mutants\tprequalify_not_propagated_mutants\tprequalify_propagated_mutants\tprequalify_create_mutated_error_mutants\tprequalify_probe_error_mutants\tprequalify_cmd_token_not_propagated_mutants\tprequalify_cmd_token_propagated_mutants\tprequalify_cmd_rc_not_propagated_mutants\tprequalify_cmd_rc_propagated_mutants\tprequalify_cmd_timeout_propagated_mutants\tprequalify_cmd_error_mutants\tlane_contract_fingerprint\tlane_mutation_source_fingerprint\n" > "$RESULTS_FILE"
failures="$parse_failures"
passes=0
generated_cache_hit_lanes=0
generated_cache_miss_lanes=0
generated_cache_saved_runtime_ns=0
generated_cache_lock_wait_ns=0
generated_cache_lock_contended_lanes=0
declare -A GATE_COUNTS=()
declare -A CONTRACT_FINGERPRINT_COUNTS=()
declare -A MUTATION_SOURCE_FINGERPRINT_COUNTS=()
if [[ "$parse_failures" -gt 0 ]]; then
  GATE_COUNTS["PARSE_ERROR"]="$parse_failures"
fi

for i in "${EXECUTED_INDICES[@]}"; do
  lane_status_file="${OUT_DIR}/${LANE_ID[$i]}/lane_status.tsv"
  if [[ ! -f "$lane_status_file" ]]; then
    failures=$((failures + 1))
    printf "%s\tFAIL\t1\t0.00\tMISSING_STATUS\t%s\t%s\t%s\tdisabled\t0\t0\t0\t0\t0\t-\t-\t0\t-\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t-\t-\n" \
      "${LANE_ID[$i]}" "${OUT_DIR}/${LANE_ID[$i]}" \
      "${OUT_DIR}/${LANE_ID[$i]}/metrics.tsv" "${OUT_DIR}/${LANE_ID[$i]}/summary.json" >> "$RESULTS_FILE"
    GATE_COUNTS["MISSING_STATUS"]=$(( ${GATE_COUNTS["MISSING_STATUS"]:-0} + 1 ))
    continue
  fi
  cat "$lane_status_file" >> "$RESULTS_FILE"
  lane_status="$(awk -F$'\t' 'NR==1{print $2}' "$lane_status_file")"
  lane_gate="$(awk -F$'\t' 'NR==1{print $5}' "$lane_status_file")"
  lane_gate="${lane_gate:-UNKNOWN}"
  GATE_COUNTS["$lane_gate"]=$(( ${GATE_COUNTS["$lane_gate"]:-0} + 1 ))
  if [[ "$lane_status" == "PASS" ]]; then
    passes=$((passes + 1))
  else
    failures=$((failures + 1))
  fi
  lane_cache_hit="$(awk -F$'\t' 'NR==1{print $10}' "$lane_status_file")"
  lane_cache_miss="$(awk -F$'\t' 'NR==1{print $11}' "$lane_status_file")"
  lane_cache_saved_runtime_ns="$(awk -F$'\t' 'NR==1{print $12}' "$lane_status_file")"
  lane_cache_lock_wait_ns="$(awk -F$'\t' 'NR==1{print $13}' "$lane_status_file")"
  lane_cache_lock_contended="$(awk -F$'\t' 'NR==1{print $14}' "$lane_status_file")"
  lane_cache_hit="${lane_cache_hit:-0}"
  lane_cache_miss="${lane_cache_miss:-0}"
  lane_cache_saved_runtime_ns="${lane_cache_saved_runtime_ns:-0}"
  lane_cache_lock_wait_ns="${lane_cache_lock_wait_ns:-0}"
  lane_cache_lock_contended="${lane_cache_lock_contended:-0}"
  if [[ "$lane_cache_hit" =~ ^[0-9]+$ ]]; then
    generated_cache_hit_lanes=$((generated_cache_hit_lanes + lane_cache_hit))
  fi
  if [[ "$lane_cache_miss" =~ ^[0-9]+$ ]]; then
    generated_cache_miss_lanes=$((generated_cache_miss_lanes + lane_cache_miss))
  fi
  if [[ "$lane_cache_saved_runtime_ns" =~ ^[0-9]+$ ]]; then
    generated_cache_saved_runtime_ns=$((generated_cache_saved_runtime_ns + lane_cache_saved_runtime_ns))
  fi
  if [[ "$lane_cache_lock_wait_ns" =~ ^[0-9]+$ ]]; then
    generated_cache_lock_wait_ns=$((generated_cache_lock_wait_ns + lane_cache_lock_wait_ns))
  fi
  if [[ "$lane_cache_lock_contended" =~ ^[0-9]+$ ]]; then
    generated_cache_lock_contended_lanes=$((generated_cache_lock_contended_lanes + lane_cache_lock_contended))
  fi
  lane_contract_fingerprint="$(awk -F$'\t' 'NR==1{print $(NF-1)}' "$lane_status_file")"
  lane_mutation_source_fingerprint="$(awk -F$'\t' 'NR==1{print $NF}' "$lane_status_file")"
  if [[ -n "$lane_contract_fingerprint" && "$lane_contract_fingerprint" != "-" ]]; then
    CONTRACT_FINGERPRINT_COUNTS["$lane_contract_fingerprint"]=$(( ${CONTRACT_FINGERPRINT_COUNTS["$lane_contract_fingerprint"]:-0} + 1 ))
  fi
  if [[ -n "$lane_mutation_source_fingerprint" && "$lane_mutation_source_fingerprint" != "-" ]]; then
    MUTATION_SOURCE_FINGERPRINT_COUNTS["$lane_mutation_source_fingerprint"]=$(( ${MUTATION_SOURCE_FINGERPRINT_COUNTS["$lane_mutation_source_fingerprint"]:-0} + 1 ))
  fi
done

{
  printf "gate_status\tcount\n"
  if [[ "${#GATE_COUNTS[@]}" -gt 0 ]]; then
    for gate_name in $(printf "%s\n" "${!GATE_COUNTS[@]}" | sort); do
      printf "%s\t%s\n" "$gate_name" "${GATE_COUNTS[$gate_name]}"
    done
  fi
} > "$GATE_SUMMARY_FILE"

contract_fingerprints_digest="-"
mutation_source_fingerprints_digest="-"
if [[ "${#CONTRACT_FINGERPRINT_COUNTS[@]}" -gt 0 ]]; then
  contract_fingerprints_digest="$(
    for fingerprint in $(printf "%s\n" "${!CONTRACT_FINGERPRINT_COUNTS[@]}" | sort); do
      printf "%s\t%s\n" "$fingerprint" "${CONTRACT_FINGERPRINT_COUNTS[$fingerprint]}"
    done | hash_stdin
  )"
fi
if [[ "${#MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}" -gt 0 ]]; then
  mutation_source_fingerprints_digest="$(
    for fingerprint in $(printf "%s\n" "${!MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}" | sort); do
      printf "%s\t%s\n" "$fingerprint" "${MUTATION_SOURCE_FINGERPRINT_COUNTS[$fingerprint]}"
    done | hash_stdin
  )"
fi

{
  printf "scope\tname\tvalue\n"
  printf "metric\texecuted_lanes\t%s\n" "${#EXECUTED_INDICES[@]}"
  printf "metric\tcontract_fingerprints_unique\t%s\n" "${#CONTRACT_FINGERPRINT_COUNTS[@]}"
  printf "metric\tmutation_source_fingerprints_unique\t%s\n" "${#MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}"
  printf "metric\tcontract_fingerprints_digest\t%s\n" "$contract_fingerprints_digest"
  printf "metric\tmutation_source_fingerprints_digest\t%s\n" "$mutation_source_fingerprints_digest"
  if [[ "${#CONTRACT_FINGERPRINT_COUNTS[@]}" -gt 0 ]]; then
    for fingerprint in $(printf "%s\n" "${!CONTRACT_FINGERPRINT_COUNTS[@]}" | sort); do
      printf "contract_fingerprint_count\t%s\t%s\n" "$fingerprint" "${CONTRACT_FINGERPRINT_COUNTS[$fingerprint]}"
    done
  fi
  if [[ "${#MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}" -gt 0 ]]; then
    for fingerprint in $(printf "%s\n" "${!MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}" | sort); do
      printf "mutation_source_fingerprint_count\t%s\t%s\n" "$fingerprint" "${MUTATION_SOURCE_FINGERPRINT_COUNTS[$fingerprint]}"
    done
  fi
} > "$PROVENANCE_SUMMARY_FILE"


provenance_gate_failures=0
PROVENANCE_GATE_DIAGNOSTICS_FILE=""
if [[ "$PROVENANCE_GATE_REPORT_ENABLED" -eq 1 ]]; then
  PROVENANCE_GATE_DIAGNOSTICS_FILE="${OUT_DIR}/provenance_gate_diagnostics.tsv.tmp"
  : > "$PROVENANCE_GATE_DIAGNOSTICS_FILE"
fi
if [[ "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS" -eq 1 ]]; then
  baseline_contract_case_ids="$(collect_fingerprint_case_ids_from_results "$BASELINE_RESULTS_FILE" "lane_contract_fingerprint")"
  current_contract_case_ids="$(collect_fingerprint_case_ids_from_results "$RESULTS_FILE" "lane_contract_fingerprint")"
  mapfile -t contract_case_diff < <(compute_new_case_ids "$baseline_contract_case_ids" "$current_contract_case_ids")
  baseline_contract_count="${contract_case_diff[0]:-0}"
  current_contract_count="${contract_case_diff[1]:-0}"
  new_contract_count="${contract_case_diff[2]:-0}"
  new_contract_case_ids="${contract_case_diff[3]:-}"
  if [[ "$new_contract_count" -gt 0 ]]; then
    contract_sample="$(case_ids_sample "$new_contract_case_ids" 3)"
    contract_detail="new lane contract fingerprint case ids observed (baseline=${baseline_contract_count} current=${current_contract_count}): ${contract_sample}"
    contract_message="Provenance gate: ${contract_detail}"
    echo "$contract_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.contract_fingerprint_case_ids.new" "$contract_detail" "$contract_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi
if [[ "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS" -eq 1 ]]; then
  baseline_source_case_ids="$(collect_fingerprint_case_ids_from_results "$BASELINE_RESULTS_FILE" "lane_mutation_source_fingerprint")"
  current_source_case_ids="$(collect_fingerprint_case_ids_from_results "$RESULTS_FILE" "lane_mutation_source_fingerprint")"
  mapfile -t source_case_diff < <(compute_new_case_ids "$baseline_source_case_ids" "$current_source_case_ids")
  baseline_source_count="${source_case_diff[0]:-0}"
  current_source_count="${source_case_diff[1]:-0}"
  new_source_count="${source_case_diff[2]:-0}"
  new_source_case_ids="${source_case_diff[3]:-}"
  if [[ "$new_source_count" -gt 0 ]]; then
    source_sample="$(case_ids_sample "$new_source_case_ids" 3)"
    source_detail="new lane mutation-source fingerprint case ids observed (baseline=${baseline_source_count} current=${current_source_count}): ${source_sample}"
    source_message="Provenance gate: ${source_detail}"
    echo "$source_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.mutation_source_fingerprint_case_ids.new" "$source_detail" "$source_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi
if [[ "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES" -eq 1 ]]; then
  baseline_contract_identities="$(collect_fingerprint_identities_from_results "$BASELINE_RESULTS_FILE" "lane_contract_fingerprint")"
  current_contract_identities="$(collect_fingerprint_identities_from_results "$RESULTS_FILE" "lane_contract_fingerprint")"
  mapfile -t contract_identity_diff < <(compute_new_case_ids "$baseline_contract_identities" "$current_contract_identities")
  baseline_contract_identity_count="${contract_identity_diff[0]:-0}"
  current_contract_identity_count="${contract_identity_diff[1]:-0}"
  new_contract_identity_count="${contract_identity_diff[2]:-0}"
  new_contract_identities="${contract_identity_diff[3]:-}"
  if [[ "$new_contract_identity_count" -gt 0 ]]; then
    contract_identity_sample="$(case_ids_sample "$new_contract_identities" 3)"
    contract_identity_detail="new contract fingerprint identities observed (baseline=${baseline_contract_identity_count} current=${current_contract_identity_count}): ${contract_identity_sample}"
    contract_identity_message="Provenance gate: ${contract_identity_detail}"
    echo "$contract_identity_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.contract_fingerprint_identities.new" "$contract_identity_detail" "$contract_identity_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi
if [[ "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES" -eq 1 ]]; then
  baseline_source_identities="$(collect_fingerprint_identities_from_results "$BASELINE_RESULTS_FILE" "lane_mutation_source_fingerprint")"
  current_source_identities="$(collect_fingerprint_identities_from_results "$RESULTS_FILE" "lane_mutation_source_fingerprint")"
  mapfile -t source_identity_diff < <(compute_new_case_ids "$baseline_source_identities" "$current_source_identities")
  baseline_source_identity_count="${source_identity_diff[0]:-0}"
  current_source_identity_count="${source_identity_diff[1]:-0}"
  new_source_identity_count="${source_identity_diff[2]:-0}"
  new_source_identities="${source_identity_diff[3]:-}"
  if [[ "$new_source_identity_count" -gt 0 ]]; then
    source_identity_sample="$(case_ids_sample "$new_source_identities" 3)"
    source_identity_detail="new mutation-source fingerprint identities observed (baseline=${baseline_source_identity_count} current=${current_source_identity_count}): ${source_identity_sample}"
    source_identity_message="Provenance gate: ${source_identity_detail}"
    echo "$source_identity_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.mutation_source_fingerprint_identities.new" "$source_identity_detail" "$source_identity_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi
if [[ "$FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE" -eq 1 ]]; then
  contract_identity_cardinality="${#CONTRACT_FINGERPRINT_COUNTS[@]}"
  if [[ "$contract_identity_cardinality" -gt 1 ]]; then
    contract_identity_divergence_sample="$(sample_fingerprint_count_pairs CONTRACT_FINGERPRINT_COUNTS 3)"
    contract_identity_divergence_detail="contract fingerprint divergence observed (unique=${contract_identity_cardinality}): ${contract_identity_divergence_sample}"
    contract_identity_divergence_message="Provenance gate: ${contract_identity_divergence_detail}"
    echo "$contract_identity_divergence_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.contract_fingerprint_identities.divergent" "$contract_identity_divergence_detail" "$contract_identity_divergence_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi
if [[ "$FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE" -eq 1 ]]; then
  source_identity_cardinality="${#MUTATION_SOURCE_FINGERPRINT_COUNTS[@]}"
  if [[ "$source_identity_cardinality" -gt 1 ]]; then
    source_identity_divergence_sample="$(sample_fingerprint_count_pairs MUTATION_SOURCE_FINGERPRINT_COUNTS 3)"
    source_identity_divergence_detail="mutation-source fingerprint divergence observed (unique=${source_identity_cardinality}): ${source_identity_divergence_sample}"
    source_identity_divergence_message="Provenance gate: ${source_identity_divergence_detail}"
    echo "$source_identity_divergence_message" >&2
    append_provenance_gate_diagnostic "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "mutation_matrix.provenance.mutation_source_fingerprint_identities.divergent" "$source_identity_divergence_detail" "$source_identity_divergence_message"
    provenance_gate_failures=$((provenance_gate_failures + 1))
  fi
fi

summary_failures="$failures"
if [[ "$provenance_gate_failures" -gt 0 ]]; then
  summary_failures=$((summary_failures + provenance_gate_failures))
fi
if [[ "$PROVENANCE_GATE_REPORT_ENABLED" -eq 1 ]]; then
  provenance_gate_status="pass"
  if [[ "$provenance_gate_failures" -gt 0 ]]; then
    provenance_gate_status="fail"
  fi
  write_provenance_gate_report "$provenance_gate_status" "$PROVENANCE_GATE_DIAGNOSTICS_FILE" "$PROVENANCE_GATE_REPORT_JSON" "$PROVENANCE_GATE_REPORT_TSV"
  rm -f "$PROVENANCE_GATE_DIAGNOSTICS_FILE"
fi

echo "Mutation matrix summary: pass=${passes} fail=${summary_failures}"
echo "Gate summary: $GATE_SUMMARY_FILE"
echo "Provenance summary: $PROVENANCE_SUMMARY_FILE"
if [[ "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_CASE_IDS" -eq 1 || "$FAIL_ON_NEW_CONTRACT_FINGERPRINT_IDENTITIES" -eq 1 || "$FAIL_ON_NEW_MUTATION_SOURCE_FINGERPRINT_IDENTITIES" -eq 1 || "$FAIL_ON_CONTRACT_FINGERPRINT_DIVERGENCE" -eq 1 || "$FAIL_ON_MUTATION_SOURCE_FINGERPRINT_DIVERGENCE" -eq 1 ]]; then
  echo "Provenance gate failures: ${provenance_gate_failures}"
fi
echo "Mutation matrix generated-mutation cache: hit_lanes=${generated_cache_hit_lanes} miss_lanes=${generated_cache_miss_lanes} saved_runtime_ns=${generated_cache_saved_runtime_ns} lock_wait_ns=${generated_cache_lock_wait_ns} lock_contended_lanes=${generated_cache_lock_contended_lanes}"
echo "Results: $RESULTS_FILE"
if [[ "$summary_failures" -ne 0 ]]; then
  exit 1
fi
