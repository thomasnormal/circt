#!/usr/bin/env bash
# CIRCT mutation coverage harness with formal pre-qualification and 4-way classes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
usage: run_mutation_cover.sh [options]

Required:
  --design FILE              Original design netlist (.il/.v/.sv)
  --tests-manifest FILE      TSV with columns:
                               test_id<TAB>run_cmd<TAB>result_file<TAB>kill_pattern<TAB>survive_pattern
  One of:
    --mutations-file FILE    Existing mutation list file
    --generate-mutations N   Generate N mutations via yosys mutate -list

Optional:
  --work-dir DIR             Output/work directory (default: ./mutation-cover-results)
  --summary-file FILE        Mutant-level classification TSV (default: <work-dir>/summary.tsv)
  --pair-file FILE           Pair-level qualification TSV (default: <work-dir>/pair_qualification.tsv)
  --results-file FILE        Pair-level detection TSV (default: <work-dir>/results.tsv)
  --metrics-file FILE        Metric-mode summary TSV (default: <work-dir>/metrics.tsv)
  --summary-json-file FILE   Machine-readable summary JSON (default: <work-dir>/summary.json)
  --improvement-file FILE    Improvement-mode summary TSV (default: <work-dir>/improvement.tsv)
  --reuse-pair-file FILE     Reuse activation/propagation from prior pair_qualification.tsv
  --reuse-summary-file FILE  Reuse prior summary.tsv detected_by_test as test-order hints
  --reuse-compat-mode MODE   Reuse compatibility policy: off|warn|strict (default: warn)
  --reuse-manifest-file FILE Write run compatibility manifest (default: <work-dir>/reuse_manifest.json)
  --reuse-cache-dir DIR      Compatibility-hash cache root for auto reuse artifacts
  --reuse-cache-mode MODE    Cache policy: off|read|read-write (default: read-write)
  --create-mutated-script FILE
                             Script compatible with mcy scripts/create_mutated.sh
                             (default: ~/mcy/scripts/create_mutated.sh)
  --mutant-format EXT        Mutant file extension: il|v|sv (default: il)
  --formal-activate-cmd CMD  Optional per-(test,mutant) activation classification cmd
  --formal-propagate-cmd CMD Optional per-(test,mutant) propagation classification cmd
  --formal-global-propagate-cmd CMD
                             Optional per-mutant propagation filter cmd run once
                             before per-test qualification/detection
  --formal-global-propagate-circt-lec PATH
                             Use circt-lec as built-in per-mutant global
                             propagation filter (mutually exclusive with
                             --formal-global-propagate-cmd)
  --formal-global-propagate-circt-lec-args ARGS
                             Extra args passed to circt-lec global filter
  --formal-global-propagate-c1 NAME
                             circt-lec -c1 module name (default: top)
  --formal-global-propagate-c2 NAME
                             circt-lec -c2 module name (default: top)
  --formal-global-propagate-z3 PATH
                             Optional z3 path passed to circt-lec --z3-path
  --formal-global-propagate-assume-known-inputs
                             Pass --assume-known-inputs to circt-lec global
                             propagation filter
  --formal-global-propagate-accept-xprop-only
                             Pass --accept-xprop-only to circt-lec global
                             propagation filter
  --formal-global-propagate-circt-bmc PATH
                             Use circt-bmc as built-in differential global
                             filter (orig vs mutant; mutually exclusive with
                             other global filter modes)
  --formal-global-propagate-circt-chain MODE
                             Built-in chained global filter strategy
                             (lec-then-bmc|bmc-then-lec|consensus). Requires both
                             --formal-global-propagate-circt-lec and
                             --formal-global-propagate-circt-bmc.
  --formal-global-propagate-circt-bmc-args ARGS
                             Extra args passed to circt-bmc global filter
  --formal-global-propagate-bmc-bound N
                             circt-bmc bound for global filter (default: 20)
  --formal-global-propagate-bmc-module NAME
                             circt-bmc --module name (default: top)
  --formal-global-propagate-bmc-run-smtlib
                             Pass --run-smtlib to circt-bmc global filter
  --formal-global-propagate-bmc-z3 PATH
                             Optional z3 path passed to circt-bmc --z3-path
  --formal-global-propagate-bmc-assume-known-inputs
                             Pass --assume-known-inputs to circt-bmc global
                             filter
  --formal-global-propagate-bmc-ignore-asserts-until N
                             Pass --ignore-asserts-until to circt-bmc global
                             filter (default: 0)
  --mutation-limit N         Process first N mutations (default: all)
  --mutations-top NAME       Top module name when auto-generating mutations
  --mutations-modes CSV      Comma-separated mutate modes for auto-generation
                             (concrete: inv,const0,const1,cnot0,cnot1;
                             families: arith,control,balanced,all)
  --mutations-mode-counts CSV
                             Comma-separated mode=count allocation for
                             auto-generation (sum must match --generate-mutations)
  --mutations-profiles CSV   Comma-separated named mutate profiles for
                             auto-generation
  --mutations-cfg CSV        Comma-separated KEY=VALUE mutate cfg entries
  --mutations-select CSV     Comma-separated mutate select expressions
  --mutations-seed N         Seed used with --generate-mutations (default: 1)
  --mutations-yosys PATH     Yosys executable for auto-generation (default: yosys)
  --jobs N                   Worker processes for per-mutant execution (default: 1)
  --resume                   Reuse existing per-mutant artifacts when present
  --coverage-threshold PCT   Hard-fail if detected/relevant * 100 below threshold
  --skip-baseline            Skip baseline sanity checks for test manifest
  --fail-on-undetected       Hard-fail if any propagated_not_detected mutant remains
  --fail-on-errors           Hard-fail if any infra/formal/test errors are observed
  -h, --help                 Show this help

Formal command conventions:
  Activation:
    - NOT_ACTIVATED in output => not activated
    - ACTIVATED in output     => activated
    - else fallback: exit 0 => not activated, exit 1 => activated, other => error
  Propagation:
    - NOT_PROPAGATED in output => not propagated
    - PROPAGATED in output     => propagated
    - else fallback: exit 0 => not propagated, exit 1 => propagated, other => error
  Built-in global filters:
    - circt-lec:
        LEC_RESULT=EQ => not_propagated
        LEC_RESULT=NEQ|UNKNOWN => propagated
    - circt-bmc (differential orig vs mutant):
        same BMC_RESULT (SAT/UNSAT) => not_propagated
        different BMC_RESULT or UNKNOWN => propagated
    - circt-chain lec-then-bmc:
        run circt-lec first; on LEC UNKNOWN/error, fall back to circt-bmc
        differential classification.
    - circt-chain bmc-then-lec:
        run circt-bmc differential first; on BMC UNKNOWN/error, fall back to
        circt-lec classification.
    - circt-chain consensus:
        run both circt-lec and differential circt-bmc; classify as
        not_propagated only when both agree not_propagated, otherwise
        propagated.

Test command conventions:
  Each test command runs in a per-(test,mutant) directory and must write result_file.
  If kill_pattern matches result_file => detected.
  If survive_pattern matches result_file => survived (propagated_not_detected for that pair).
USAGE
}

DESIGN=""
MUTATIONS_FILE=""
TESTS_MANIFEST=""
WORK_DIR="${PWD}/mutation-cover-results"
SUMMARY_FILE=""
PAIR_FILE=""
RESULTS_FILE=""
METRICS_FILE=""
SUMMARY_JSON_FILE=""
IMPROVEMENT_FILE=""
REUSE_PAIR_FILE=""
REUSE_SUMMARY_FILE=""
REUSE_COMPAT_MODE="warn"
REUSE_MANIFEST_FILE=""
REUSE_CACHE_DIR=""
REUSE_CACHE_MODE="read-write"
CREATE_MUTATED_SCRIPT="${HOME}/mcy/scripts/create_mutated.sh"
MUTANT_FORMAT="il"
FORMAL_ACTIVATE_CMD=""
FORMAL_PROPAGATE_CMD=""
FORMAL_GLOBAL_PROPAGATE_CMD=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS=""
FORMAL_GLOBAL_PROPAGATE_C1="top"
FORMAL_GLOBAL_PROPAGATE_C2="top"
FORMAL_GLOBAL_PROPAGATE_Z3=""
FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS=0
FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY=0
FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED=""
FORMAL_GLOBAL_PROPAGATE_Z3_RESOLVED=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS=""
FORMAL_GLOBAL_PROPAGATE_BMC_BOUND=20
FORMAL_GLOBAL_PROPAGATE_BMC_MODULE="top"
FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB=0
FORMAL_GLOBAL_PROPAGATE_BMC_Z3=""
FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS=0
FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL=0
FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED=""
FORMAL_GLOBAL_PROPAGATE_BMC_Z3_RESOLVED=""
MUTATION_LIMIT=0
GENERATE_MUTATIONS=0
MUTATIONS_TOP=""
MUTATIONS_MODES=""
MUTATIONS_MODE_COUNTS=""
MUTATIONS_PROFILES=""
MUTATIONS_CFG=""
MUTATIONS_SELECT=""
MUTATIONS_SEED=1
MUTATIONS_YOSYS="yosys"
JOBS=1
RESUME=0
COVERAGE_THRESHOLD=""
SKIP_BASELINE=0
FAIL_ON_UNDETECTED=0
FAIL_ON_ERRORS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --design) DESIGN="$2"; shift 2 ;;
    --mutations-file) MUTATIONS_FILE="$2"; shift 2 ;;
    --tests-manifest) TESTS_MANIFEST="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --summary-file) SUMMARY_FILE="$2"; shift 2 ;;
    --pair-file) PAIR_FILE="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --metrics-file) METRICS_FILE="$2"; shift 2 ;;
    --summary-json-file) SUMMARY_JSON_FILE="$2"; shift 2 ;;
    --improvement-file) IMPROVEMENT_FILE="$2"; shift 2 ;;
    --reuse-pair-file) REUSE_PAIR_FILE="$2"; shift 2 ;;
    --reuse-summary-file) REUSE_SUMMARY_FILE="$2"; shift 2 ;;
    --reuse-compat-mode) REUSE_COMPAT_MODE="$2"; shift 2 ;;
    --reuse-manifest-file) REUSE_MANIFEST_FILE="$2"; shift 2 ;;
    --reuse-cache-dir) REUSE_CACHE_DIR="$2"; shift 2 ;;
    --reuse-cache-mode) REUSE_CACHE_MODE="$2"; shift 2 ;;
    --create-mutated-script) CREATE_MUTATED_SCRIPT="$2"; shift 2 ;;
    --mutant-format) MUTANT_FORMAT="$2"; shift 2 ;;
    --formal-activate-cmd) FORMAL_ACTIVATE_CMD="$2"; shift 2 ;;
    --formal-propagate-cmd) FORMAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --formal-global-propagate-cmd) FORMAL_GLOBAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --formal-global-propagate-circt-lec) FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC="$2"; shift 2 ;;
    --formal-global-propagate-circt-lec-args) FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS="$2"; shift 2 ;;
    --formal-global-propagate-c1) FORMAL_GLOBAL_PROPAGATE_C1="$2"; shift 2 ;;
    --formal-global-propagate-c2) FORMAL_GLOBAL_PROPAGATE_C2="$2"; shift 2 ;;
    --formal-global-propagate-z3) FORMAL_GLOBAL_PROPAGATE_Z3="$2"; shift 2 ;;
    --formal-global-propagate-assume-known-inputs) FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS=1; shift ;;
    --formal-global-propagate-accept-xprop-only) FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY=1; shift ;;
    --formal-global-propagate-circt-bmc) FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC="$2"; shift 2 ;;
    --formal-global-propagate-circt-chain) FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN="$2"; shift 2 ;;
    --formal-global-propagate-circt-bmc-args) FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS="$2"; shift 2 ;;
    --formal-global-propagate-bmc-bound) FORMAL_GLOBAL_PROPAGATE_BMC_BOUND="$2"; shift 2 ;;
    --formal-global-propagate-bmc-module) FORMAL_GLOBAL_PROPAGATE_BMC_MODULE="$2"; shift 2 ;;
    --formal-global-propagate-bmc-run-smtlib) FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB=1; shift ;;
    --formal-global-propagate-bmc-z3) FORMAL_GLOBAL_PROPAGATE_BMC_Z3="$2"; shift 2 ;;
    --formal-global-propagate-bmc-assume-known-inputs) FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --formal-global-propagate-bmc-ignore-asserts-until) FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL="$2"; shift 2 ;;
    --mutation-limit) MUTATION_LIMIT="$2"; shift 2 ;;
    --generate-mutations) GENERATE_MUTATIONS="$2"; shift 2 ;;
    --mutations-top) MUTATIONS_TOP="$2"; shift 2 ;;
    --mutations-modes) MUTATIONS_MODES="$2"; shift 2 ;;
    --mutations-mode-counts) MUTATIONS_MODE_COUNTS="$2"; shift 2 ;;
    --mutations-profiles) MUTATIONS_PROFILES="$2"; shift 2 ;;
    --mutations-cfg) MUTATIONS_CFG="$2"; shift 2 ;;
    --mutations-select) MUTATIONS_SELECT="$2"; shift 2 ;;
    --mutations-seed) MUTATIONS_SEED="$2"; shift 2 ;;
    --mutations-yosys) MUTATIONS_YOSYS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --coverage-threshold) COVERAGE_THRESHOLD="$2"; shift 2 ;;
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --fail-on-undetected) FAIL_ON_UNDETECTED=1; shift ;;
    --fail-on-errors) FAIL_ON_ERRORS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DESIGN" || -z "$TESTS_MANIFEST" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$DESIGN" ]]; then
  echo "Design file not found: $DESIGN" >&2
  exit 1
fi
if [[ ! -f "$TESTS_MANIFEST" ]]; then
  echo "Tests manifest not found: $TESTS_MANIFEST" >&2
  exit 1
fi
if [[ -n "$REUSE_PAIR_FILE" && ! -f "$REUSE_PAIR_FILE" ]]; then
  echo "Reuse pair file not found: $REUSE_PAIR_FILE" >&2
  exit 1
fi
if [[ -n "$REUSE_SUMMARY_FILE" && ! -f "$REUSE_SUMMARY_FILE" ]]; then
  echo "Reuse summary file not found: $REUSE_SUMMARY_FILE" >&2
  exit 1
fi
if [[ ! "$REUSE_COMPAT_MODE" =~ ^(off|warn|strict)$ ]]; then
  echo "Invalid --reuse-compat-mode value: $REUSE_COMPAT_MODE (expected off|warn|strict)." >&2
  exit 1
fi
if [[ ! "$REUSE_CACHE_MODE" =~ ^(off|read|read-write)$ ]]; then
  echo "Invalid --reuse-cache-mode value: $REUSE_CACHE_MODE (expected off|read|read-write)." >&2
  exit 1
fi
if [[ -z "$REUSE_CACHE_DIR" ]]; then
  REUSE_CACHE_MODE="off"
fi
if [[ ! -x "$CREATE_MUTATED_SCRIPT" ]]; then
  echo "Mutant creation script is not executable: $CREATE_MUTATED_SCRIPT" >&2
  exit 1
fi
if [[ ! "$MUTANT_FORMAT" =~ ^(il|v|sv)$ ]]; then
  echo "Unsupported mutant format: $MUTANT_FORMAT (expected il|v|sv)." >&2
  exit 1
fi
if [[ ! "$MUTATION_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "Invalid --mutation-limit value: $MUTATION_LIMIT" >&2
  exit 1
fi
if [[ ! "$GENERATE_MUTATIONS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --generate-mutations value: $GENERATE_MUTATIONS" >&2
  exit 1
fi
if [[ ! "$MUTATIONS_SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --mutations-seed value: $MUTATIONS_SEED" >&2
  exit 1
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs value: $JOBS" >&2
  exit 1
fi
if [[ -n "$COVERAGE_THRESHOLD" ]] && ! [[ "$COVERAGE_THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid --coverage-threshold value: $COVERAGE_THRESHOLD" >&2
  exit 1
fi
if [[ -n "$MUTATIONS_FILE" && "$GENERATE_MUTATIONS" -gt 0 ]]; then
  echo "Use either --mutations-file or --generate-mutations, not both." >&2
  exit 1
fi
if [[ -z "$MUTATIONS_FILE" && "$GENERATE_MUTATIONS" -eq 0 ]]; then
  echo "Provide one mutation source: --mutations-file or --generate-mutations." >&2
  exit 1
fi
if [[ ! "$FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --formal-global-propagate-bmc-bound value: $FORMAL_GLOBAL_PROPAGATE_BMC_BOUND" >&2
  exit 1
fi
if [[ ! "$FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" =~ ^[0-9]+$ ]]; then
  echo "Invalid --formal-global-propagate-bmc-ignore-asserts-until value: $FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL" >&2
  exit 1
fi

if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" ]]; then
  if [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" != "lec-then-bmc" && "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" != "bmc-then-lec" && "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" != "consensus" ]]; then
    echo "Invalid --formal-global-propagate-circt-chain value: $FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN (expected lec-then-bmc|bmc-then-lec|consensus)." >&2
    exit 1
  fi
  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CMD" ]]; then
    echo "--formal-global-propagate-circt-chain cannot be combined with --formal-global-propagate-cmd." >&2
    exit 1
  fi
  if [[ -z "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC" || -z "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC" ]]; then
    echo "--formal-global-propagate-circt-chain requires both --formal-global-propagate-circt-lec and --formal-global-propagate-circt-bmc." >&2
    exit 1
  fi
else
  global_filter_mode_count=0
  [[ -n "$FORMAL_GLOBAL_PROPAGATE_CMD" ]] && global_filter_mode_count=$((global_filter_mode_count + 1))
  [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC" ]] && global_filter_mode_count=$((global_filter_mode_count + 1))
  [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC" ]] && global_filter_mode_count=$((global_filter_mode_count + 1))
  if [[ "$global_filter_mode_count" -gt 1 ]]; then
    echo "Use only one global filter mode: --formal-global-propagate-cmd, --formal-global-propagate-circt-lec, --formal-global-propagate-circt-bmc, or --formal-global-propagate-circt-chain." >&2
    exit 1
  fi
fi

resolve_tool_path() {
  local tool="$1"
  local resolved=""
  if [[ "$tool" == */* ]]; then
    if [[ -x "$tool" ]]; then
      printf "%s\n" "$tool"
      return 0
    fi
    return 1
  fi
  resolved="$(command -v "$tool" 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    printf "%s\n" "$resolved"
    return 0
  fi
  return 1
}

if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC" ]]; then
  if ! FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED="$(resolve_tool_path "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC")"; then
    echo "Unable to resolve --formal-global-propagate-circt-lec executable: $FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC" >&2
    exit 1
  fi
fi
if [[ -n "$FORMAL_GLOBAL_PROPAGATE_Z3" ]]; then
  if ! FORMAL_GLOBAL_PROPAGATE_Z3_RESOLVED="$(resolve_tool_path "$FORMAL_GLOBAL_PROPAGATE_Z3")"; then
    echo "Unable to resolve --formal-global-propagate-z3 executable: $FORMAL_GLOBAL_PROPAGATE_Z3" >&2
    exit 1
  fi
fi
if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC" ]]; then
  if ! FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED="$(resolve_tool_path "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC")"; then
    echo "Unable to resolve --formal-global-propagate-circt-bmc executable: $FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC" >&2
    exit 1
  fi
fi
if [[ -n "$FORMAL_GLOBAL_PROPAGATE_BMC_Z3" ]]; then
  if ! FORMAL_GLOBAL_PROPAGATE_BMC_Z3_RESOLVED="$(resolve_tool_path "$FORMAL_GLOBAL_PROPAGATE_BMC_Z3")"; then
    echo "Unable to resolve --formal-global-propagate-bmc-z3 executable: $FORMAL_GLOBAL_PROPAGATE_BMC_Z3" >&2
    exit 1
  fi
fi

mkdir -p "$WORK_DIR/mutations"
SUMMARY_FILE="${SUMMARY_FILE:-${WORK_DIR}/summary.tsv}"
PAIR_FILE="${PAIR_FILE:-${WORK_DIR}/pair_qualification.tsv}"
RESULTS_FILE="${RESULTS_FILE:-${WORK_DIR}/results.tsv}"
METRICS_FILE="${METRICS_FILE:-${WORK_DIR}/metrics.tsv}"
SUMMARY_JSON_FILE="${SUMMARY_JSON_FILE:-${WORK_DIR}/summary.json}"
IMPROVEMENT_FILE="${IMPROVEMENT_FILE:-${WORK_DIR}/improvement.tsv}"
REUSE_MANIFEST_FILE="${REUSE_MANIFEST_FILE:-${WORK_DIR}/reuse_manifest.json}"
if [[ "$REUSE_CACHE_MODE" != "off" ]]; then
  mkdir -p "$REUSE_CACHE_DIR"
fi

if [[ "$GENERATE_MUTATIONS" -gt 0 ]]; then
  MUTATIONS_FILE="${WORK_DIR}/generated_mutations.txt"
  gen_cmd=(
    "${SCRIPT_DIR}/generate_mutations_yosys.sh"
    --design "$DESIGN"
    --out "$MUTATIONS_FILE"
    --count "$GENERATE_MUTATIONS"
    --seed "$MUTATIONS_SEED"
    --yosys "$MUTATIONS_YOSYS"
  )
  if [[ -n "$MUTATIONS_TOP" ]]; then
    gen_cmd+=(--top "$MUTATIONS_TOP")
  fi
  if [[ -n "$MUTATIONS_MODES" ]]; then
    gen_cmd+=(--modes "$MUTATIONS_MODES")
  fi
  if [[ -n "$MUTATIONS_MODE_COUNTS" ]]; then
    gen_cmd+=(--mode-counts "$MUTATIONS_MODE_COUNTS")
  fi
  if [[ -n "$MUTATIONS_PROFILES" ]]; then
    gen_cmd+=(--profiles "$MUTATIONS_PROFILES")
  fi
  if [[ -n "$MUTATIONS_CFG" ]]; then
    gen_cmd+=(--cfgs "$MUTATIONS_CFG")
  fi
  if [[ -n "$MUTATIONS_SELECT" ]]; then
    gen_cmd+=(--selects "$MUTATIONS_SELECT")
  fi
  "${gen_cmd[@]}" > "${WORK_DIR}/generate_mutations.log" 2>&1
fi
if [[ ! -f "$MUTATIONS_FILE" ]]; then
  echo "Mutations file not found: $MUTATIONS_FILE" >&2
  exit 1
fi

declare -A TEST_CMD
declare -A TEST_RESULT_FILE
declare -A TEST_KILL_PATTERN
declare -A TEST_SURVIVE_PATTERN
declare -a TEST_ORDER
declare -A REUSE_ACTIVATION
declare -A REUSE_PROPAGATION
declare -A REUSE_ACTIVATE_EXIT
declare -A REUSE_PROPAGATE_EXIT
declare -A REUSE_DETECTED_TEST
declare -A REUSE_GLOBAL_FILTER_STATE
declare -A REUSE_GLOBAL_FILTER_RC
declare -A REUSE_GLOBAL_CHAIN_LEC_UNKNOWN_FALLBACK
declare -A REUSE_GLOBAL_CHAIN_BMC_RESOLVED_NOT_PROPAGATED
declare -A REUSE_GLOBAL_CHAIN_BMC_UNKNOWN_FALLBACK
declare -A REUSE_GLOBAL_CHAIN_LEC_RESOLVED_NOT_PROPAGATED
declare -A REUSE_GLOBAL_CHAIN_CONSENSUS_NOT_PROPAGATED
declare -A REUSE_GLOBAL_CHAIN_CONSENSUS_DISAGREEMENT
declare -A REUSE_GLOBAL_CHAIN_CONSENSUS_ERROR

REUSE_COMPAT_HASH=""
DESIGN_FILE_HASH=""
TESTS_MANIFEST_HASH=""
MUTATIONS_SET_HASH=""
CREATE_MUTATED_SCRIPT_HASH=""
FORMAL_ACTIVATE_CMD_HASH=""
FORMAL_PROPAGATE_CMD_HASH=""
FORMAL_GLOBAL_PROPAGATE_CMD_HASH=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_HASH=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS_HASH=""
FORMAL_GLOBAL_PROPAGATE_Z3_HASH=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_HASH=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN_HASH=""
FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS_HASH=""
FORMAL_GLOBAL_PROPAGATE_BMC_Z3_HASH=""
REUSE_COMPAT_SCHEMA_VERSION=1
REUSE_CACHE_ENTRY_DIR=""
REUSE_PAIR_SOURCE="none"
REUSE_SUMMARY_SOURCE="none"
REUSE_CACHE_WRITE_STATUS="disabled"

declare -a MUTATION_IDS
declare -a MUTATION_SPECS
MALFORMED_MUTATION_LINES=0

load_tests_manifest() {
  local line=""
  local test_id=""
  local run_cmd=""
  local result_file=""
  local kill_pattern=""
  local survive_pattern=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    IFS=$'\t' read -r test_id run_cmd result_file kill_pattern survive_pattern _ <<< "$line"
    if [[ -z "$test_id" || -z "$run_cmd" || -z "$result_file" || -z "$kill_pattern" || -z "$survive_pattern" ]]; then
      echo "Malformed test manifest line: $line" >&2
      exit 1
    fi
    TEST_ORDER+=("$test_id")
    TEST_CMD["$test_id"]="$run_cmd"
    TEST_RESULT_FILE["$test_id"]="$result_file"
    TEST_KILL_PATTERN["$test_id"]="$kill_pattern"
    TEST_SURVIVE_PATTERN["$test_id"]="$survive_pattern"
  done < "$TESTS_MANIFEST"

  if [[ "${#TEST_ORDER[@]}" -eq 0 ]]; then
    echo "Tests manifest has no usable entries: $TESTS_MANIFEST" >&2
    exit 1
  fi
}

load_mutations() {
  local raw_line=""
  local line=""
  local mutation_id=""
  local mutation_spec=""
  local count=0
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    if [[ "$MUTATION_LIMIT" -gt 0 && "$count" -ge "$MUTATION_LIMIT" ]]; then
      break
    fi

    mutation_id="${line%%[[:space:]]*}"
    mutation_spec="${line#"$mutation_id"}"
    mutation_spec="${mutation_spec#"${mutation_spec%%[![:space:]]*}"}"
    mutation_spec="${mutation_spec//$'\t'/ }"
    if [[ -z "$mutation_id" || -z "$mutation_spec" ]]; then
      MALFORMED_MUTATION_LINES=$((MALFORMED_MUTATION_LINES + 1))
      continue
    fi
    MUTATION_IDS+=("$mutation_id")
    MUTATION_SPECS+=("$mutation_spec")
    count=$((count + 1))
  done < "$MUTATIONS_FILE"
}

load_reuse_pairs() {
  local line=""
  local mutation_id=""
  local test_id=""
  local activation=""
  local propagation=""
  local activate_exit=""
  local propagate_exit=""
  local note=""
  local key=""

  if [[ -z "$REUSE_PAIR_FILE" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    IFS=$'\t' read -r mutation_id test_id activation propagation activate_exit propagate_exit note <<< "$line"
    if [[ -z "$mutation_id" || -z "$test_id" ]]; then
      continue
    fi
    if [[ "$test_id" == "-" ]]; then
      case "$propagation" in
        not_propagated|propagated) ;;
        *) continue ;;
      esac
      if [[ "$note" != global_filter_* ]]; then
        continue
      fi
      REUSE_GLOBAL_FILTER_STATE["$mutation_id"]="$propagation"
      REUSE_GLOBAL_FILTER_RC["$mutation_id"]="${propagate_exit:--1}"
      REUSE_GLOBAL_CHAIN_LEC_UNKNOWN_FALLBACK["$mutation_id"]="$([[ "$note" == *"chain_lec_unknown_fallback=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_BMC_RESOLVED_NOT_PROPAGATED["$mutation_id"]="$([[ "$note" == *"chain_bmc_resolved_not_propagated=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_BMC_UNKNOWN_FALLBACK["$mutation_id"]="$([[ "$note" == *"chain_bmc_unknown_fallback=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_LEC_RESOLVED_NOT_PROPAGATED["$mutation_id"]="$([[ "$note" == *"chain_lec_resolved_not_propagated=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_CONSENSUS_NOT_PROPAGATED["$mutation_id"]="$([[ "$note" == *"chain_consensus_not_propagated=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_CONSENSUS_DISAGREEMENT["$mutation_id"]="$([[ "$note" == *"chain_consensus_disagreement=1"* ]] && printf "1" || printf "0")"
      REUSE_GLOBAL_CHAIN_CONSENSUS_ERROR["$mutation_id"]="$([[ "$note" == *"chain_consensus_error=1"* ]] && printf "1" || printf "0")"
      continue
    fi
    if [[ "$activation" != "activated" && "$activation" != "not_activated" ]]; then
      continue
    fi
    case "$propagation" in
      -|not_propagated|propagated) ;;
      *) continue ;;
    esac
    if [[ "$activation" == "not_activated" ]]; then
      propagation="-"
      propagate_exit="-1"
    fi
    key="${mutation_id}"$'\t'"${test_id}"
    REUSE_ACTIVATION["$key"]="$activation"
    REUSE_PROPAGATION["$key"]="$propagation"
    REUSE_ACTIVATE_EXIT["$key"]="${activate_exit:--1}"
    REUSE_PROPAGATE_EXIT["$key"]="${propagate_exit:--1}"
  done < "$REUSE_PAIR_FILE"
}

load_reuse_summary() {
  local line=""
  local mutation_id=""
  local classification=""
  local detected_by_test=""
  local base_class=""

  if [[ -z "$REUSE_SUMMARY_FILE" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    IFS=$'\t' read -r mutation_id classification _ detected_by_test _ <<< "$line"
    if [[ -z "$mutation_id" || -z "$classification" ]]; then
      continue
    fi
    base_class="${classification%%+*}"
    if [[ "$base_class" != "detected" ]]; then
      continue
    fi
    if [[ -z "$detected_by_test" || "$detected_by_test" == "-" ]]; then
      continue
    fi
    if [[ -z "${TEST_CMD[$detected_by_test]+x}" ]]; then
      continue
    fi
    REUSE_DETECTED_TEST["$mutation_id"]="$detected_by_test"
  done < "$REUSE_SUMMARY_FILE"
}

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
  echo "No SHA-256 hashing tool found (sha256sum/shasum/openssl)." >&2
  exit 1
}

hash_string() {
  local value="$1"
  printf "%s" "$value" | hash_stdin
}

hash_file() {
  local file="$1"
  hash_stdin < "$file"
}

json_escape() {
  printf "%s" "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

build_reuse_compat_hash() {
  local i=""
  local compat_payload=""

  DESIGN_FILE_HASH="$(hash_file "$DESIGN")"
  TESTS_MANIFEST_HASH="$(hash_file "$TESTS_MANIFEST")"
  CREATE_MUTATED_SCRIPT_HASH="$(hash_file "$CREATE_MUTATED_SCRIPT")"
  FORMAL_ACTIVATE_CMD_HASH="$(hash_string "$FORMAL_ACTIVATE_CMD")"
  FORMAL_PROPAGATE_CMD_HASH="$(hash_string "$FORMAL_PROPAGATE_CMD")"
  FORMAL_GLOBAL_PROPAGATE_CMD_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CMD")"
  FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED")"
  FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS")"
  FORMAL_GLOBAL_PROPAGATE_Z3_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_Z3_RESOLVED")"
  FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED")"
  FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN")"
  FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS")"
  FORMAL_GLOBAL_PROPAGATE_BMC_Z3_HASH="$(hash_string "$FORMAL_GLOBAL_PROPAGATE_BMC_Z3_RESOLVED")"

  MUTATIONS_SET_HASH="$(
    for i in "${!MUTATION_IDS[@]}"; do
      printf "%s\t%s\n" "${MUTATION_IDS[$i]}" "${MUTATION_SPECS[$i]}"
    done | hash_stdin
  )"

  compat_payload="$(
    cat <<EOF
schema_version=${REUSE_COMPAT_SCHEMA_VERSION}
design_hash=${DESIGN_FILE_HASH}
tests_manifest_hash=${TESTS_MANIFEST_HASH}
mutations_set_hash=${MUTATIONS_SET_HASH}
mutant_format=${MUTANT_FORMAT}
create_mutated_script_hash=${CREATE_MUTATED_SCRIPT_HASH}
formal_activate_cmd_hash=${FORMAL_ACTIVATE_CMD_HASH}
formal_propagate_cmd_hash=${FORMAL_PROPAGATE_CMD_HASH}
formal_global_propagate_cmd_hash=${FORMAL_GLOBAL_PROPAGATE_CMD_HASH}
formal_global_propagate_circt_lec_hash=${FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_HASH}
formal_global_propagate_circt_lec_args_hash=${FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS_HASH}
formal_global_propagate_z3_hash=${FORMAL_GLOBAL_PROPAGATE_Z3_HASH}
formal_global_propagate_c1=${FORMAL_GLOBAL_PROPAGATE_C1}
formal_global_propagate_c2=${FORMAL_GLOBAL_PROPAGATE_C2}
formal_global_propagate_assume_known_inputs=${FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS}
formal_global_propagate_accept_xprop_only=${FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY}
formal_global_propagate_circt_bmc_hash=${FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_HASH}
formal_global_propagate_circt_chain_hash=${FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN_HASH}
formal_global_propagate_circt_bmc_args_hash=${FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS_HASH}
formal_global_propagate_bmc_bound=${FORMAL_GLOBAL_PROPAGATE_BMC_BOUND}
formal_global_propagate_bmc_module=${FORMAL_GLOBAL_PROPAGATE_BMC_MODULE}
formal_global_propagate_bmc_run_smtlib=${FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB}
formal_global_propagate_bmc_z3_hash=${FORMAL_GLOBAL_PROPAGATE_BMC_Z3_HASH}
formal_global_propagate_bmc_assume_known_inputs=${FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS}
formal_global_propagate_bmc_ignore_asserts_until=${FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL}
EOF
  )"
  REUSE_COMPAT_HASH="$(hash_string "$compat_payload")"
}

extract_manifest_compat_hash() {
  local manifest_file="$1"
  grep -o '"compat_hash"[[:space:]]*:[[:space:]]*"[^"]*"' "$manifest_file" | \
    head -n1 | sed 's/.*"compat_hash"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/'
}

validate_reuse_file_compat() {
  local label="$1"
  local reuse_file="$2"
  local manifest_file="${reuse_file}.manifest.json"
  local manifest_hash=""
  local reason=""

  if [[ "$REUSE_COMPAT_MODE" == "off" ]]; then
    return 0
  fi

  if [[ ! -f "$manifest_file" ]]; then
    if [[ "$REUSE_COMPAT_MODE" == "strict" ]]; then
      reason="missing sidecar manifest (${manifest_file})"
    else
      echo "Warning: Reuse ${label} file has no sidecar manifest (${manifest_file}); proceeding without compatibility proof." >&2
      return 0
    fi
  fi

  if [[ -z "$reason" ]]; then
    manifest_hash="$(extract_manifest_compat_hash "$manifest_file")"
    if [[ -z "$manifest_hash" ]]; then
      reason="manifest missing compat_hash (${manifest_file})"
    elif [[ "$manifest_hash" != "$REUSE_COMPAT_HASH" ]]; then
      reason="compat_hash mismatch (${manifest_hash} != ${REUSE_COMPAT_HASH})"
    fi
  fi

  if [[ -z "$reason" ]]; then
    return 0
  fi

  if [[ "$REUSE_COMPAT_MODE" == "strict" ]]; then
    echo "Reuse ${label} file compatibility check failed: ${reason}" >&2
    return 1
  fi

  echo "Warning: Reuse ${label} file compatibility check failed: ${reason}. Disabling reuse for ${reuse_file}." >&2
  return 2
}

write_reuse_manifest() {
  local out_file="$1"
  local artifact_type="$2"
  local now_utc=""
  local out_dir=""

  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  out_dir="$(dirname "$out_file")"
  mkdir -p "$out_dir"
  cat > "$out_file" <<EOF
{
  "schema_version": ${REUSE_COMPAT_SCHEMA_VERSION},
  "artifact_type": "$(json_escape "$artifact_type")",
  "generated_at_utc": "${now_utc}",
  "compat_hash": "${REUSE_COMPAT_HASH}",
  "design_file": "$(json_escape "$DESIGN")",
  "design_file_sha256": "${DESIGN_FILE_HASH}",
  "tests_manifest": "$(json_escape "$TESTS_MANIFEST")",
  "tests_manifest_sha256": "${TESTS_MANIFEST_HASH}",
  "mutation_count": ${#MUTATION_IDS[@]},
  "mutations_set_sha256": "${MUTATIONS_SET_HASH}",
  "mutant_format": "${MUTANT_FORMAT}",
  "create_mutated_script": "$(json_escape "$CREATE_MUTATED_SCRIPT")",
  "create_mutated_script_sha256": "${CREATE_MUTATED_SCRIPT_HASH}",
  "formal_activate_cmd_sha256": "${FORMAL_ACTIVATE_CMD_HASH}",
  "formal_propagate_cmd_sha256": "${FORMAL_PROPAGATE_CMD_HASH}",
  "formal_global_propagate_cmd_sha256": "${FORMAL_GLOBAL_PROPAGATE_CMD_HASH}",
  "formal_global_propagate_circt_lec_sha256": "${FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_HASH}",
  "formal_global_propagate_circt_lec_args_sha256": "${FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS_HASH}",
  "formal_global_propagate_z3_sha256": "${FORMAL_GLOBAL_PROPAGATE_Z3_HASH}",
  "formal_global_propagate_c1": "$(json_escape "$FORMAL_GLOBAL_PROPAGATE_C1")",
  "formal_global_propagate_c2": "$(json_escape "$FORMAL_GLOBAL_PROPAGATE_C2")",
  "formal_global_propagate_assume_known_inputs": ${FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS},
  "formal_global_propagate_accept_xprop_only": ${FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY},
  "formal_global_propagate_circt_bmc_sha256": "${FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_HASH}",
  "formal_global_propagate_circt_chain_sha256": "${FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN_HASH}",
  "formal_global_propagate_circt_bmc_args_sha256": "${FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS_HASH}",
  "formal_global_propagate_bmc_bound": ${FORMAL_GLOBAL_PROPAGATE_BMC_BOUND},
  "formal_global_propagate_bmc_module": "$(json_escape "$FORMAL_GLOBAL_PROPAGATE_BMC_MODULE")",
  "formal_global_propagate_bmc_run_smtlib": ${FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB},
  "formal_global_propagate_bmc_z3_sha256": "${FORMAL_GLOBAL_PROPAGATE_BMC_Z3_HASH}",
  "formal_global_propagate_bmc_assume_known_inputs": ${FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS},
  "formal_global_propagate_bmc_ignore_asserts_until": ${FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL}
}
EOF
}

resolve_cache_reuse_inputs() {
  local cache_pair_file=""
  local cache_summary_file=""

  REUSE_PAIR_SOURCE="$([[ -n "$REUSE_PAIR_FILE" ]] && printf "explicit" || printf "none")"
  REUSE_SUMMARY_SOURCE="$([[ -n "$REUSE_SUMMARY_FILE" ]] && printf "explicit" || printf "none")"
  REUSE_CACHE_ENTRY_DIR=""
  if [[ "$REUSE_CACHE_MODE" == "off" ]]; then
    return
  fi

  REUSE_CACHE_ENTRY_DIR="${REUSE_CACHE_DIR}/${REUSE_COMPAT_HASH}"
  cache_pair_file="${REUSE_CACHE_ENTRY_DIR}/pair_qualification.tsv"
  cache_summary_file="${REUSE_CACHE_ENTRY_DIR}/summary.tsv"

  if [[ -z "$REUSE_PAIR_FILE" && -f "$cache_pair_file" ]]; then
    REUSE_PAIR_FILE="$cache_pair_file"
    REUSE_PAIR_SOURCE="cache"
  fi

  if [[ -z "$REUSE_SUMMARY_FILE" && -f "$cache_summary_file" ]]; then
    REUSE_SUMMARY_FILE="$cache_summary_file"
    REUSE_SUMMARY_SOURCE="cache"
  fi
}

cache_copy_file() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.tmp.$$"

  cp "$src" "$tmp"
  mv "$tmp" "$dst"
}

publish_reuse_cache() {
  if [[ "$REUSE_CACHE_MODE" == "off" ]]; then
    REUSE_CACHE_WRITE_STATUS="disabled"
    return 0
  fi
  if [[ "$REUSE_CACHE_MODE" == "read" ]]; then
    REUSE_CACHE_WRITE_STATUS="read_only"
    return 0
  fi
  if [[ "$errors" -gt 0 ]]; then
    REUSE_CACHE_WRITE_STATUS="skipped_errors"
    return 0
  fi
  mkdir -p "$REUSE_CACHE_ENTRY_DIR"
  cache_copy_file "$PAIR_FILE" "${REUSE_CACHE_ENTRY_DIR}/pair_qualification.tsv"
  cache_copy_file "${PAIR_FILE}.manifest.json" "${REUSE_CACHE_ENTRY_DIR}/pair_qualification.tsv.manifest.json"
  cache_copy_file "$SUMMARY_FILE" "${REUSE_CACHE_ENTRY_DIR}/summary.tsv"
  cache_copy_file "${SUMMARY_FILE}.manifest.json" "${REUSE_CACHE_ENTRY_DIR}/summary.tsv.manifest.json"
  cache_copy_file "$REUSE_MANIFEST_FILE" "${REUSE_CACHE_ENTRY_DIR}/reuse_manifest.json"
  REUSE_CACHE_WRITE_STATUS="written"
}

run_command() {
  local run_dir="$1"
  local cmd="$2"
  local log_file="$3"
  local exit_code=0
  set +e
  (
    cd "$run_dir"
    bash -lc "$cmd"
  ) >"$log_file" 2>&1
  exit_code=$?
  set -e
  return "$exit_code"
}

run_command_argv() {
  local run_dir="$1"
  local log_file="$2"
  shift 2
  local exit_code=0
  set +e
  (
    cd "$run_dir"
    "$@"
  ) >"$log_file" 2>&1
  exit_code=$?
  set -e
  return "$exit_code"
}

classify_activate() {
  local run_dir="$1"
  local log_file="$2"
  local rc=0

  if [[ -z "$FORMAL_ACTIVATE_CMD" ]]; then
    printf "activated\t-1\n"
    return
  fi

  rc=0
  if ! run_command "$run_dir" "$FORMAL_ACTIVATE_CMD" "$log_file"; then
    rc=$?
  fi
  if grep -Eiq '(^|[^[:alnum:]_])NOT_ACTIVATED([^[:alnum:]_]|$)' "$log_file"; then
    printf "not_activated\t%s\n" "$rc"
    return
  fi
  if grep -Eiq '(^|[^[:alnum:]_])ACTIVATED([^[:alnum:]_]|$)' "$log_file"; then
    printf "activated\t%s\n" "$rc"
    return
  fi
  case "$rc" in
    0) printf "not_activated\t0\n" ;;
    1) printf "activated\t1\n" ;;
    *) printf "error\t%s\n" "$rc" ;;
  esac
}

classify_propagate_cmd() {
  local run_dir="$1"
  local log_file="$2"
  local propagate_cmd="$3"
  local rc=0

  if [[ -z "$propagate_cmd" ]]; then
    printf "propagated\t-1\n"
    return
  fi

  rc=0
  if ! run_command "$run_dir" "$propagate_cmd" "$log_file"; then
    rc=$?
  fi
  if grep -Eiq '(^|[^[:alnum:]_])NOT_PROPAGATED([^[:alnum:]_]|$)' "$log_file"; then
    printf "not_propagated\t%s\n" "$rc"
    return
  fi
  if grep -Eiq '(^|[^[:alnum:]_])PROPAGATED([^[:alnum:]_]|$)' "$log_file"; then
    printf "propagated\t%s\n" "$rc"
    return
  fi
  case "$rc" in
    0) printf "not_propagated\t0\n" ;;
    1) printf "propagated\t1\n" ;;
    *) printf "error\t%s\n" "$rc" ;;
  esac
}

classify_propagate() {
  classify_propagate_cmd "$1" "$2" "$FORMAL_PROPAGATE_CMD"
}

classify_global_propagate_circt_lec_raw() {
  local run_dir="$1"
  local log_file="$2"
  local rc=0
  local -a lec_cmd
  local -a extra_args

  if [[ -z "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED" ]]; then
    printf "error\t-1\n"
    return
  fi

  lec_cmd=("$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED" "--run-smtlib")
  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_Z3_RESOLVED" ]]; then
    lec_cmd+=("--z3-path=$FORMAL_GLOBAL_PROPAGATE_Z3_RESOLVED")
  fi
  if [[ "$FORMAL_GLOBAL_PROPAGATE_ASSUME_KNOWN_INPUTS" -eq 1 ]]; then
    lec_cmd+=("--assume-known-inputs")
  fi
  if [[ "$FORMAL_GLOBAL_PROPAGATE_ACCEPT_XPROP_ONLY" -eq 1 ]]; then
    lec_cmd+=("--accept-xprop-only")
  fi
  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS" ]]; then
    read -r -a extra_args <<< "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_ARGS"
    lec_cmd+=("${extra_args[@]}")
  fi
  lec_cmd+=("-c1=$FORMAL_GLOBAL_PROPAGATE_C1" "-c2=$FORMAL_GLOBAL_PROPAGATE_C2" "$ORIG_DESIGN" "$MUTANT_DESIGN")

  rc=0
  if ! run_command_argv "$run_dir" "$log_file" "${lec_cmd[@]}"; then
    rc=$?
  fi

  if grep -Eq 'LEC_RESULT=EQ' "$log_file"; then
    printf "eq\t%s\n" "$rc"
    return
  fi
  if grep -Eq 'LEC_RESULT=NEQ' "$log_file"; then
    printf "neq\t%s\n" "$rc"
    return
  fi
  if grep -Eq 'LEC_RESULT=UNKNOWN' "$log_file"; then
    printf "unknown\t%s\n" "$rc"
    return
  fi
  printf "error\t%s\n" "$rc"
}

classify_global_propagate_circt_lec() {
  local run_dir="$1"
  local log_file="$2"
  local raw_state=""
  local raw_rc="-1"
  read -r raw_state raw_rc < <(classify_global_propagate_circt_lec_raw "$run_dir" "$log_file")
  case "$raw_state" in
    eq) printf "not_propagated\t%s\n" "$raw_rc" ;;
    neq|unknown) printf "propagated\t%s\n" "$raw_rc" ;;
    *) printf "error\t%s\n" "$raw_rc" ;;
  esac
}

extract_bmc_result_token() {
  local log_file="$1"
  local token=""
  token="$(grep -Eo 'BMC_RESULT=(SAT|UNSAT|UNKNOWN)' "$log_file" | tail -n1 || true)"
  case "$token" in
    BMC_RESULT=SAT) printf "sat\n" ;;
    BMC_RESULT=UNSAT) printf "unsat\n" ;;
    BMC_RESULT=UNKNOWN) printf "unknown\n" ;;
    *) printf "\n" ;;
  esac
}

classify_global_propagate_circt_bmc_raw() {
  local run_dir="$1"
  local log_file="$2"
  local rc=0
  local orig_rc=0
  local mutant_rc=0
  local orig_result=""
  local mutant_result=""
  local orig_log="${log_file}.orig"
  local mutant_log="${log_file}.mutant"
  local -a bmc_common_cmd
  local -a extra_args

  if [[ -z "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED" ]]; then
    printf "error\t-1\n"
    return
  fi

  bmc_common_cmd=(
    "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED"
    -b "$FORMAL_GLOBAL_PROPAGATE_BMC_BOUND"
    "--module=$FORMAL_GLOBAL_PROPAGATE_BMC_MODULE"
    "--ignore-asserts-until=$FORMAL_GLOBAL_PROPAGATE_BMC_IGNORE_ASSERTS_UNTIL"
  )
  if [[ "$FORMAL_GLOBAL_PROPAGATE_BMC_RUN_SMTLIB" -eq 1 ]]; then
    bmc_common_cmd+=("--run-smtlib")
  fi
  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_BMC_Z3_RESOLVED" ]]; then
    bmc_common_cmd+=("--z3-path=$FORMAL_GLOBAL_PROPAGATE_BMC_Z3_RESOLVED")
  fi
  if [[ "$FORMAL_GLOBAL_PROPAGATE_BMC_ASSUME_KNOWN_INPUTS" -eq 1 ]]; then
    bmc_common_cmd+=("--assume-known-inputs")
  fi
  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS" ]]; then
    read -r -a extra_args <<< "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_ARGS"
    bmc_common_cmd+=("${extra_args[@]}")
  fi

  orig_rc=0
  if ! run_command_argv "$run_dir" "$orig_log" "${bmc_common_cmd[@]}" "$ORIG_DESIGN"; then
    orig_rc=$?
  fi
  mutant_rc=0
  if ! run_command_argv "$run_dir" "$mutant_log" "${bmc_common_cmd[@]}" "$MUTANT_DESIGN"; then
    mutant_rc=$?
  fi

  orig_result="$(extract_bmc_result_token "$orig_log")"
  mutant_result="$(extract_bmc_result_token "$mutant_log")"
  {
    printf "# bmc_orig_exit=%s bmc_orig_result=%s\n" "$orig_rc" "${orig_result:-none}"
    cat "$orig_log"
    printf "\n# bmc_mutant_exit=%s bmc_mutant_result=%s\n" "$mutant_rc" "${mutant_result:-none}"
    cat "$mutant_log"
  } > "$log_file"

  if [[ -z "$orig_result" || -z "$mutant_result" ]]; then
    rc="$orig_rc"
    if [[ "$rc" -eq 0 ]]; then
      rc="$mutant_rc"
    fi
    printf "error\t%s\n" "$rc"
    return
  fi

  rc="$orig_rc"
  if [[ "$rc" -eq 0 ]]; then
    rc="$mutant_rc"
  fi
  if [[ "$orig_result" == "unknown" || "$mutant_result" == "unknown" ]]; then
    printf "unknown\t%s\n" "$rc"
    return
  fi
  if [[ "$orig_result" == "$mutant_result" ]]; then
    printf "equal\t%s\n" "$rc"
  else
    printf "different\t%s\n" "$rc"
  fi
}

classify_global_propagate_circt_bmc() {
  local run_dir="$1"
  local log_file="$2"
  local raw_state=""
  local raw_rc="-1"
  read -r raw_state raw_rc < <(classify_global_propagate_circt_bmc_raw "$run_dir" "$log_file")
  case "$raw_state" in
    equal) printf "not_propagated\t%s\n" "$raw_rc" ;;
    different|unknown) printf "propagated\t%s\n" "$raw_rc" ;;
    *) printf "error\t%s\n" "$raw_rc" ;;
  esac
}

classify_global_propagate_circt_chain() {
  local run_dir="$1"
  local log_file="$2"
  local chain_mode="$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN"
  local primary_tool=""
  local primary_state=""
  local primary_rc="-1"
  local fallback_tool=""
  local fallback_state=""
  local fallback_rc="-1"
  local lec_state=""
  local lec_rc="-1"
  local bmc_state=""
  local bmc_rc="-1"
  local final_rc="-1"
  local lec_log="${log_file}.lec"
  local bmc_log="${log_file}.bmc"

  if [[ "$chain_mode" == "consensus" ]]; then
    read -r lec_state lec_rc < <(classify_global_propagate_circt_lec_raw "$run_dir" "$lec_log")
    read -r bmc_state bmc_rc < <(classify_global_propagate_circt_bmc_raw "$run_dir" "$bmc_log")
    final_rc="$lec_rc"
    if [[ "$final_rc" -eq 0 ]]; then
      final_rc="$bmc_rc"
    fi
    {
      printf "# chain_mode=consensus lec_state=%s lec_rc=%s\n" "$lec_state" "$lec_rc"
      cat "$lec_log"
      printf "\n# chain_consensus_bmc_state=%s bmc_rc=%s\n" "$bmc_state" "$bmc_rc"
      cat "$bmc_log"
    } > "$log_file"
    if [[ "$lec_state" == "eq" && "$bmc_state" == "equal" ]]; then
      printf "not_propagated\t%s\n" "$final_rc"
      return
    fi
    if [[ "$lec_state" == "neq" || "$lec_state" == "unknown" || "$bmc_state" == "different" || "$bmc_state" == "unknown" ]]; then
      printf "propagated\t%s\n" "$final_rc"
      return
    fi
    printf "error\t%s\n" "$final_rc"
    return
  fi

  case "$chain_mode" in
    lec-then-bmc)
      primary_tool="lec"
      fallback_tool="bmc"
      ;;
    bmc-then-lec)
      primary_tool="bmc"
      fallback_tool="lec"
      ;;
    *)
      printf "error\t-1\n"
      return
      ;;
  esac

  if [[ "$primary_tool" == "lec" ]]; then
    read -r primary_state primary_rc < <(classify_global_propagate_circt_lec_raw "$run_dir" "$lec_log")
    case "$primary_state" in
      eq)
        cp "$lec_log" "$log_file"
        printf "not_propagated\t%s\n" "$primary_rc"
        return
        ;;
      neq)
        cp "$lec_log" "$log_file"
        printf "propagated\t%s\n" "$primary_rc"
        return
        ;;
    esac
  elif [[ "$primary_tool" == "bmc" ]]; then
    read -r primary_state primary_rc < <(classify_global_propagate_circt_bmc_raw "$run_dir" "$bmc_log")
    case "$primary_state" in
      equal)
        cp "$bmc_log" "$log_file"
        printf "not_propagated\t%s\n" "$primary_rc"
        return
        ;;
      different)
        cp "$bmc_log" "$log_file"
        printf "propagated\t%s\n" "$primary_rc"
        return
        ;;
    esac
  else
    printf "error\t-1\n"
    return
  fi

  if [[ "$fallback_tool" == "bmc" ]]; then
    read -r fallback_state fallback_rc < <(classify_global_propagate_circt_bmc_raw "$run_dir" "$bmc_log")
  elif [[ "$fallback_tool" == "lec" ]]; then
    read -r fallback_state fallback_rc < <(classify_global_propagate_circt_lec_raw "$run_dir" "$lec_log")
  else
    printf "error\t-1\n"
    return
  fi
  final_rc="$fallback_rc"
  if [[ "$final_rc" -eq 0 ]]; then
    final_rc="$primary_rc"
  fi
  {
    printf "# chain_mode=%s primary=%s primary_state=%s primary_rc=%s\n" \
      "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" "$primary_tool" "$primary_state" "$primary_rc"
    if [[ "$primary_tool" == "lec" ]]; then
      cat "$lec_log"
    else
      cat "$bmc_log"
    fi
    printf "\n# chain_fallback=%s fallback_state=%s fallback_rc=%s\n" \
      "$fallback_tool" "$fallback_state" "$fallback_rc"
    if [[ "$fallback_tool" == "bmc" ]]; then
      cat "$bmc_log"
    else
      cat "$lec_log"
    fi
  } > "$log_file"

  if [[ "$fallback_tool" == "bmc" ]]; then
    case "$fallback_state" in
      equal) printf "not_propagated\t%s\n" "$final_rc" ;;
      different|unknown) printf "propagated\t%s\n" "$final_rc" ;;
      *) printf "error\t%s\n" "$final_rc" ;;
    esac
  else
    case "$fallback_state" in
      eq) printf "not_propagated\t%s\n" "$final_rc" ;;
      neq|unknown) printf "propagated\t%s\n" "$final_rc" ;;
      *) printf "error\t%s\n" "$final_rc" ;;
    esac
  fi
}

run_test_and_classify() {
  local run_dir="$1"
  local test_id="$2"
  local log_file="$3"
  local result_file="${TEST_RESULT_FILE[$test_id]}"
  local kill_pattern="${TEST_KILL_PATTERN[$test_id]}"
  local survive_pattern="${TEST_SURVIVE_PATTERN[$test_id]}"
  local rc=0

  rc=0
  if ! run_command "$run_dir" "${TEST_CMD[$test_id]}" "$log_file"; then
    rc=$?
  fi

  if [[ ! -f "$run_dir/$result_file" ]]; then
    printf "error\t%s\t%s\tmissing_result_file\n" "$rc" "$result_file"
    return
  fi
  if grep -Eq "$kill_pattern" "$run_dir/$result_file"; then
    printf "detected\t%s\t%s\tmatched_kill_pattern\n" "$rc" "$result_file"
    return
  fi
  if grep -Eq "$survive_pattern" "$run_dir/$result_file"; then
    printf "survived\t%s\t%s\tmatched_survive_pattern\n" "$rc" "$result_file"
    return
  fi
  printf "error\t%s\t%s\tno_pattern_match\n" "$rc" "$result_file"
}

process_mutation() {
  local mutation_id="$1"
  local mutation_spec="$2"
  local mutation_dir="${WORK_DIR}/mutations/${mutation_id}"
  local mutant_design="${mutation_dir}/mutant.${MUTANT_FORMAT}"
  local summary_local="${mutation_dir}/mutant_summary.tsv"
  local pair_local="${mutation_dir}/pair_qualification.tsv"
  local results_local="${mutation_dir}/results.tsv"
  local meta_local="${mutation_dir}/local_metrics.tsv"

  if [[ "$RESUME" -eq 1 && -f "$summary_local" && -f "$pair_local" && -f "$results_local" && -f "$meta_local" ]]; then
    return 0
  fi

  mkdir -p "$mutation_dir"
  : > "$pair_local"
  : > "$results_local"

  local activated_any=0
  local propagated_any=0
  local detected_by_test="-"
  local relevant_pairs=0
  local reused_pairs=0
  local mutant_class="not_activated"
  local mutation_errors=0
  local create_rc=0
  local test_id=""
  local test_dir=""
  local activate_state=""
  local activate_rc=""
  local propagate_state=""
  local propagate_rc=""
  local test_result=""
  local test_exit=""
  local result_file=""
  local test_note=""
  local reuse_key=""
  local reuse_hit=0
  local run_note="run_detection"
  local hinted_test=""
  local hinted_mutant=0
  local hint_hit=0
  local global_filter_state=""
  local global_filter_rc="-1"
  local global_filter_note=""
  local global_filtered_not_propagated=0
  local reused_global_filter=0
  local chain_lec_unknown_fallback=0
  local chain_bmc_resolved_not_propagated=0
  local chain_bmc_unknown_fallback=0
  local chain_lec_resolved_not_propagated=0
  local chain_consensus_not_propagated=0
  local chain_consensus_disagreement=0
  local chain_consensus_error=0
  local -a ORDERED_TESTS=()

  printf "1 %s\n" "$mutation_spec" > "$mutation_dir/input.txt"

  set +e
  "$CREATE_MUTATED_SCRIPT" \
    -i "$mutation_dir/input.txt" \
    -o "$mutant_design" \
    -d "$DESIGN" > "$mutation_dir/create_mutated.log" 2>&1
  create_rc=$?
  set -e
  if [[ "$create_rc" -ne 0 ]]; then
    mutation_errors=$((mutation_errors + 1))
    mutant_class="not_activated+error"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$mutation_id" "$mutant_class" "0" "-" "$mutant_design" "$mutation_spec" > "$summary_local"
    {
      printf "errors\t%s\n" "$mutation_errors"
      printf "reused_pairs\t0\n"
      printf "hinted_mutant\t0\n"
      printf "hint_hit\t0\n"
      printf "global_filtered_not_propagated\t0\n"
      printf "reused_global_filter\t0\n"
      printf "chain_lec_unknown_fallback\t0\n"
      printf "chain_bmc_resolved_not_propagated\t0\n"
      printf "chain_bmc_unknown_fallback\t0\n"
      printf "chain_lec_resolved_not_propagated\t0\n"
      printf "chain_consensus_not_propagated\t0\n"
      printf "chain_consensus_disagreement\t0\n"
      printf "chain_consensus_error\t0\n"
    } > "$meta_local"
    return 0
  fi

  export BASELINE=0
  export ORIG_DESIGN="$DESIGN"
  export MUTANT_DESIGN="$mutant_design"
  export MUTATION_ID="$mutation_id"
  export MUTATION_SPEC="$mutation_spec"
  export MUTATION_WORKDIR="$mutation_dir"

  if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" || -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED" || -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED" || -n "$FORMAL_GLOBAL_PROPAGATE_CMD" ]]; then
    if [[ -n "$REUSE_PAIR_FILE" && -n "${REUSE_GLOBAL_FILTER_STATE[$mutation_id]+x}" ]]; then
      global_filter_state="${REUSE_GLOBAL_FILTER_STATE[$mutation_id]}"
      global_filter_rc="${REUSE_GLOBAL_FILTER_RC[$mutation_id]:--1}"
      reused_global_filter=1
      chain_lec_unknown_fallback="${REUSE_GLOBAL_CHAIN_LEC_UNKNOWN_FALLBACK[$mutation_id]:-0}"
      chain_bmc_resolved_not_propagated="${REUSE_GLOBAL_CHAIN_BMC_RESOLVED_NOT_PROPAGATED[$mutation_id]:-0}"
      chain_bmc_unknown_fallback="${REUSE_GLOBAL_CHAIN_BMC_UNKNOWN_FALLBACK[$mutation_id]:-0}"
      chain_lec_resolved_not_propagated="${REUSE_GLOBAL_CHAIN_LEC_RESOLVED_NOT_PROPAGATED[$mutation_id]:-0}"
      chain_consensus_not_propagated="${REUSE_GLOBAL_CHAIN_CONSENSUS_NOT_PROPAGATED[$mutation_id]:-0}"
      chain_consensus_disagreement="${REUSE_GLOBAL_CHAIN_CONSENSUS_DISAGREEMENT[$mutation_id]:-0}"
      chain_consensus_error="${REUSE_GLOBAL_CHAIN_CONSENSUS_ERROR[$mutation_id]:-0}"
      {
        printf "# reused_global_filter=1 source=%s state=%s rc=%s\n" "$REUSE_PAIR_SOURCE" "$global_filter_state" "$global_filter_rc"
      } > "$mutation_dir/global_propagate.log"
    elif [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" ]]; then
      read -r global_filter_state global_filter_rc < <(classify_global_propagate_circt_chain "$mutation_dir" "$mutation_dir/global_propagate.log")
    elif [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_LEC_RESOLVED" ]]; then
      read -r global_filter_state global_filter_rc < <(classify_global_propagate_circt_lec "$mutation_dir" "$mutation_dir/global_propagate.log")
    elif [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_BMC_RESOLVED" ]]; then
      read -r global_filter_state global_filter_rc < <(classify_global_propagate_circt_bmc "$mutation_dir" "$mutation_dir/global_propagate.log")
    elif [[ -n "$FORMAL_GLOBAL_PROPAGATE_CMD" ]]; then
      read -r global_filter_state global_filter_rc < <(classify_propagate_cmd "$mutation_dir" "$mutation_dir/global_propagate.log" "$FORMAL_GLOBAL_PROPAGATE_CMD")
    fi
  fi
  if [[ "$reused_global_filter" -eq 0 && -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" ]]; then
    if [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "lec-then-bmc" ]]; then
      if grep -Eq '^# chain_mode=lec-then-bmc primary=lec primary_state=unknown ' "$mutation_dir/global_propagate.log"; then
        chain_lec_unknown_fallback=1
      fi
      if grep -Eq '^# chain_fallback=bmc fallback_state=equal ' "$mutation_dir/global_propagate.log"; then
        chain_bmc_resolved_not_propagated=1
      fi
    elif [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "bmc-then-lec" ]]; then
      if grep -Eq '^# chain_mode=bmc-then-lec primary=bmc primary_state=unknown ' "$mutation_dir/global_propagate.log"; then
        chain_bmc_unknown_fallback=1
      fi
      if grep -Eq '^# chain_fallback=lec fallback_state=eq ' "$mutation_dir/global_propagate.log"; then
        chain_lec_resolved_not_propagated=1
      fi
    elif [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "consensus" ]]; then
      if grep -Eq '^# chain_mode=consensus lec_state=eq ' "$mutation_dir/global_propagate.log" && \
         grep -Eq '^# chain_consensus_bmc_state=equal ' "$mutation_dir/global_propagate.log"; then
        chain_consensus_not_propagated=1
      fi
      if (grep -Eq '^# chain_mode=consensus lec_state=eq ' "$mutation_dir/global_propagate.log" && \
          grep -Eq '^# chain_consensus_bmc_state=different ' "$mutation_dir/global_propagate.log") || \
         (grep -Eq '^# chain_mode=consensus lec_state=neq ' "$mutation_dir/global_propagate.log" && \
          grep -Eq '^# chain_consensus_bmc_state=equal ' "$mutation_dir/global_propagate.log"); then
        chain_consensus_disagreement=1
      fi
    fi
  fi
  if [[ -n "$global_filter_state" ]]; then
    case "$global_filter_state" in
      not_propagated)
        global_filtered_not_propagated=1
        activated_any=1
        if [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "consensus" ]]; then
          chain_consensus_not_propagated=1
        fi
        global_filter_note="$([[ "$reused_global_filter" -eq 1 ]] && printf "global_filter_cached_not_propagated" || printf "global_filter_not_propagated")"
        ;;
      propagated)
        global_filter_note="$([[ "$reused_global_filter" -eq 1 ]] && printf "global_filter_cached_propagated" || printf "global_filter_propagated")"
        ;;
      error)
        mutation_errors=$((mutation_errors + 1))
        if [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "consensus" ]]; then
          chain_consensus_error=1
        fi
        global_filter_note="global_filter_error_continue"
        ;;
      *)
        mutation_errors=$((mutation_errors + 1))
        if [[ "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" == "consensus" ]]; then
          chain_consensus_error=1
        fi
        global_filter_note="global_filter_invalid_state"
        ;;
    esac
    if [[ -n "$FORMAL_GLOBAL_PROPAGATE_CIRCT_CHAIN" ]]; then
      global_filter_note="${global_filter_note};chain_lec_unknown_fallback=${chain_lec_unknown_fallback};chain_bmc_resolved_not_propagated=${chain_bmc_resolved_not_propagated};chain_bmc_unknown_fallback=${chain_bmc_unknown_fallback};chain_lec_resolved_not_propagated=${chain_lec_resolved_not_propagated};chain_consensus_not_propagated=${chain_consensus_not_propagated};chain_consensus_disagreement=${chain_consensus_disagreement};chain_consensus_error=${chain_consensus_error}"
    fi
    if [[ "$global_filter_state" == "not_propagated" || "$global_filter_state" == "propagated" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "-" "activated" "$global_filter_state" "-1" "$global_filter_rc" "$global_filter_note" >> "$pair_local"
    else
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "-" "activated" "error" "-1" "$global_filter_rc" "$global_filter_note" >> "$pair_local"
    fi
  fi

  hinted_test="${REUSE_DETECTED_TEST[$mutation_id]:-}"
  if [[ -n "$hinted_test" ]]; then
    hinted_mutant=1
    ORDERED_TESTS+=("$hinted_test")
  fi
  for test_id in "${TEST_ORDER[@]}"; do
    if [[ -n "$hinted_test" && "$test_id" == "$hinted_test" ]]; then
      continue
    fi
    ORDERED_TESTS+=("$test_id")
  done

  if [[ "$global_filtered_not_propagated" -eq 0 ]]; then
    for test_id in "${ORDERED_TESTS[@]}"; do
      export TEST_ID="$test_id"
      test_dir="${mutation_dir}/${test_id}"
      mkdir -p "$test_dir"
      reuse_key="${mutation_id}"$'\t'"${test_id}"
      reuse_hit=0
      run_note="run_detection"

      if [[ -n "$REUSE_PAIR_FILE" && -n "${REUSE_ACTIVATION[$reuse_key]+x}" ]]; then
        activate_state="${REUSE_ACTIVATION[$reuse_key]}"
        activate_rc="${REUSE_ACTIVATE_EXIT[$reuse_key]}"
        propagate_state="${REUSE_PROPAGATION[$reuse_key]}"
        propagate_rc="${REUSE_PROPAGATE_EXIT[$reuse_key]}"
        reuse_hit=1
        reused_pairs=$((reused_pairs + 1))
        run_note="cached_run_detection"
      else
        read -r activate_state activate_rc < <(classify_activate "$test_dir" "$test_dir/activate.log")
      fi
      if [[ -n "$hinted_test" && "$test_id" == "$hinted_test" ]]; then
        if [[ "$run_note" == "cached_run_detection" ]]; then
          run_note="hinted_cached_run_detection"
        else
          run_note="hinted_run_detection"
        fi
      fi

      if [[ "$activate_state" == "error" ]]; then
        mutation_errors=$((mutation_errors + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$mutation_id" "$test_id" "error" "-" "$activate_rc" "-1" "activation_error" >> "$pair_local"
        continue
      fi
      if [[ "$activate_state" == "not_activated" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$mutation_id" "$test_id" "not_activated" "-" "$activate_rc" "-1" \
          "$([[ "$reuse_hit" -eq 1 ]] && printf "cached_no_activation" || printf "skipped_no_activation")" >> "$pair_local"
        continue
      fi

      activated_any=1
      if [[ "$reuse_hit" -eq 0 ]]; then
        read -r propagate_state propagate_rc < <(classify_propagate "$test_dir" "$test_dir/propagate.log")
      fi
      if [[ "$propagate_state" == "error" ]]; then
        mutation_errors=$((mutation_errors + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$mutation_id" "$test_id" "activated" "error" "$activate_rc" "$propagate_rc" "propagation_error" >> "$pair_local"
        continue
      fi
      if [[ "$propagate_state" != "not_propagated" && "$propagate_state" != "propagated" ]]; then
        mutation_errors=$((mutation_errors + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$mutation_id" "$test_id" "activated" "error" "$activate_rc" "$propagate_rc" "invalid_propagation_state" >> "$pair_local"
        continue
      fi
      if [[ "$propagate_state" == "not_propagated" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$mutation_id" "$test_id" "activated" "not_propagated" "$activate_rc" "$propagate_rc" \
          "$([[ "$reuse_hit" -eq 1 ]] && printf "cached_no_propagation" || printf "skipped_no_propagation")" >> "$pair_local"
        continue
      fi

      propagated_any=1
      relevant_pairs=$((relevant_pairs + 1))
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "activated" "propagated" "$activate_rc" "$propagate_rc" "$run_note" >> "$pair_local"

      read -r test_result test_exit result_file test_note < <(run_test_and_classify "$test_dir" "$test_id" "$test_dir/test.log")
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "$test_result" "$test_exit" "$result_file" "$test_note" >> "$results_local"

      if [[ "$test_result" == "error" ]]; then
        mutation_errors=$((mutation_errors + 1))
        continue
      fi
      if [[ "$test_result" == "detected" ]]; then
        mutant_class="detected"
        detected_by_test="$test_id"
        if [[ -n "$hinted_test" && "$test_id" == "$hinted_test" ]]; then
          hint_hit=1
        fi
        break
      fi
    done
  fi

  if [[ "$mutant_class" != "detected" ]]; then
    if [[ "$propagated_any" -eq 1 ]]; then
      mutant_class="propagated_not_detected"
    elif [[ "$activated_any" -eq 1 ]]; then
      mutant_class="not_propagated"
    else
      mutant_class="not_activated"
    fi
  fi
  if [[ "$mutation_errors" -gt 0 ]]; then
    mutant_class="${mutant_class}+error"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mutation_id" "$mutant_class" "$relevant_pairs" "$detected_by_test" "$mutant_design" "$mutation_spec" > "$summary_local"
  {
    printf "errors\t%s\n" "$mutation_errors"
    printf "reused_pairs\t%s\n" "$reused_pairs"
    printf "hinted_mutant\t%s\n" "$hinted_mutant"
    printf "hint_hit\t%s\n" "$hint_hit"
    printf "global_filtered_not_propagated\t%s\n" "$global_filtered_not_propagated"
    printf "reused_global_filter\t%s\n" "$reused_global_filter"
    printf "chain_lec_unknown_fallback\t%s\n" "$chain_lec_unknown_fallback"
    printf "chain_bmc_resolved_not_propagated\t%s\n" "$chain_bmc_resolved_not_propagated"
    printf "chain_bmc_unknown_fallback\t%s\n" "$chain_bmc_unknown_fallback"
    printf "chain_lec_resolved_not_propagated\t%s\n" "$chain_lec_resolved_not_propagated"
    printf "chain_consensus_not_propagated\t%s\n" "$chain_consensus_not_propagated"
    printf "chain_consensus_disagreement\t%s\n" "$chain_consensus_disagreement"
    printf "chain_consensus_error\t%s\n" "$chain_consensus_error"
  } > "$meta_local"
}

load_tests_manifest
load_mutations
if [[ "${#MUTATION_IDS[@]}" -eq 0 ]]; then
  echo "No usable mutations loaded from: $MUTATIONS_FILE" >&2
  exit 1
fi
build_reuse_compat_hash
resolve_cache_reuse_inputs
if [[ -n "$REUSE_PAIR_FILE" ]]; then
  rc=0
  validate_reuse_file_compat "pair" "$REUSE_PAIR_FILE" || rc=$?
  case "$rc" in
    0) ;;
    1) exit 1 ;;
    2)
      REUSE_PAIR_FILE=""
      REUSE_PAIR_SOURCE="none"
      ;;
    *) exit 1 ;;
  esac
fi
if [[ -n "$REUSE_SUMMARY_FILE" ]]; then
  rc=0
  validate_reuse_file_compat "summary" "$REUSE_SUMMARY_FILE" || rc=$?
  case "$rc" in
    0) ;;
    1) exit 1 ;;
    2)
      REUSE_SUMMARY_FILE=""
      REUSE_SUMMARY_SOURCE="none"
      ;;
    *) exit 1 ;;
  esac
fi
load_reuse_pairs
load_reuse_summary

if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  baseline_dir="${WORK_DIR}/baseline"
  mkdir -p "$baseline_dir"
  export BASELINE=1
  export ORIG_DESIGN="$DESIGN"
  export MUTANT_DESIGN="$DESIGN"
  export MUTATION_ID="baseline"
  export MUTATION_SPEC="baseline"
  export MUTATION_WORKDIR="$baseline_dir"
  for test_id in "${TEST_ORDER[@]}"; do
    test_dir="${baseline_dir}/${test_id}"
    mkdir -p "$test_dir"
    export TEST_ID="$test_id"
    read -r baseline_result _ _ baseline_note < <(run_test_and_classify "$test_dir" "$test_id" "$test_dir/test.log")
    if [[ "$baseline_result" != "survived" ]]; then
      echo "Baseline test '${test_id}' failed sanity check (${baseline_result}: ${baseline_note})." >&2
      exit 1
    fi
  done
fi

worker_fail=0
if [[ "$JOBS" -le 1 ]]; then
  for i in "${!MUTATION_IDS[@]}"; do
    process_mutation "${MUTATION_IDS[$i]}" "${MUTATION_SPECS[$i]}"
  done
else
  active_jobs=0
  for i in "${!MUTATION_IDS[@]}"; do
    process_mutation "${MUTATION_IDS[$i]}" "${MUTATION_SPECS[$i]}" &
    active_jobs=$((active_jobs + 1))
    if [[ "$active_jobs" -ge "$JOBS" ]]; then
      if ! wait -n; then
        worker_fail=1
      fi
      active_jobs=$((active_jobs - 1))
    fi
  done
  while [[ "$active_jobs" -gt 0 ]]; do
    if ! wait -n; then
      worker_fail=1
    fi
    active_jobs=$((active_jobs - 1))
  done
fi

printf "mutation_id\tclassification\trelevant_pairs\tdetected_by_test\tmutant_design\tmutation_spec\n" > "$SUMMARY_FILE"
printf "mutation_id\ttest_id\tactivation\tpropagation\tactivate_exit\tpropagate_exit\tnote\n" > "$PAIR_FILE"
printf "mutation_id\ttest_id\tresult\ttest_exit\tresult_file\tnote\n" > "$RESULTS_FILE"

errors="$MALFORMED_MUTATION_LINES"
if [[ "$worker_fail" -ne 0 ]]; then
  errors=$((errors + 1))
fi

total_mutants=0
count_not_activated=0
count_not_propagated=0
count_detected=0
count_propagated_not_detected=0
count_reused_pairs=0
count_reused_global_filters=0
count_hinted_mutants=0
count_hint_hits=0
count_global_filtered_not_propagated=0
count_chain_lec_unknown_fallbacks=0
count_chain_bmc_resolved_not_propagated=0
count_chain_bmc_unknown_fallbacks=0
count_chain_lec_resolved_not_propagated=0
count_chain_consensus_not_propagated=0
count_chain_consensus_disagreement=0
count_chain_consensus_error=0

for i in "${!MUTATION_IDS[@]}"; do
  mutation_id="${MUTATION_IDS[$i]}"
  mutation_dir="${WORK_DIR}/mutations/${mutation_id}"
  summary_local="${mutation_dir}/mutant_summary.tsv"
  pair_local="${mutation_dir}/pair_qualification.tsv"
  results_local="${mutation_dir}/results.tsv"
  meta_local="${mutation_dir}/local_metrics.tsv"

  if [[ ! -f "$summary_local" ]]; then
    errors=$((errors + 1))
    continue
  fi

  total_mutants=$((total_mutants + 1))
  cat "$summary_local" >> "$SUMMARY_FILE"
  [[ -f "$pair_local" ]] && cat "$pair_local" >> "$PAIR_FILE"
  [[ -f "$results_local" ]] && cat "$results_local" >> "$RESULTS_FILE"

  classification="$(awk -F$'\t' 'NR==1{print $2}' "$summary_local")"
  base_class="${classification%%+*}"
  case "$base_class" in
    detected) count_detected=$((count_detected + 1)) ;;
    propagated_not_detected) count_propagated_not_detected=$((count_propagated_not_detected + 1)) ;;
    not_propagated) count_not_propagated=$((count_not_propagated + 1)) ;;
    not_activated) count_not_activated=$((count_not_activated + 1)) ;;
    *) ;;
  esac

  local_errors=0
  if [[ -f "$meta_local" ]]; then
    local_errors="$(awk -F$'\t' '$1=="errors"{print $2}' "$meta_local" | head -n1)"
    local_errors="${local_errors:-0}"
  fi
  if [[ "$local_errors" =~ ^[0-9]+$ ]]; then
    errors=$((errors + local_errors))
  else
    errors=$((errors + 1))
  fi

  local_reused=0
  if [[ -f "$meta_local" ]]; then
    local_reused="$(awk -F$'\t' '$1=="reused_pairs"{print $2}' "$meta_local" | head -n1)"
    local_reused="${local_reused:-0}"
  fi
  if [[ "$local_reused" =~ ^[0-9]+$ ]]; then
    count_reused_pairs=$((count_reused_pairs + local_reused))
  fi

  local_reused_global_filter=0
  if [[ -f "$meta_local" ]]; then
    local_reused_global_filter="$(awk -F$'\t' '$1=="reused_global_filter"{print $2}' "$meta_local" | head -n1)"
    local_reused_global_filter="${local_reused_global_filter:-0}"
  fi
  if [[ "$local_reused_global_filter" =~ ^[0-9]+$ ]]; then
    count_reused_global_filters=$((count_reused_global_filters + local_reused_global_filter))
  fi

  local_hinted=0
  if [[ -f "$meta_local" ]]; then
    local_hinted="$(awk -F$'\t' '$1=="hinted_mutant"{print $2}' "$meta_local" | head -n1)"
    local_hinted="${local_hinted:-0}"
  fi
  if [[ "$local_hinted" =~ ^[0-9]+$ ]]; then
    count_hinted_mutants=$((count_hinted_mutants + local_hinted))
  fi

  local_hint_hit=0
  if [[ -f "$meta_local" ]]; then
    local_hint_hit="$(awk -F$'\t' '$1=="hint_hit"{print $2}' "$meta_local" | head -n1)"
    local_hint_hit="${local_hint_hit:-0}"
  fi
  if [[ "$local_hint_hit" =~ ^[0-9]+$ ]]; then
    count_hint_hits=$((count_hint_hits + local_hint_hit))
  fi

  local_global_filtered=0
  if [[ -f "$meta_local" ]]; then
    local_global_filtered="$(awk -F$'\t' '$1=="global_filtered_not_propagated"{print $2}' "$meta_local" | head -n1)"
    local_global_filtered="${local_global_filtered:-0}"
  fi
  if [[ "$local_global_filtered" =~ ^[0-9]+$ ]]; then
    count_global_filtered_not_propagated=$((count_global_filtered_not_propagated + local_global_filtered))
  fi

  local_chain_lec_unknown_fallback=0
  if [[ -f "$meta_local" ]]; then
    local_chain_lec_unknown_fallback="$(awk -F$'\t' '$1=="chain_lec_unknown_fallback"{print $2}' "$meta_local" | head -n1)"
    local_chain_lec_unknown_fallback="${local_chain_lec_unknown_fallback:-0}"
  fi
  if [[ "$local_chain_lec_unknown_fallback" =~ ^[0-9]+$ ]]; then
    count_chain_lec_unknown_fallbacks=$((count_chain_lec_unknown_fallbacks + local_chain_lec_unknown_fallback))
  fi

  local_chain_bmc_resolved_not_propagated=0
  if [[ -f "$meta_local" ]]; then
    local_chain_bmc_resolved_not_propagated="$(awk -F$'\t' '$1=="chain_bmc_resolved_not_propagated"{print $2}' "$meta_local" | head -n1)"
    local_chain_bmc_resolved_not_propagated="${local_chain_bmc_resolved_not_propagated:-0}"
  fi
  if [[ "$local_chain_bmc_resolved_not_propagated" =~ ^[0-9]+$ ]]; then
    count_chain_bmc_resolved_not_propagated=$((count_chain_bmc_resolved_not_propagated + local_chain_bmc_resolved_not_propagated))
  fi

  local_chain_bmc_unknown_fallback=0
  if [[ -f "$meta_local" ]]; then
    local_chain_bmc_unknown_fallback="$(awk -F$'\t' '$1=="chain_bmc_unknown_fallback"{print $2}' "$meta_local" | head -n1)"
    local_chain_bmc_unknown_fallback="${local_chain_bmc_unknown_fallback:-0}"
  fi
  if [[ "$local_chain_bmc_unknown_fallback" =~ ^[0-9]+$ ]]; then
    count_chain_bmc_unknown_fallbacks=$((count_chain_bmc_unknown_fallbacks + local_chain_bmc_unknown_fallback))
  fi

  local_chain_lec_resolved_not_propagated=0
  if [[ -f "$meta_local" ]]; then
    local_chain_lec_resolved_not_propagated="$(awk -F$'\t' '$1=="chain_lec_resolved_not_propagated"{print $2}' "$meta_local" | head -n1)"
    local_chain_lec_resolved_not_propagated="${local_chain_lec_resolved_not_propagated:-0}"
  fi
  if [[ "$local_chain_lec_resolved_not_propagated" =~ ^[0-9]+$ ]]; then
    count_chain_lec_resolved_not_propagated=$((count_chain_lec_resolved_not_propagated + local_chain_lec_resolved_not_propagated))
  fi

  local_chain_consensus_not_propagated=0
  if [[ -f "$meta_local" ]]; then
    local_chain_consensus_not_propagated="$(awk -F$'\t' '$1=="chain_consensus_not_propagated"{print $2}' "$meta_local" | head -n1)"
    local_chain_consensus_not_propagated="${local_chain_consensus_not_propagated:-0}"
  fi
  if [[ "$local_chain_consensus_not_propagated" =~ ^[0-9]+$ ]]; then
    count_chain_consensus_not_propagated=$((count_chain_consensus_not_propagated + local_chain_consensus_not_propagated))
  fi

  local_chain_consensus_disagreement=0
  if [[ -f "$meta_local" ]]; then
    local_chain_consensus_disagreement="$(awk -F$'\t' '$1=="chain_consensus_disagreement"{print $2}' "$meta_local" | head -n1)"
    local_chain_consensus_disagreement="${local_chain_consensus_disagreement:-0}"
  fi
  if [[ "$local_chain_consensus_disagreement" =~ ^[0-9]+$ ]]; then
    count_chain_consensus_disagreement=$((count_chain_consensus_disagreement + local_chain_consensus_disagreement))
  fi

  local_chain_consensus_error=0
  if [[ -f "$meta_local" ]]; then
    local_chain_consensus_error="$(awk -F$'\t' '$1=="chain_consensus_error"{print $2}' "$meta_local" | head -n1)"
    local_chain_consensus_error="${local_chain_consensus_error:-0}"
  fi
  if [[ "$local_chain_consensus_error" =~ ^[0-9]+$ ]]; then
    count_chain_consensus_error=$((count_chain_consensus_error + local_chain_consensus_error))
  fi
done

count_relevant=$((count_detected + count_propagated_not_detected))
coverage_pct="100.00"
if [[ "$count_relevant" -gt 0 ]]; then
  coverage_pct="$(awk -v d="$count_detected" -v r="$count_relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
fi

{
  printf "bucket\tcount\taction\n"
  printf "not_activated\t%s\tstrengthen_stimulus\n" "$count_not_activated"
  printf "not_propagated\t%s\timprove_observability_or_outputs\n" "$count_not_propagated"
  printf "propagated_not_detected\t%s\timprove_checkers_scoreboards_assertions\n" "$count_propagated_not_detected"
  printf "detected\t%s\tretain_as_detection_evidence\n" "$count_detected"
} > "$IMPROVEMENT_FILE"

gate_status="PASS"
exit_code=0
if [[ "$errors" -gt 0 && "$FAIL_ON_ERRORS" -eq 1 ]]; then
  gate_status="FAIL_ERRORS"
  exit_code=1
elif [[ "$count_propagated_not_detected" -gt 0 && "$FAIL_ON_UNDETECTED" -eq 1 ]]; then
  gate_status="FAIL_UNDETECTED"
  exit_code=3
elif [[ -n "$COVERAGE_THRESHOLD" ]]; then
  threshold_fail=0
  if awk -v cov="$coverage_pct" -v thr="$COVERAGE_THRESHOLD" 'BEGIN { exit (cov + 0.0 < thr + 0.0 ? 0 : 1) }'; then
    threshold_fail=1
  fi
  if [[ "$threshold_fail" -eq 1 ]]; then
    gate_status="FAIL_THRESHOLD"
    exit_code=2
  fi
fi

threshold_json="null"
if [[ -n "$COVERAGE_THRESHOLD" ]]; then
  threshold_json="$COVERAGE_THRESHOLD"
fi

if [[ "$REUSE_CACHE_MODE" == "off" ]]; then
  REUSE_CACHE_WRITE_STATUS="disabled"
elif [[ "$REUSE_CACHE_MODE" == "read" ]]; then
  REUSE_CACHE_WRITE_STATUS="read_only"
elif [[ "$errors" -gt 0 ]]; then
  REUSE_CACHE_WRITE_STATUS="skipped_errors"
else
  REUSE_CACHE_WRITE_STATUS="pending_write"
fi

write_reuse_manifest "$REUSE_MANIFEST_FILE" "run"
write_reuse_manifest "${PAIR_FILE}.manifest.json" "pair_qualification"
write_reuse_manifest "${SUMMARY_FILE}.manifest.json" "summary"

if [[ "$REUSE_CACHE_WRITE_STATUS" == "pending_write" ]]; then
  if ! publish_reuse_cache; then
    REUSE_CACHE_WRITE_STATUS="write_error"
  fi
fi
if [[ "$REUSE_CACHE_WRITE_STATUS" == "write_error" ]]; then
  errors=$((errors + 1))
  if [[ "$FAIL_ON_ERRORS" -eq 1 ]]; then
    gate_status="FAIL_ERRORS"
    exit_code=1
  fi
fi

{
  printf "metric\tvalue\n"
  printf "total_mutants\t%s\n" "$total_mutants"
  printf "relevant_mutants\t%s\n" "$count_relevant"
  printf "detected_mutants\t%s\n" "$count_detected"
  printf "propagated_not_detected_mutants\t%s\n" "$count_propagated_not_detected"
  printf "not_propagated_mutants\t%s\n" "$count_not_propagated"
  printf "not_activated_mutants\t%s\n" "$count_not_activated"
  printf "reused_pairs\t%s\n" "$count_reused_pairs"
  printf "reused_global_filters\t%s\n" "$count_reused_global_filters"
  printf "hinted_mutants\t%s\n" "$count_hinted_mutants"
  printf "hint_hits\t%s\n" "$count_hint_hits"
  printf "global_filtered_not_propagated_mutants\t%s\n" "$count_global_filtered_not_propagated"
  printf "chain_lec_unknown_fallbacks\t%s\n" "$count_chain_lec_unknown_fallbacks"
  printf "chain_bmc_resolved_not_propagated_mutants\t%s\n" "$count_chain_bmc_resolved_not_propagated"
  printf "chain_bmc_unknown_fallbacks\t%s\n" "$count_chain_bmc_unknown_fallbacks"
  printf "chain_lec_resolved_not_propagated_mutants\t%s\n" "$count_chain_lec_resolved_not_propagated"
  printf "chain_consensus_not_propagated_mutants\t%s\n" "$count_chain_consensus_not_propagated"
  printf "chain_consensus_disagreement_mutants\t%s\n" "$count_chain_consensus_disagreement"
  printf "chain_consensus_error_mutants\t%s\n" "$count_chain_consensus_error"
  printf "reuse_pair_source\t%s\n" "$REUSE_PAIR_SOURCE"
  printf "reuse_summary_source\t%s\n" "$REUSE_SUMMARY_SOURCE"
  printf "reuse_cache_mode\t%s\n" "$REUSE_CACHE_MODE"
  printf "reuse_cache_write_status\t%s\n" "$REUSE_CACHE_WRITE_STATUS"
  printf "errors\t%s\n" "$errors"
  printf "mutation_coverage_percent\t%s\n" "$coverage_pct"
} > "$METRICS_FILE"

cat > "$SUMMARY_JSON_FILE" <<EOF
{
  "total_mutants": $total_mutants,
  "relevant_mutants": $count_relevant,
  "detected_mutants": $count_detected,
  "propagated_not_detected_mutants": $count_propagated_not_detected,
  "not_propagated_mutants": $count_not_propagated,
  "not_activated_mutants": $count_not_activated,
  "reused_pairs": $count_reused_pairs,
  "reused_global_filters": $count_reused_global_filters,
  "hinted_mutants": $count_hinted_mutants,
  "hint_hits": $count_hint_hits,
  "global_filtered_not_propagated_mutants": $count_global_filtered_not_propagated,
  "chain_lec_unknown_fallbacks": $count_chain_lec_unknown_fallbacks,
  "chain_bmc_resolved_not_propagated_mutants": $count_chain_bmc_resolved_not_propagated,
  "chain_bmc_unknown_fallbacks": $count_chain_bmc_unknown_fallbacks,
  "chain_lec_resolved_not_propagated_mutants": $count_chain_lec_resolved_not_propagated,
  "chain_consensus_not_propagated_mutants": $count_chain_consensus_not_propagated,
  "chain_consensus_disagreement_mutants": $count_chain_consensus_disagreement,
  "chain_consensus_error_mutants": $count_chain_consensus_error,
  "reuse_pair_source": "$(json_escape "$REUSE_PAIR_SOURCE")",
  "reuse_summary_source": "$(json_escape "$REUSE_SUMMARY_SOURCE")",
  "reuse_cache_mode": "$(json_escape "$REUSE_CACHE_MODE")",
  "reuse_cache_write_status": "$(json_escape "$REUSE_CACHE_WRITE_STATUS")",
  "reuse_cache_dir": "$(json_escape "$REUSE_CACHE_DIR")",
  "reuse_cache_entry_dir": "$(json_escape "$REUSE_CACHE_ENTRY_DIR")",
  "errors": $errors,
  "mutation_coverage_percent": $coverage_pct,
  "coverage_threshold_percent": $threshold_json,
  "gate_status": "$gate_status",
  "fail_on_undetected": $FAIL_ON_UNDETECTED,
  "fail_on_errors": $FAIL_ON_ERRORS
}
EOF

echo "Mutation coverage summary: total=${total_mutants} relevant=${count_relevant} detected=${count_detected} propagated_not_detected=${count_propagated_not_detected} not_propagated=${count_not_propagated} not_activated=${count_not_activated} reused_pairs=${count_reused_pairs} reused_global_filters=${count_reused_global_filters} hinted_mutants=${count_hinted_mutants} hint_hits=${count_hint_hits} global_filtered_not_propagated=${count_global_filtered_not_propagated} chain_lec_unknown_fallbacks=${count_chain_lec_unknown_fallbacks} chain_bmc_resolved_not_propagated=${count_chain_bmc_resolved_not_propagated} chain_bmc_unknown_fallbacks=${count_chain_bmc_unknown_fallbacks} chain_lec_resolved_not_propagated=${count_chain_lec_resolved_not_propagated} chain_consensus_not_propagated=${count_chain_consensus_not_propagated} chain_consensus_disagreement=${count_chain_consensus_disagreement} chain_consensus_error=${count_chain_consensus_error} errors=${errors} coverage=${coverage_pct}%"
echo "Gate status: ${gate_status}"
echo "Summary: ${SUMMARY_FILE}"
echo "Pair qualification: ${PAIR_FILE}"
echo "Results: ${RESULTS_FILE}"
echo "Metrics: ${METRICS_FILE}"
echo "Summary JSON: ${SUMMARY_JSON_FILE}"
echo "Improvement: ${IMPROVEMENT_FILE}"
echo "Reuse manifest: ${REUSE_MANIFEST_FILE}"
echo "Reuse pair source: ${REUSE_PAIR_SOURCE}"
echo "Reuse summary source: ${REUSE_SUMMARY_SOURCE}"
echo "Reuse cache mode: ${REUSE_CACHE_MODE}"
echo "Reuse cache write status: ${REUSE_CACHE_WRITE_STATUS}"
if [[ "$REUSE_CACHE_MODE" != "off" ]]; then
  echo "Reuse cache entry: ${REUSE_CACHE_ENTRY_DIR}"
fi

exit "$exit_code"
