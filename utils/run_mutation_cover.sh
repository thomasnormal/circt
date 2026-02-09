#!/usr/bin/env bash
# CIRCT mutation coverage harness with formal pre-qualification and 4-way classes.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_cover.sh [options]

Required:
  --design FILE              Original design netlist (.il/.v/.sv)
  --mutations-file FILE      Mutations file: "<id> <mutation-spec>" per line
  --tests-manifest FILE      TSV with columns:
                               test_id<TAB>run_cmd<TAB>result_file<TAB>kill_pattern<TAB>survive_pattern

Optional:
  --work-dir DIR             Output/work directory (default: ./mutation-cover-results)
  --summary-file FILE        Mutant-level classification TSV (default: <work-dir>/summary.tsv)
  --pair-file FILE           Pair-level qualification TSV (default: <work-dir>/pair_qualification.tsv)
  --results-file FILE        Pair-level detection TSV (default: <work-dir>/results.tsv)
  --metrics-file FILE        Metric-mode summary TSV (default: <work-dir>/metrics.tsv)
  --improvement-file FILE    Improvement-mode summary TSV (default: <work-dir>/improvement.tsv)
  --create-mutated-script FILE
                             Script compatible with mcy scripts/create_mutated.sh
                             (default: ~/mcy/scripts/create_mutated.sh)
  --mutant-format EXT        Mutant file extension: il|v|sv (default: il)
  --formal-activate-cmd CMD  Optional per-(test,mutant) activation classification cmd
  --formal-propagate-cmd CMD Optional per-(test,mutant) propagation classification cmd
  --mutation-limit N         Process first N mutations (default: all)
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
IMPROVEMENT_FILE=""
CREATE_MUTATED_SCRIPT="${HOME}/mcy/scripts/create_mutated.sh"
MUTANT_FORMAT="il"
FORMAL_ACTIVATE_CMD=""
FORMAL_PROPAGATE_CMD=""
MUTATION_LIMIT=0
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
    --improvement-file) IMPROVEMENT_FILE="$2"; shift 2 ;;
    --create-mutated-script) CREATE_MUTATED_SCRIPT="$2"; shift 2 ;;
    --mutant-format) MUTANT_FORMAT="$2"; shift 2 ;;
    --formal-activate-cmd) FORMAL_ACTIVATE_CMD="$2"; shift 2 ;;
    --formal-propagate-cmd) FORMAL_PROPAGATE_CMD="$2"; shift 2 ;;
    --mutation-limit) MUTATION_LIMIT="$2"; shift 2 ;;
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

if [[ -z "$DESIGN" || -z "$MUTATIONS_FILE" || -z "$TESTS_MANIFEST" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$DESIGN" ]]; then
  echo "Design file not found: $DESIGN" >&2
  exit 1
fi
if [[ ! -f "$MUTATIONS_FILE" ]]; then
  echo "Mutations file not found: $MUTATIONS_FILE" >&2
  exit 1
fi
if [[ ! -f "$TESTS_MANIFEST" ]]; then
  echo "Tests manifest not found: $TESTS_MANIFEST" >&2
  exit 1
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
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs value: $JOBS" >&2
  exit 1
fi
if [[ -n "$COVERAGE_THRESHOLD" ]] && ! [[ "$COVERAGE_THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid --coverage-threshold value: $COVERAGE_THRESHOLD" >&2
  exit 1
fi

mkdir -p "$WORK_DIR/mutations"
SUMMARY_FILE="${SUMMARY_FILE:-${WORK_DIR}/summary.tsv}"
PAIR_FILE="${PAIR_FILE:-${WORK_DIR}/pair_qualification.tsv}"
RESULTS_FILE="${RESULTS_FILE:-${WORK_DIR}/results.tsv}"
METRICS_FILE="${METRICS_FILE:-${WORK_DIR}/metrics.tsv}"
IMPROVEMENT_FILE="${IMPROVEMENT_FILE:-${WORK_DIR}/improvement.tsv}"

declare -A TEST_CMD
declare -A TEST_RESULT_FILE
declare -A TEST_KILL_PATTERN
declare -A TEST_SURVIVE_PATTERN
declare -a TEST_ORDER

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

classify_propagate() {
  local run_dir="$1"
  local log_file="$2"
  local rc=0

  if [[ -z "$FORMAL_PROPAGATE_CMD" ]]; then
    printf "propagated\t-1\n"
    return
  fi

  rc=0
  if ! run_command "$run_dir" "$FORMAL_PROPAGATE_CMD" "$log_file"; then
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
    printf "errors\t%s\n" "$mutation_errors" > "$meta_local"
    return 0
  fi

  export BASELINE=0
  export ORIG_DESIGN="$DESIGN"
  export MUTANT_DESIGN="$mutant_design"
  export MUTATION_ID="$mutation_id"
  export MUTATION_SPEC="$mutation_spec"
  export MUTATION_WORKDIR="$mutation_dir"

  for test_id in "${TEST_ORDER[@]}"; do
    export TEST_ID="$test_id"
    test_dir="${mutation_dir}/${test_id}"
    mkdir -p "$test_dir"

    read -r activate_state activate_rc < <(classify_activate "$test_dir" "$test_dir/activate.log")
    if [[ "$activate_state" == "error" ]]; then
      mutation_errors=$((mutation_errors + 1))
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "error" "-" "$activate_rc" "-1" "activation_error" >> "$pair_local"
      continue
    fi
    if [[ "$activate_state" == "not_activated" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "not_activated" "-" "$activate_rc" "-1" "skipped_no_activation" >> "$pair_local"
      continue
    fi

    activated_any=1
    read -r propagate_state propagate_rc < <(classify_propagate "$test_dir" "$test_dir/propagate.log")
    if [[ "$propagate_state" == "error" ]]; then
      mutation_errors=$((mutation_errors + 1))
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "activated" "error" "$activate_rc" "$propagate_rc" "propagation_error" >> "$pair_local"
      continue
    fi
    if [[ "$propagate_state" == "not_propagated" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mutation_id" "$test_id" "activated" "not_propagated" "$activate_rc" "$propagate_rc" "skipped_no_propagation" >> "$pair_local"
      continue
    fi

    propagated_any=1
    relevant_pairs=$((relevant_pairs + 1))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$mutation_id" "$test_id" "activated" "propagated" "$activate_rc" "$propagate_rc" "run_detection" >> "$pair_local"

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
      break
    fi
  done

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
  printf "errors\t%s\n" "$mutation_errors" > "$meta_local"
}

load_tests_manifest
load_mutations

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
done

count_relevant=$((count_detected + count_propagated_not_detected))
coverage_pct="100.00"
if [[ "$count_relevant" -gt 0 ]]; then
  coverage_pct="$(awk -v d="$count_detected" -v r="$count_relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
fi

{
  printf "metric\tvalue\n"
  printf "total_mutants\t%s\n" "$total_mutants"
  printf "relevant_mutants\t%s\n" "$count_relevant"
  printf "detected_mutants\t%s\n" "$count_detected"
  printf "propagated_not_detected_mutants\t%s\n" "$count_propagated_not_detected"
  printf "not_propagated_mutants\t%s\n" "$count_not_propagated"
  printf "not_activated_mutants\t%s\n" "$count_not_activated"
  printf "errors\t%s\n" "$errors"
  printf "mutation_coverage_percent\t%s\n" "$coverage_pct"
} > "$METRICS_FILE"

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

echo "Mutation coverage summary: total=${total_mutants} relevant=${count_relevant} detected=${count_detected} propagated_not_detected=${count_propagated_not_detected} not_propagated=${count_not_propagated} not_activated=${count_not_activated} errors=${errors} coverage=${coverage_pct}%"
echo "Gate status: ${gate_status}"
echo "Summary: ${SUMMARY_FILE}"
echo "Pair qualification: ${PAIR_FILE}"
echo "Results: ${RESULTS_FILE}"
echo "Metrics: ${METRICS_FILE}"
echo "Improvement: ${IMPROVEMENT_FILE}"

exit "$exit_code"
