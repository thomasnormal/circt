#!/usr/bin/env bash
# Shared argument validation and preparation for mutation MCY runner.

validate_and_prepare_runner_state() {
if [[ "$STRICT_BASELINE_GOVERNANCE" -eq 1 ]]; then
  REQUIRE_POLICY_FINGERPRINT_BASELINE=1
  REQUIRE_BASELINE_EXAMPLE_PARITY=1
  REQUIRE_BASELINE_SCHEMA_VERSION_MATCH=1
  REQUIRE_BASELINE_SCHEMA_CONTRACT_MATCH=1
  REQUIRE_UNIQUE_EXAMPLE_SELECTION=1
fi
if [[ "$STRICT_RETRY_REASON_BASELINE_GOVERNANCE" -eq 1 ]]; then
  REQUIRE_RETRY_REASON_BASELINE_SCHEMA_ARTIFACTS=1
  REQUIRE_RETRY_REASON_SCHEMA_ARTIFACT_VALIDITY=1
  REQUIRE_RETRY_REASON_BASELINE_PARITY=1
fi

if ! is_pos_int "$GENERATE_COUNT"; then
  echo "--generate-count must be a positive integer: $GENERATE_COUNT" >&2
  exit 1
fi
if ! is_pos_int "$JOBS"; then
  echo "--jobs must be a positive integer: $JOBS" >&2
  exit 1
fi
if ! is_nonneg_int "$EXAMPLE_TIMEOUT_SEC"; then
  echo "--example-timeout-sec must be a non-negative integer: $EXAMPLE_TIMEOUT_SEC" >&2
  exit 1
fi
if ! is_nonneg_int "$EXAMPLE_RETRIES"; then
  echo "--example-retries must be a non-negative integer: $EXAMPLE_RETRIES" >&2
  exit 1
fi
if ! is_nonneg_int "$EXAMPLE_RETRY_DELAY_MS"; then
  echo "--example-retry-delay-ms must be a non-negative integer: $EXAMPLE_RETRY_DELAY_MS" >&2
  exit 1
fi
if ! is_nonneg_int "$MUTATIONS_SEED"; then
  echo "--mutations-seed must be a non-negative integer: $MUTATIONS_SEED" >&2
  exit 1
fi
if [[ -n "$MUTATIONS_MODE_COUNTS" && -n "$MUTATIONS_MODE_WEIGHTS" ]]; then
  echo "Use either --mutations-mode-counts or --mutations-mode-weights, not both." >&2
  exit 1
fi
if ! is_pos_int "$MUTATION_LIMIT"; then
  echo "--mutation-limit must be a positive integer: $MUTATION_LIMIT" >&2
  exit 1
fi
if [[ "$MUTATIONS_BACKEND" != "yosys" && "$MUTATIONS_BACKEND" != "native" ]]; then
  echo "--mutations-backend must be one of: yosys|native" >&2
  exit 1
fi
if [[ "$REQUIRE_NATIVE_BACKEND" != "0" && "$REQUIRE_NATIVE_BACKEND" != "1" ]]; then
  echo "CIRCT_MUT_REQUIRE_NATIVE_BACKEND must be 0 or 1: $REQUIRE_NATIVE_BACKEND" >&2
  exit 1
fi
if [[ "$REQUIRE_NATIVE_BACKEND" == "1" && "$MUTATIONS_BACKEND" != "native" ]]; then
  echo "--require-native-backend requires --mutations-backend native" >&2
  exit 1
fi
if [[ "$FAIL_ON_NATIVE_NOOP_FALLBACK" != "0" && "$FAIL_ON_NATIVE_NOOP_FALLBACK" != "1" ]]; then
  echo "CIRCT_MUT_FAIL_ON_NATIVE_NOOP_FALLBACK must be 0 or 1: $FAIL_ON_NATIVE_NOOP_FALLBACK" >&2
  exit 1
fi
if [[ "$FAIL_ON_NATIVE_NOOP_FALLBACK" == "1" && "$MUTATIONS_BACKEND" != "native" ]]; then
  echo "--fail-on-native-noop-fallback requires --mutations-backend native" >&2
  exit 1
fi
if [[ "$NATIVE_TESTS_MODE" != "synthetic" && "$NATIVE_TESTS_MODE" != "real" ]]; then
  echo "--native-tests-mode must be one of: synthetic|real" >&2
  exit 1
fi
if [[ "$NATIVE_MUTATION_OPS_EXPLICIT" -eq 1 && -z "$(trim_whitespace "$NATIVE_MUTATION_OPS")" ]]; then
  echo "--native-mutation-ops must not be empty when provided" >&2
  exit 1
fi
if [[ -n "$NATIVE_MUTATION_OPS" ]]; then
  if ! validate_native_mutation_ops_spec "$NATIVE_MUTATION_OPS" "--native-mutation-ops"; then
    exit 1
  fi
  NATIVE_MUTATION_OPS="$(canonicalize_native_mutation_ops_spec "$NATIVE_MUTATION_OPS")"
fi
if [[ "$NATIVE_REAL_HARNESS_ARGS_EXPLICIT" -eq 1 && -z "$(trim_whitespace "$NATIVE_REAL_HARNESS_ARGS")" ]]; then
  echo "--native-real-harness-args must not be empty when provided" >&2
  exit 1
fi
if [[ -n "$NATIVE_REAL_HARNESS_ARGS" ]]; then
  if ! validate_native_real_harness_args_spec "$NATIVE_REAL_HARNESS_ARGS" "--native-real-harness-args"; then
    exit 1
  fi
  NATIVE_REAL_HARNESS_ARGS="$(trim_whitespace "$NATIVE_REAL_HARNESS_ARGS")"
fi
if ! is_nonneg_int "$MIN_DETECTED"; then
  echo "--min-detected must be a non-negative integer: $MIN_DETECTED" >&2
  exit 1
fi
if ! is_nonneg_int "$MIN_RELEVANT"; then
  echo "--min-relevant must be a non-negative integer: $MIN_RELEVANT" >&2
  exit 1
fi
if [[ -n "$MIN_COVERAGE_PERCENT" ]]; then
  if ! is_nonneg_decimal "$MIN_COVERAGE_PERCENT"; then
    echo "--min-coverage-percent must be numeric in range [0,100]: $MIN_COVERAGE_PERCENT" >&2
    exit 1
  fi
  if ! awk -v v="$MIN_COVERAGE_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
    echo "--min-coverage-percent must be numeric in range [0,100]: $MIN_COVERAGE_PERCENT" >&2
    exit 1
  fi
fi
if [[ -n "$MAX_ERRORS" ]] && ! is_nonneg_int "$MAX_ERRORS"; then
  echo "--max-errors must be a non-negative integer: $MAX_ERRORS" >&2
  exit 1
fi
if [[ -n "$MIN_TOTAL_DETECTED" ]] && ! is_nonneg_int "$MIN_TOTAL_DETECTED"; then
  echo "--min-total-detected must be a non-negative integer: $MIN_TOTAL_DETECTED" >&2
  exit 1
fi
if [[ -n "$MIN_TOTAL_RELEVANT" ]] && ! is_nonneg_int "$MIN_TOTAL_RELEVANT"; then
  echo "--min-total-relevant must be a non-negative integer: $MIN_TOTAL_RELEVANT" >&2
  exit 1
fi
if [[ -n "$MIN_TOTAL_COVERAGE_PERCENT" ]]; then
  if ! is_nonneg_decimal "$MIN_TOTAL_COVERAGE_PERCENT"; then
    echo "--min-total-coverage-percent must be numeric in range [0,100]: $MIN_TOTAL_COVERAGE_PERCENT" >&2
    exit 1
  fi
  if ! awk -v v="$MIN_TOTAL_COVERAGE_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
    echo "--min-total-coverage-percent must be numeric in range [0,100]: $MIN_TOTAL_COVERAGE_PERCENT" >&2
    exit 1
  fi
fi
if [[ -n "$MAX_TOTAL_ERRORS" ]] && ! is_nonneg_int "$MAX_TOTAL_ERRORS"; then
  echo "--max-total-errors must be a non-negative integer: $MAX_TOTAL_ERRORS" >&2
  exit 1
fi
if [[ -n "$MAX_TOTAL_RETRIES" ]] && ! is_nonneg_int "$MAX_TOTAL_RETRIES"; then
  echo "--max-total-retries must be a non-negative integer: $MAX_TOTAL_RETRIES" >&2
  exit 1
fi
if ! parse_retry_reason_budgets "$MAX_TOTAL_RETRIES_BY_REASON"; then
  exit 1
fi
if ! is_nonneg_int "$MAX_DETECTED_DROP"; then
  echo "--max-detected-drop must be a non-negative integer: $MAX_DETECTED_DROP" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_DETECTED_DROP_PERCENT"; then
  echo "--max-detected-drop-percent must be numeric in range [0,100]: $MAX_DETECTED_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_DETECTED_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-detected-drop-percent must be numeric in range [0,100]: $MAX_DETECTED_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_int "$MAX_RELEVANT_DROP"; then
  echo "--max-relevant-drop must be a non-negative integer: $MAX_RELEVANT_DROP" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_RELEVANT_DROP_PERCENT"; then
  echo "--max-relevant-drop-percent must be numeric in range [0,100]: $MAX_RELEVANT_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_RELEVANT_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-relevant-drop-percent must be numeric in range [0,100]: $MAX_RELEVANT_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_COVERAGE_DROP_PERCENT"; then
  echo "--max-coverage-drop-percent must be numeric in range [0,100]: $MAX_COVERAGE_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_COVERAGE_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-coverage-drop-percent must be numeric in range [0,100]: $MAX_COVERAGE_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_int "$MAX_ERRORS_INCREASE"; then
  echo "--max-errors-increase must be a non-negative integer: $MAX_ERRORS_INCREASE" >&2
  exit 1
fi
if ! is_nonneg_int "$MAX_TOTAL_DETECTED_DROP"; then
  echo "--max-total-detected-drop must be a non-negative integer: $MAX_TOTAL_DETECTED_DROP" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_TOTAL_DETECTED_DROP_PERCENT"; then
  echo "--max-total-detected-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_DETECTED_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_TOTAL_DETECTED_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-total-detected-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_DETECTED_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_int "$MAX_TOTAL_RELEVANT_DROP"; then
  echo "--max-total-relevant-drop must be a non-negative integer: $MAX_TOTAL_RELEVANT_DROP" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_TOTAL_RELEVANT_DROP_PERCENT"; then
  echo "--max-total-relevant-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_RELEVANT_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_TOTAL_RELEVANT_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-total-relevant-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_RELEVANT_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_decimal "$MAX_TOTAL_COVERAGE_DROP_PERCENT"; then
  echo "--max-total-coverage-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_COVERAGE_DROP_PERCENT" >&2
  exit 1
fi
if ! awk -v v="$MAX_TOTAL_COVERAGE_DROP_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--max-total-coverage-drop-percent must be numeric in range [0,100]: $MAX_TOTAL_COVERAGE_DROP_PERCENT" >&2
  exit 1
fi
if ! is_nonneg_int "$MAX_TOTAL_ERRORS_INCREASE"; then
  echo "--max-total-errors-increase must be a non-negative integer: $MAX_TOTAL_ERRORS_INCREASE" >&2
  exit 1
fi
if ! parse_retry_reason_drift_tolerances "$RETRY_REASON_DRIFT_TOLERANCES"; then
  exit 1
fi
if ! parse_retry_reason_drift_percent_tolerances "$RETRY_REASON_DRIFT_PERCENT_TOLERANCES"; then
  exit 1
fi
if ! is_nonneg_int "$RETRY_REASON_DRIFT_SUITE_TOLERANCE"; then
  echo "--retry-reason-drift-suite-tolerance must be a non-negative integer: $RETRY_REASON_DRIFT_SUITE_TOLERANCE" >&2
  exit 1
fi
if ! is_nonneg_decimal "$RETRY_REASON_DRIFT_SUITE_PERCENT_TOLERANCE"; then
  echo "--retry-reason-drift-suite-percent-tolerance must be a non-negative decimal: $RETRY_REASON_DRIFT_SUITE_PERCENT_TOLERANCE" >&2
  exit 1
fi
if [[ "$UPDATE_BASELINE" -eq 1 || "$FAIL_ON_DIFF" -eq 1 ]]; then
  if [[ -z "$BASELINE_FILE" ]]; then
    echo "--baseline-file is required with --update-baseline or --fail-on-diff" >&2
    exit 1
  fi
fi
if [[ ${#SUITE_BASELINE_HISTORY_FILES[@]} -gt 0 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--suite-baseline-history-file requires --fail-on-diff" >&2
  exit 1
fi
if [[ ${#EXAMPLE_BASELINE_HISTORY_FILES[@]} -gt 0 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--example-baseline-history-file requires --fail-on-diff" >&2
  exit 1
fi
if ! is_nonneg_decimal "$SUITE_HISTORY_PERCENTILE"; then
  echo "--suite-history-percentile must be numeric in range [0,100]: $SUITE_HISTORY_PERCENTILE" >&2
  exit 1
fi
if ! awk -v v="$SUITE_HISTORY_PERCENTILE" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--suite-history-percentile must be numeric in range [0,100]: $SUITE_HISTORY_PERCENTILE" >&2
  exit 1
fi
if ! is_nonneg_decimal "$EXAMPLE_HISTORY_PERCENTILE"; then
  echo "--example-history-percentile must be numeric in range [0,100]: $EXAMPLE_HISTORY_PERCENTILE" >&2
  exit 1
fi
if ! awk -v v="$EXAMPLE_HISTORY_PERCENTILE" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
  echo "--example-history-percentile must be numeric in range [0,100]: $EXAMPLE_HISTORY_PERCENTILE" >&2
  exit 1
fi
if [[ "$EXAMPLE_HISTORY_MISSING_POLICY" != "ignore" && "$EXAMPLE_HISTORY_MISSING_POLICY" != "warn" && "$EXAMPLE_HISTORY_MISSING_POLICY" != "fail" ]]; then
  echo "--example-history-missing-policy must be one of: ignore|warn|fail" >&2
  exit 1
fi
if [[ "$HISTORY_AGGREGATION_MODE" != "percentile" && "$HISTORY_AGGREGATION_MODE" != "ewma" ]]; then
  echo "--history-aggregation-mode must be one of: percentile|ewma" >&2
  exit 1
fi
if ! is_nonneg_decimal "$HISTORY_EWMA_ALPHA"; then
  echo "--history-ewma-alpha must be numeric in range (0,1]: $HISTORY_EWMA_ALPHA" >&2
  exit 1
fi
if ! awk -v v="$HISTORY_EWMA_ALPHA" 'BEGIN { exit !(v > 0 && v <= 1) }'; then
  echo "--history-ewma-alpha must be numeric in range (0,1]: $HISTORY_EWMA_ALPHA" >&2
  exit 1
fi
if [[ "$HISTORY_DETECTED_AGGREGATION_MODE" != "inherit" && "$HISTORY_DETECTED_AGGREGATION_MODE" != "percentile" && "$HISTORY_DETECTED_AGGREGATION_MODE" != "ewma" && "$HISTORY_DETECTED_AGGREGATION_MODE" != "max" ]]; then
  echo "--history-detected-aggregation-mode must be one of: inherit|percentile|ewma|max" >&2
  exit 1
fi
if [[ "$HISTORY_RELEVANT_AGGREGATION_MODE" != "inherit" && "$HISTORY_RELEVANT_AGGREGATION_MODE" != "percentile" && "$HISTORY_RELEVANT_AGGREGATION_MODE" != "ewma" && "$HISTORY_RELEVANT_AGGREGATION_MODE" != "max" ]]; then
  echo "--history-relevant-aggregation-mode must be one of: inherit|percentile|ewma|max" >&2
  exit 1
fi
if [[ "$HISTORY_COVERAGE_AGGREGATION_MODE" != "inherit" && "$HISTORY_COVERAGE_AGGREGATION_MODE" != "percentile" && "$HISTORY_COVERAGE_AGGREGATION_MODE" != "ewma" && "$HISTORY_COVERAGE_AGGREGATION_MODE" != "max" ]]; then
  echo "--history-coverage-aggregation-mode must be one of: inherit|percentile|ewma|max" >&2
  exit 1
fi
if [[ "$HISTORY_ERRORS_AGGREGATION_MODE" != "inherit" && "$HISTORY_ERRORS_AGGREGATION_MODE" != "percentile" && "$HISTORY_ERRORS_AGGREGATION_MODE" != "ewma" && "$HISTORY_ERRORS_AGGREGATION_MODE" != "max" ]]; then
  echo "--history-errors-aggregation-mode must be one of: inherit|percentile|ewma|max" >&2
  exit 1
fi
for history_file in "${SUITE_BASELINE_HISTORY_FILES[@]}"; do
  if [[ ! -f "$history_file" ]]; then
    echo "Suite baseline history file not found: $history_file" >&2
    exit 1
  fi
  if [[ ! -r "$history_file" ]]; then
    echo "Suite baseline history file not readable: $history_file" >&2
    exit 1
  fi
done
for history_file in "${EXAMPLE_BASELINE_HISTORY_FILES[@]}"; do
  if [[ ! -f "$history_file" ]]; then
    echo "Example baseline history file not found: $history_file" >&2
    exit 1
  fi
  if [[ ! -r "$history_file" ]]; then
    echo "Example baseline history file not readable: $history_file" >&2
    exit 1
  fi
done
if [[ "$UPDATE_BASELINE" -eq 1 && "$FAIL_ON_DIFF" -eq 1 ]]; then
  echo "Use either --update-baseline or --fail-on-diff, not both." >&2
  exit 1
fi
if [[ "$UPDATE_BASELINE" -eq 1 && "$FAIL_ON_RETRY_REASON_DIFF" -eq 1 ]]; then
  echo "Use either --update-baseline or --fail-on-retry-reason-diff, not both." >&2
  exit 1
fi
if [[ "$ALLOW_UPDATE_BASELINE_ON_FAILURE" -eq 1 && "$UPDATE_BASELINE" -ne 1 ]]; then
  echo "--allow-update-baseline-on-failure requires --update-baseline" >&2
  exit 1
fi
if [[ -z "$SUMMARY_SCHEMA_VERSION_FILE" ]]; then
  SUMMARY_SCHEMA_VERSION_FILE="${OUT_DIR}/summary.schema-version"
fi
if [[ -z "$BASELINE_SCHEMA_VERSION_FILE" && -n "$BASELINE_FILE" ]]; then
  BASELINE_SCHEMA_VERSION_FILE="${BASELINE_FILE}.schema-version"
fi
if [[ -z "$RETRY_REASON_BASELINE_FILE" && -n "$BASELINE_FILE" ]]; then
  RETRY_REASON_BASELINE_FILE="${BASELINE_FILE}.retry-reason-summary.tsv"
fi
if [[ -z "$RETRY_REASON_SUMMARY_SCHEMA_VERSION_FILE" ]]; then
  RETRY_REASON_SUMMARY_SCHEMA_VERSION_FILE="${OUT_DIR}/retry-reason-summary.schema-version"
fi
if [[ -z "$RETRY_REASON_SUMMARY_SCHEMA_CONTRACT_FILE" ]]; then
  RETRY_REASON_SUMMARY_SCHEMA_CONTRACT_FILE="${OUT_DIR}/retry-reason-summary.schema-contract"
fi
if [[ -z "$RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" && -n "$RETRY_REASON_BASELINE_FILE" ]]; then
  RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE="${RETRY_REASON_BASELINE_FILE}.schema-version"
fi
if [[ -z "$RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE" && -n "$RETRY_REASON_BASELINE_FILE" ]]; then
  RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE="${RETRY_REASON_BASELINE_FILE}.schema-contract"
fi
if [[ "$BASELINE_SCHEMA_VERSION_FILE_EXPLICIT" -eq 1 && "$FAIL_ON_DIFF" -eq 1 && "$REQUIRE_BASELINE_SCHEMA_VERSION_MATCH" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_SCHEMA_VERSION_FILE" ]]; then
    echo "Baseline schema-version file not found: $BASELINE_SCHEMA_VERSION_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_SCHEMA_VERSION_FILE" ]]; then
    echo "Baseline schema-version file not readable: $BASELINE_SCHEMA_VERSION_FILE" >&2
    exit 1
  fi
fi

if [[ -z "$SUMMARY_SCHEMA_CONTRACT_FILE" ]]; then
  SUMMARY_SCHEMA_CONTRACT_FILE="${OUT_DIR}/summary.schema-contract"
fi
if [[ -z "$BASELINE_SCHEMA_CONTRACT_FILE" && -n "$BASELINE_FILE" ]]; then
  BASELINE_SCHEMA_CONTRACT_FILE="${BASELINE_FILE}.schema-contract"
fi
if [[ "$BASELINE_SCHEMA_CONTRACT_FILE_EXPLICIT" -eq 1 && "$FAIL_ON_DIFF" -eq 1 && "$REQUIRE_BASELINE_SCHEMA_CONTRACT_MATCH" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_SCHEMA_CONTRACT_FILE" ]]; then
    echo "Baseline schema-contract file not found: $BASELINE_SCHEMA_CONTRACT_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_SCHEMA_CONTRACT_FILE" ]]; then
    echo "Baseline schema-contract file not readable: $BASELINE_SCHEMA_CONTRACT_FILE" >&2
    exit 1
  fi
fi

if [[ "$MIGRATE_BASELINE_SCHEMA_ARTIFACTS" -eq 1 ]]; then
  if [[ "$UPDATE_BASELINE" -eq 1 || "$FAIL_ON_DIFF" -eq 1 ]]; then
    echo "--migrate-baseline-schema-artifacts cannot be combined with --update-baseline or --fail-on-diff" >&2
    exit 1
  fi
  if [[ "$FAIL_ON_RETRY_REASON_DIFF" -eq 1 ]]; then
    echo "--migrate-baseline-schema-artifacts cannot be combined with --fail-on-retry-reason-diff" >&2
    exit 1
  fi
  if [[ -z "$BASELINE_FILE" ]]; then
    echo "--baseline-file is required with --migrate-baseline-schema-artifacts" >&2
    exit 1
  fi
  if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Baseline file not found: $BASELINE_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_FILE" ]]; then
    echo "Baseline file not readable: $BASELINE_FILE" >&2
    exit 1
  fi
  if ! migrate_baseline_schema_artifacts "$BASELINE_FILE" "$BASELINE_SCHEMA_VERSION_FILE" "$BASELINE_SCHEMA_CONTRACT_FILE"; then
    exit 1
  fi
  exit 0
fi

if [[ "$MIGRATE_RETRY_REASON_BASELINE_SCHEMA_ARTIFACTS" -eq 1 ]]; then
  if [[ "$UPDATE_BASELINE" -eq 1 || "$FAIL_ON_DIFF" -eq 1 ]]; then
    echo "--migrate-retry-reason-baseline-schema-artifacts cannot be combined with --update-baseline or --fail-on-diff" >&2
    exit 1
  fi
  if [[ "$FAIL_ON_RETRY_REASON_DIFF" -eq 1 ]]; then
    echo "--migrate-retry-reason-baseline-schema-artifacts cannot be combined with --fail-on-retry-reason-diff" >&2
    exit 1
  fi
  if [[ -z "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "--migrate-retry-reason-baseline-schema-artifacts requires --retry-reason-baseline-file or --baseline-file" >&2
    exit 1
  fi
  if [[ ! -f "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "Retry-reason baseline file not found: $RETRY_REASON_BASELINE_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "Retry-reason baseline file not readable: $RETRY_REASON_BASELINE_FILE" >&2
    exit 1
  fi
  if ! migrate_retry_reason_baseline_schema_artifacts "$RETRY_REASON_BASELINE_FILE" "$RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" "$RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE"; then
    exit 1
  fi
  exit 0
fi

if [[ "$STRICT_BASELINE_GOVERNANCE" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--strict-baseline-governance requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_POLICY_FINGERPRINT_BASELINE" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-policy-fingerprint-baseline requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_BASELINE_EXAMPLE_PARITY" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-baseline-example-parity requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_BASELINE_SCHEMA_VERSION_MATCH" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-baseline-schema-version-match requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_BASELINE_SCHEMA_CONTRACT_MATCH" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-baseline-schema-contract-match requires --fail-on-diff" >&2
  exit 1
fi
if [[ -n "$DRIFT_ALLOWLIST_FILE" && "$FAIL_ON_DIFF" -ne 1 && "$REQUIRE_UNIQUE_SUMMARY_ROWS" -ne 1 ]]; then
  echo "--drift-allowlist-file requires --fail-on-diff or --require-unique-summary-rows" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNUSED_DRIFT_ALLOWLIST" -eq 1 && -z "$DRIFT_ALLOWLIST_FILE" ]]; then
  echo "--fail-on-unused-drift-allowlist requires --drift-allowlist-file" >&2
  exit 1
fi
if [[ -n "$DRIFT_ALLOWLIST_UNUSED_FILE" && -z "$DRIFT_ALLOWLIST_FILE" ]]; then
  echo "--drift-allowlist-unused-file requires --drift-allowlist-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_DIFF" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Baseline file not found: $BASELINE_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_FILE" ]]; then
    echo "Baseline file not readable: $BASELINE_FILE" >&2
    exit 1
  fi
fi
if [[ "$FAIL_ON_RETRY_REASON_DIFF" -eq 1 ]]; then
  if [[ -z "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "--fail-on-retry-reason-diff requires --retry-reason-baseline-file or --baseline-file" >&2
    exit 1
  fi
  if [[ ! -f "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "Retry-reason baseline file not found: $RETRY_REASON_BASELINE_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$RETRY_REASON_BASELINE_FILE" ]]; then
    echo "Retry-reason baseline file not readable: $RETRY_REASON_BASELINE_FILE" >&2
    exit 1
  fi
  if [[ "$RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE_EXPLICIT" -eq 1 || "$REQUIRE_RETRY_REASON_BASELINE_SCHEMA_ARTIFACTS" -eq 1 ]]; then
    if [[ ! -f "$RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" ]]; then
      echo "Retry-reason baseline schema-version file not found: $RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" >&2
      exit 1
    fi
    if [[ ! -r "$RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" ]]; then
      echo "Retry-reason baseline schema-version file not readable: $RETRY_REASON_BASELINE_SCHEMA_VERSION_FILE" >&2
      exit 1
    fi
  fi
  if [[ "$RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE_EXPLICIT" -eq 1 || "$REQUIRE_RETRY_REASON_BASELINE_SCHEMA_ARTIFACTS" -eq 1 ]]; then
    if [[ ! -f "$RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE" ]]; then
      echo "Retry-reason baseline schema-contract file not found: $RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE" >&2
      exit 1
    fi
    if [[ ! -r "$RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE" ]]; then
      echo "Retry-reason baseline schema-contract file not readable: $RETRY_REASON_BASELINE_SCHEMA_CONTRACT_FILE" >&2
      exit 1
    fi
  fi
fi
if [[ "$STRICT_RETRY_REASON_BASELINE_GOVERNANCE" -eq 1 && "$FAIL_ON_RETRY_REASON_DIFF" -ne 1 ]]; then
  echo "--strict-retry-reason-baseline-governance requires --fail-on-retry-reason-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_RETRY_REASON_BASELINE_SCHEMA_ARTIFACTS" -eq 1 && "$FAIL_ON_RETRY_REASON_DIFF" -ne 1 ]]; then
  echo "--require-retry-reason-baseline-schema-artifacts requires --fail-on-retry-reason-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_RETRY_REASON_SCHEMA_ARTIFACT_VALIDITY" -eq 1 && "$FAIL_ON_RETRY_REASON_DIFF" -ne 1 ]]; then
  echo "--require-retry-reason-schema-artifact-validity requires --fail-on-retry-reason-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_RETRY_REASON_BASELINE_PARITY" -eq 1 && "$FAIL_ON_RETRY_REASON_DIFF" -ne 1 ]]; then
  echo "--require-retry-reason-baseline-parity requires --fail-on-retry-reason-diff" >&2
  exit 1
fi
if [[ -n "$DRIFT_ALLOWLIST_FILE" ]]; then
  if [[ ! -f "$DRIFT_ALLOWLIST_FILE" ]]; then
    echo "Drift allowlist file not found: $DRIFT_ALLOWLIST_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$DRIFT_ALLOWLIST_FILE" ]]; then
    echo "Drift allowlist file not readable: $DRIFT_ALLOWLIST_FILE" >&2
    exit 1
  fi
  load_drift_allowlist "$DRIFT_ALLOWLIST_FILE"
fi

if [[ -n "$EXAMPLE_MANIFEST" ]]; then
  if [[ ! -f "$EXAMPLE_MANIFEST" ]]; then
    echo "Example manifest file not found: $EXAMPLE_MANIFEST" >&2
    exit 1
  fi
  if [[ ! -r "$EXAMPLE_MANIFEST" ]]; then
    echo "Example manifest file not readable: $EXAMPLE_MANIFEST" >&2
    exit 1
  fi
  load_example_manifest "$EXAMPLE_MANIFEST"
else
  load_default_examples
fi

if [[ "$SMOKE" -eq 1 ]]; then
  if [[ "$MUTATION_GENERATION_FLAGS_SEEN" -eq 1 ]] || has_manifest_generation_overrides; then
    echo "Mutation generation options (--mutations-*) and native real harness overrides (--native-real-harness-args, manifest native_real_harness_script/native_real_harness_args) require non-smoke mode." >&2
    exit 1
  fi
fi

if [[ ${#EXAMPLE_IDS[@]} -eq 0 ]]; then
  EXAMPLE_IDS=("${AVAILABLE_EXAMPLES[@]}")
fi

for example_id in "${EXAMPLE_IDS[@]}"; do
  if [[ -z "${EXAMPLE_TO_DESIGN[$example_id]+x}" ]]; then
    echo "Unknown --example value: $example_id" >&2
    exit 1
  fi
done

if [[ "$REQUIRE_UNIQUE_EXAMPLE_SELECTION" -eq 1 ]]; then
  ensure_unique_example_selection
fi

if [[ -z "$CIRCT_MUT" ]]; then
  if [[ -x "$DEFAULT_CIRCT_MUT" ]]; then
    CIRCT_MUT="$DEFAULT_CIRCT_MUT"
  else
    CIRCT_MUT="circt-mut"
  fi
fi

if ! CIRCT_MUT_RESOLVED="$(resolve_tool "$CIRCT_MUT")"; then
  echo "circt-mut not found or not executable: $CIRCT_MUT" >&2
  exit 1
fi

if [[ "$EXAMPLE_TIMEOUT_SEC" -gt 0 ]] || has_manifest_timeout_overrides_enabled; then
  if ! TIMEOUT_RESOLVED="$(resolve_tool "$TIMEOUT_BIN")"; then
    echo "timeout utility not found or not executable: $TIMEOUT_BIN (set TIMEOUT or disable --example-timeout-sec)" >&2
    exit 1
  fi
else
  TIMEOUT_RESOLVED=""
fi

if [[ "$SMOKE" -ne 1 ]]; then
  if [[ "$MUTATIONS_BACKEND" == "yosys" ]]; then
    if ! YOSYS_RESOLVED="$(resolve_tool "$YOSYS_BIN")"; then
      echo "yosys not found or not executable: $YOSYS_BIN (use --smoke, --yosys PATH, or --mutations-backend native)" >&2
      exit 1
    fi
  else
    YOSYS_RESOLVED=""
  fi
else
  YOSYS_RESOLVED=""
fi

}
