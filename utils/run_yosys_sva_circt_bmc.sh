#!/usr/bin/env bash
set -euo pipefail

YOSYS_SVA_DIR="${1:-/home/thomas-ahle/yosys/tests/sva}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"

# Memory limit settings to prevent system hangs
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with memory limit
run_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_TIMEOUT_SECS "$@"
  )
}
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
# Yosys SVA tests are 2-state; default to known inputs to avoid X-driven
# counterexamples. Set BMC_ASSUME_KNOWN_INPUTS=0 to exercise 4-state behavior.
BMC_ASSUME_KNOWN_INPUTS="${BMC_ASSUME_KNOWN_INPUTS:-1}"
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
SKIP_VHDL="${SKIP_VHDL:-1}"
SKIP_FAIL_WITHOUT_MACRO="${SKIP_FAIL_WITHOUT_MACRO:-1}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
YOSYS_SVA_MODE_SUMMARY_TSV_FILE="${YOSYS_SVA_MODE_SUMMARY_TSV_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_JSON_FILE="${YOSYS_SVA_MODE_SUMMARY_JSON_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES="${YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES:-0}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS="${YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS:-0}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS="${YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS:-86400}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY:-error}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION:-1}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH:-auto}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY:-infer}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY:-infer}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_RUN_ID_REGEX="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_RUN_ID_REGEX:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_REASON_REGEX="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_REASON_REGEX:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_REGEX="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_REGEX:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_REGEX="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_REGEX:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_LIST="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_LIST:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_LIST="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_LIST:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MODE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MODE:-all}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_CLAUSES_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_CLAUSES_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILES_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILES_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_LIST="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_LIST:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_DEFAULT_LIST="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_DEFAULT_LIST:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_OVERLAY_LIST="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_OVERLAY_LIST:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE:-off}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_PROVIDER="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_PROVIDER:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_JOB="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_JOB:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_BRANCH="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_BRANCH:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_TARGET="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_TARGET:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_JSON="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_JSON:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_VERSION="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_VERSION:-1}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MIN="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MIN:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MAX="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MAX:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES:-0}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS:-0}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_FILE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_FILE:-}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND:-auto}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS:-30}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS:-300}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR="${YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR:-auto}"
YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE:-auto}"
YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION="${YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION:-1}"
YOSYS_SVA_MODE_SUMMARY_RUN_ID="${YOSYS_SVA_MODE_SUMMARY_RUN_ID:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECT_FILE="${EXPECT_FILE:-$SCRIPT_DIR/yosys-sva-bmc-expected.txt}"
# Backward-compatible fallback for legacy xfail-only rows.
XFAIL_FILE="${XFAIL_FILE:-$SCRIPT_DIR/yosys-sva-bmc-xfail.txt}"
ALLOW_XPASS="${ALLOW_XPASS:-0}"
EXPECT_DIFF_BASELINE="${EXPECT_DIFF_BASELINE:-}"
EXPECT_DIFF_BASELINE_DEFAULT_EXPECTED="${EXPECT_DIFF_BASELINE_DEFAULT_EXPECTED:-}"
EXPECT_DIFF_FAIL_ON_CHANGE="${EXPECT_DIFF_FAIL_ON_CHANGE:-0}"
EXPECT_DIFF_FILE="${EXPECT_DIFF_FILE:-}"
EXPECT_DIFF_TSV_FILE="${EXPECT_DIFF_TSV_FILE:-}"
EXPECT_DIFF_JSON_FILE="${EXPECT_DIFF_JSON_FILE:-}"
EXPECT_OBSERVED_FILE="${EXPECT_OBSERVED_FILE:-}"
EXPECT_REGEN_FILE="${EXPECT_REGEN_FILE:-}"
EXPECT_OBSERVED_INCLUDE_SKIPPED="${EXPECT_OBSERVED_INCLUDE_SKIPPED:-0}"
EXPECT_REGEN_FAIL_POLICY="${EXPECT_REGEN_FAIL_POLICY:-xfail}"
EXPECT_REGEN_SKIP_POLICY="${EXPECT_REGEN_SKIP_POLICY:-omit}"
EXPECT_REGEN_OVERRIDE_FILE="${EXPECT_REGEN_OVERRIDE_FILE:-}"
EXPECT_SKIP_STRICT="${EXPECT_SKIP_STRICT:-0}"
EXPECT_LINT="${EXPECT_LINT:-0}"
EXPECT_LINT_FAIL_ON_ISSUES="${EXPECT_LINT_FAIL_ON_ISSUES:-0}"
EXPECT_LINT_FILE="${EXPECT_LINT_FILE:-}"
EXPECT_LINT_HINTS_FILE="${EXPECT_LINT_HINTS_FILE:-}"
EXPECT_LINT_FIXES_FILE="${EXPECT_LINT_FIXES_FILE:-}"
EXPECT_LINT_APPLY_MODE="${EXPECT_LINT_APPLY_MODE:-off}"
EXPECT_LINT_APPLY_DIFF_FILE="${EXPECT_LINT_APPLY_DIFF_FILE:-}"
EXPECT_LINT_APPLY_ACTIONS="${EXPECT_LINT_APPLY_ACTIONS:-drop-row,set-row}"
EXPECT_LINT_APPLY_ADDROW_FILTER_FILE="${EXPECT_LINT_APPLY_ADDROW_FILTER_FILE:-}"
EXPECT_FORMAT_MODE="${EXPECT_FORMAT_MODE:-off}"
EXPECT_FORMAT_FILES="${EXPECT_FORMAT_FILES:-}"
EXPECT_FORMAT_DIFF_FILE="${EXPECT_FORMAT_DIFF_FILE:-}"
EXPECT_FORMAT_REWRITE_MALFORMED="${EXPECT_FORMAT_REWRITE_MALFORMED:-0}"
EXPECT_FORMAT_MALFORMED_FIX_FILE="${EXPECT_FORMAT_MALFORMED_FIX_FILE:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE:-0}"
EXPECT_FORMAT_UNFIXABLE_FILE="${EXPECT_FORMAT_UNFIXABLE_FILE:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_FILE_FILTER="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_FILE_FILTER:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_PROFILE_FILTER="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_PROFILE_FILTER:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_REASON_FILTER="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_REASON_FILTER:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_SEVERITY_FILTER="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_SEVERITY_FILTER:-}"
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY:-custom}"
# NOTE: NO_PROPERTY_AS_SKIP defaults to 0 because the "no property provided to check"
# warning is emitted before LTLToCore/LowerClockedAssertLike run, so clocked
# assertions may be present but not lowered yet. Setting this to 1 can cause
# false SKIP results.
NO_PROPERTY_AS_SKIP="${NO_PROPERTY_AS_SKIP:-0}"

if [[ ! -d "$YOSYS_SVA_DIR" ]]; then
  echo "yosys SVA directory not found: $YOSYS_SVA_DIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

failures=0
total=0
skipped=0
xfails=0
xpasses=0
mode_total=0
mode_out_pass=0
mode_out_fail=0
mode_out_xfail=0
mode_out_xpass=0
mode_out_epass=0
mode_out_efail=0
mode_out_unskip=0
mode_skipped=0
mode_skipped_pass=0
mode_skipped_fail=0
mode_skipped_expected=0
mode_skipped_unexpected=0
mode_skip_reason_vhdl=0
mode_skip_reason_fail_no_macro=0
mode_skip_reason_no_property=0
mode_skip_reason_other=0
expect_diff_added=0
expect_diff_removed=0
expect_diff_changed=0
lint_issues=0
format_unfixable_issues=0

declare -A expected_cases
declare -A observed_cases
declare -A regen_override_cases
declare -A suite_tests
declare -a addrow_filter_source_patterns=()
declare -a addrow_filter_key_patterns=()
declare -a addrow_filter_row_patterns=()

case "$EXPECT_LINT_APPLY_MODE" in
  off|dry-run|apply) ;;
  *)
    echo "invalid EXPECT_LINT_APPLY_MODE: $EXPECT_LINT_APPLY_MODE (expected off|dry-run|apply)" >&2
    exit 1
    ;;
esac
case "$EXPECT_FORMAT_MODE" in
  off|dry-run|apply) ;;
  *)
    echo "invalid EXPECT_FORMAT_MODE: $EXPECT_FORMAT_MODE (expected off|dry-run|apply)" >&2
    exit 1
    ;;
esac
case "$EXPECT_FORMAT_REWRITE_MALFORMED" in
  0|1) ;;
  *)
    echo "invalid EXPECT_FORMAT_REWRITE_MALFORMED: $EXPECT_FORMAT_REWRITE_MALFORMED (expected 0|1)" >&2
    exit 1
    ;;
esac
case "$EXPECT_FORMAT_FAIL_ON_UNFIXABLE" in
  0|1) ;;
  *)
    echo "invalid EXPECT_FORMAT_FAIL_ON_UNFIXABLE: $EXPECT_FORMAT_FAIL_ON_UNFIXABLE (expected 0|1)" >&2
    exit 1
    ;;
esac
EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY="${EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY,,}"
case "$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY" in
  custom|all|syntax-only|semantic-only|error-only|warning-only) ;;
  *)
    echo "invalid EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY: $EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY (expected custom|all|syntax-only|semantic-only|error-only|warning-only)" >&2
    exit 1
    ;;
esac
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES: $YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS: $YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS: $YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION (expected non-negative integer)" >&2
  exit 1
fi
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH" in
  auto|cksum|crc32) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH (expected auto|cksum|crc32)" >&2
    exit 1
    ;;
esac
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY" in
  preserve|infer|rewrite) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY (expected preserve|infer|rewrite)" >&2
    exit 1
    ;;
esac
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY" in
  preserve|infer|rewrite) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY (expected preserve|infer|rewrite)" >&2
    exit 1
    ;;
esac
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS (expected non-negative integer)" >&2
  exit 1
fi
if [[ ! "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS (expected non-negative integer)" >&2
  exit 1
fi
YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY="${YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY" in
  error|warn) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY: $YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY (expected error|warn)" >&2
    exit 1
    ;;
esac
YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR="${YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR" in
  auto|python) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR: $YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR (expected auto|python)" >&2
    exit 1
    ;;
esac
YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE="${YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE" in
  auto|python|shell) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE: $YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE (expected auto|python|shell)" >&2
    exit 1
    ;;
esac
YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND,,}"
case "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND" in
  auto|flock|mkdir|none) ;;
  *)
    echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND: $YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND (expected auto|flock|mkdir|none)" >&2
    exit 1
    ;;
esac
if [[ -v YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_REGEX_POLICY ]]; then
  echo "invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_REGEX_POLICY: regex mode has been removed; unset this variable and use YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR=auto|python" >&2
  exit 1
fi
strict_unfixable_bundle_reason_filter=""
strict_unfixable_bundle_severity_filter=""
case "$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY" in
  syntax-only)
    strict_unfixable_bundle_reason_filter="syntax-*"
    strict_unfixable_bundle_severity_filter="error"
    ;;
  semantic-only)
    strict_unfixable_bundle_reason_filter="semantic-*,policy-*"
    strict_unfixable_bundle_severity_filter="warning"
    ;;
  error-only)
    strict_unfixable_bundle_severity_filter="error"
    ;;
  warning-only)
    strict_unfixable_bundle_severity_filter="warning"
    ;;
  custom|all)
    ;;
esac
STRICT_UNFIXABLE_FILE_FILTER="$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_FILE_FILTER"
STRICT_UNFIXABLE_PROFILE_FILTER="$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_PROFILE_FILTER"
STRICT_UNFIXABLE_REASON_FILTER="$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_REASON_FILTER"
STRICT_UNFIXABLE_SEVERITY_FILTER="$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_SEVERITY_FILTER"
if [[ -z "$STRICT_UNFIXABLE_REASON_FILTER" && -n "$strict_unfixable_bundle_reason_filter" ]]; then
  STRICT_UNFIXABLE_REASON_FILTER="$strict_unfixable_bundle_reason_filter"
fi
if [[ -z "$STRICT_UNFIXABLE_SEVERITY_FILTER" && -n "$strict_unfixable_bundle_severity_filter" ]]; then
  STRICT_UNFIXABLE_SEVERITY_FILTER="$strict_unfixable_bundle_severity_filter"
fi
EXPECT_LINT_APPLY_ACTIONS="${EXPECT_LINT_APPLY_ACTIONS// /}"
if [[ -n "$EXPECT_LINT_APPLY_ACTIONS" ]]; then
  local_actions_csv=",$EXPECT_LINT_APPLY_ACTIONS,"
  if [[ "$local_actions_csv" == *",all,"* ]]; then
    EXPECT_LINT_APPLY_ACTIONS="drop-row,set-row,add-row"
  fi
  while IFS= read -r action; do
    case "$action" in
      drop-row|set-row|add-row) ;;
      "")
        echo "invalid EXPECT_LINT_APPLY_ACTIONS: empty action segment" >&2
        exit 1
        ;;
      *)
        echo "invalid EXPECT_LINT_APPLY_ACTIONS action: $action (expected drop-row,set-row,add-row)" >&2
        exit 1
        ;;
    esac
  done < <(printf '%s\n' "$EXPECT_LINT_APPLY_ACTIONS" | tr ',' '\n')
fi
if [[ -n "$EXPECT_LINT_APPLY_ADDROW_FILTER_FILE" ]] && [[ ! -f "$EXPECT_LINT_APPLY_ADDROW_FILTER_FILE" ]]; then
  echo "EXPECT_LINT_APPLY_ADDROW_FILTER_FILE not found: $EXPECT_LINT_APPLY_ADDROW_FILTER_FILE" >&2
  exit 1
fi
if [[ "$EXPECT_LINT" == "1" ]] && [[ "$EXPECT_LINT_APPLY_MODE" != "off" ]] && [[ -z "$EXPECT_LINT_FIXES_FILE" ]]; then
  EXPECT_LINT_FIXES_FILE="$tmpdir/expect-lint-fixes.tsv"
fi
if [[ "$EXPECT_LINT" == "1" && -n "$EXPECT_LINT_FILE" ]]; then
  : > "$EXPECT_LINT_FILE"
fi
if [[ "$EXPECT_LINT" == "1" && -n "$EXPECT_LINT_HINTS_FILE" ]]; then
  : > "$EXPECT_LINT_HINTS_FILE"
  printf 'kind\tsource\tkey\taction\tnote\n' >> "$EXPECT_LINT_HINTS_FILE"
fi
if [[ "$EXPECT_LINT" == "1" && -n "$EXPECT_LINT_FIXES_FILE" ]]; then
  : > "$EXPECT_LINT_FIXES_FILE"
  printf 'kind\tsource\taction\tkey\trow\tnote\n' >> "$EXPECT_LINT_FIXES_FILE"
fi
if [[ "$EXPECT_FORMAT_MODE" != "off" ]] && [[ -n "$EXPECT_FORMAT_MALFORMED_FIX_FILE" ]]; then
  : > "$EXPECT_FORMAT_MALFORMED_FIX_FILE"
  printf 'file\toriginal\tsuggested\treason\tapplied\n' >> "$EXPECT_FORMAT_MALFORMED_FIX_FILE"
fi
if [[ "$EXPECT_FORMAT_MODE" != "off" ]] && [[ -n "$EXPECT_FORMAT_UNFIXABLE_FILE" ]]; then
  : > "$EXPECT_FORMAT_UNFIXABLE_FILE"
  printf 'file\tline\treason\tseverity\tprofile\tstrict_selected\n' >> "$EXPECT_FORMAT_UNFIXABLE_FILE"
fi

emit_expect_lint() {
  local kind="$1"
  local message="$2"
  local line="EXPECT_LINT($kind): $message"
  echo "$line"
  if [[ -n "$EXPECT_LINT_FILE" ]]; then
    echo "$line" >> "$EXPECT_LINT_FILE"
  fi
  lint_issues=$((lint_issues + 1))
}

emit_expect_lint_hint() {
  local kind="$1"
  local source="$2"
  local key="$3"
  local action="$4"
  local note="${5:-}"
  if [[ -z "$EXPECT_LINT_HINTS_FILE" ]]; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$source" "$key" "$action" "$note" >> "$EXPECT_LINT_HINTS_FILE"
  return 0
}

emit_expect_lint_fix() {
  local kind="$1"
  local source="$2"
  local action="$3"
  local key="$4"
  local row="${5:-}"
  local note="${6:-}"
  if [[ -z "$EXPECT_LINT_FIXES_FILE" ]]; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$source" "$action" "$key" "$row" "$note" >> "$EXPECT_LINT_FIXES_FILE"
  return 0
}

format_row_spec() {
  local key="$1"
  local expected="$2"
  printf '%s|%s' "$key" "$expected"
}

resolve_lint_fix_source_file() {
  local source="$1"
  case "$source" in
    EXPECT_FILE)
      if [[ -n "$EXPECT_FILE" ]]; then
        echo "$EXPECT_FILE"
      else
        echo ""
      fi
      ;;
    EXPECT_REGEN_OVERRIDE_FILE)
      if [[ -n "$EXPECT_REGEN_OVERRIDE_FILE" ]]; then
        echo "$EXPECT_REGEN_OVERRIDE_FILE"
      else
        echo ""
      fi
      ;;
    *)
      echo "$source"
      ;;
  esac
}

append_apply_diff() {
  local diff_file="$1"
  local source_file="$2"
  local changed="$3"
  if [[ -z "$EXPECT_LINT_APPLY_DIFF_FILE" ]]; then
    return 0
  fi
  if [[ "$changed" == "1" ]]; then
    {
      printf '=== %s ===\n' "$source_file"
      cat "$diff_file"
      printf '\n'
    } >> "$EXPECT_LINT_APPLY_DIFF_FILE"
  fi
  return 0
}

lint_apply_action_enabled() {
  local action="$1"
  if [[ -z "$EXPECT_LINT_APPLY_ACTIONS" ]]; then
    return 1
  fi
  case ",$EXPECT_LINT_APPLY_ACTIONS," in
    *",$action,"*) return 0 ;;
    *) return 1 ;;
  esac
}

load_addrow_filter_rules() {
  local source_pattern key_pattern row_pattern extra
  addrow_filter_source_patterns=()
  addrow_filter_key_patterns=()
  addrow_filter_row_patterns=()
  if [[ -z "$EXPECT_LINT_APPLY_ADDROW_FILTER_FILE" ]]; then
    return 0
  fi
  while IFS=$'\t' read -r source_pattern key_pattern row_pattern extra; do
    if [[ -n "${extra:-}" ]]; then
      continue
    fi
    [[ -z "$source_pattern" || "$source_pattern" =~ ^# ]] && continue
    [[ -z "$key_pattern" ]] && key_pattern="*"
    [[ -z "$row_pattern" ]] && row_pattern="*"
    addrow_filter_source_patterns+=("$source_pattern")
    addrow_filter_key_patterns+=("$key_pattern")
    addrow_filter_row_patterns+=("$row_pattern")
  done < "$EXPECT_LINT_APPLY_ADDROW_FILTER_FILE"
  return 0
}

addrow_filter_matches() {
  local source_raw="$1"
  local source_file="$2"
  local key="$3"
  local row="$4"
  local i source_pattern key_pattern row_pattern

  if [[ -z "$EXPECT_LINT_APPLY_ADDROW_FILTER_FILE" ]]; then
    return 0
  fi
  for i in "${!addrow_filter_source_patterns[@]}"; do
    source_pattern="${addrow_filter_source_patterns[$i]}"
    key_pattern="${addrow_filter_key_patterns[$i]}"
    row_pattern="${addrow_filter_row_patterns[$i]}"
    if [[ "$source_raw" != $source_pattern && "$source_file" != $source_pattern ]]; then
      continue
    fi
    if [[ "$key" != $key_pattern ]]; then
      continue
    fi
    if [[ "$row" != $row_pattern ]]; then
      continue
    fi
    return 0
  done
  return 1
}

apply_expect_lint_fixes() {
  local fixes_file="$1"
  local mode="$2"
  local -A drop_keys=()
  local -A set_rows=()
  local -A add_rows=()
  local -A touched_files=()
  local kind source action key row note
  local source_file map_key add_map_key

  [[ -f "$fixes_file" ]] || return 0
  load_addrow_filter_rules
  if [[ -n "$EXPECT_LINT_APPLY_DIFF_FILE" ]]; then
    : > "$EXPECT_LINT_APPLY_DIFF_FILE"
  fi

  while IFS=$'\t' read -r kind source action key row note; do
    [[ "$kind" == "kind" && "$source" == "source" ]] && continue
    [[ -z "$action" || -z "$source" ]] && continue
    source_file="$(resolve_lint_fix_source_file "$source")"
    [[ -z "$source_file" ]] && continue
    [[ "$source_file" == "/dev/null" ]] && continue
    if [[ ! -f "$source_file" ]]; then
      continue
    fi
    map_key="$source_file|$key"
    case "$action" in
      drop-row)
        if ! lint_apply_action_enabled "drop-row"; then
          continue
        fi
        touched_files["$source_file"]=1
        drop_keys["$map_key"]=1
        unset 'set_rows[$map_key]'
        ;;
      set-row)
        if ! lint_apply_action_enabled "set-row"; then
          continue
        fi
        touched_files["$source_file"]=1
        if [[ -z "${drop_keys["$map_key"]+x}" ]]; then
          set_rows["$map_key"]="$row"
        fi
        ;;
      add-row)
        if ! lint_apply_action_enabled "add-row"; then
          continue
        fi
        touched_files["$source_file"]=1
        if [[ -z "$row" ]]; then
          continue
        fi
        if ! addrow_filter_matches "$source" "$source_file" "$key" "$row"; then
          continue
        fi
        add_map_key="$source_file|$row"
        add_rows["$add_map_key"]=1
        ;;
      *)
        ;;
    esac
  done < "$fixes_file"

  local -a file_list=()
  local file
  for file in "${!touched_files[@]}"; do
    file_list+=("$file")
  done
  if ((${#file_list[@]} == 0)); then
    echo "EXPECT_LINT_APPLY: mode=$mode files=0 changed=0 actions=$EXPECT_LINT_APPLY_ACTIONS addrow_filter=${EXPECT_LINT_APPLY_ADDROW_FILTER_FILE:-none}"
    return 0
  fi

  local changed_files=0
  local processed=0
  local line test mode_field profile expected extra row_spec row_key row_expected tuple_key
  local drop_count set_count add_count changed
  local old_file new_file diff_file
  local -a append_rows=()
  local -A present_tuples=()
  local -A present_rows=()
  local mk
  while IFS= read -r file; do
    processed=$((processed + 1))
    old_file="$tmpdir/lint-apply-old-$processed.tsv"
    new_file="$tmpdir/lint-apply-new-$processed.tsv"
    diff_file="$tmpdir/lint-apply-diff-$processed.txt"
    cp "$file" "$old_file"

    drop_count=0
    set_count=0
    add_count=0
    while IFS= read -r mk; do
      if [[ "$mk" == "$file|"* ]]; then
        if [[ -n "${drop_keys["$mk"]+x}" ]]; then
          drop_count=$((drop_count + 1))
        fi
        if [[ -n "${set_rows["$mk"]+x}" ]]; then
          set_count=$((set_count + 1))
        fi
        if [[ -n "${add_rows["$mk"]+x}" ]]; then
          add_count=$((add_count + 1))
        fi
      fi
    done < <(printf '%s\n' "${!drop_keys[@]}" "${!set_rows[@]}" "${!add_rows[@]}" | sort -u)

    : > "$new_file"
    present_tuples=()
    present_rows=()
    while IFS= read -r line || [[ -n "$line" ]]; do
      if [[ -z "$line" || "$line" =~ ^# ]]; then
        echo "$line" >> "$new_file"
        continue
      fi
      IFS=$'\t' read -r test mode_field profile expected extra <<<"$line"
      if [[ -n "${extra:-}" ]]; then
        echo "$line" >> "$new_file"
        continue
      fi
      if [[ -z "$test" || -z "$mode_field" || -z "$profile" || -z "$expected" ]]; then
        echo "$line" >> "$new_file"
        continue
      fi
      map_key="$file|$test|$mode_field|$profile"
      if [[ -n "${drop_keys["$map_key"]+x}" || -n "${set_rows["$map_key"]+x}" ]]; then
        continue
      fi
      echo "$line" >> "$new_file"
      tuple_key="$test|$mode_field|$profile"
      present_tuples["$tuple_key"]=1
      present_rows["$tuple_key|$expected"]=1
    done < "$file"

    append_rows=()
    while IFS= read -r mk; do
      if [[ "$mk" != "$file|"* ]]; then
        continue
      fi
      row_spec="${set_rows["$mk"]}"
      if [[ -z "$row_spec" ]]; then
        continue
      fi
      row_expected="${row_spec##*|}"
      row_key="${row_spec%|*}"
      IFS='|' read -r test mode_field profile <<<"$row_key"
      if [[ -z "$test" || -z "$mode_field" || -z "$profile" || -z "$row_expected" ]]; then
        continue
      fi
      tuple_key="$test|$mode_field|$profile"
      if [[ -n "${present_rows["$tuple_key|$row_expected"]+x}" ]]; then
        continue
      fi
      append_rows+=("$test"$'\t'"$mode_field"$'\t'"$profile"$'\t'"$row_expected")
      present_tuples["$tuple_key"]=1
      present_rows["$tuple_key|$row_expected"]=1
    done < <(printf '%s\n' "${!set_rows[@]}" | sort)
    if ((${#append_rows[@]} > 0)); then
      while IFS= read -r line; do
        echo "$line" >> "$new_file"
      done < <(printf '%s\n' "${append_rows[@]}" | sort -u)
    fi

    append_rows=()
    while IFS= read -r mk; do
      if [[ "$mk" != "$file|"* ]]; then
        continue
      fi
      row_spec="${mk#"$file|"}"
      if [[ -z "$row_spec" ]]; then
        continue
      fi
      row_expected="${row_spec##*|}"
      row_key="${row_spec%|*}"
      IFS='|' read -r test mode_field profile <<<"$row_key"
      if [[ -z "$test" || -z "$mode_field" || -z "$profile" || -z "$row_expected" ]]; then
        continue
      fi
      tuple_key="$test|$mode_field|$profile"
      map_key="$file|$tuple_key"
      if [[ -n "${drop_keys["$map_key"]+x}" || -n "${set_rows["$map_key"]+x}" ]]; then
        continue
      fi
      if [[ -n "${present_tuples["$tuple_key"]+x}" ]]; then
        continue
      fi
      append_rows+=("$test"$'\t'"$mode_field"$'\t'"$profile"$'\t'"$row_expected")
      present_tuples["$tuple_key"]=1
      present_rows["$tuple_key|$row_expected"]=1
    done < <(printf '%s\n' "${!add_rows[@]}" | sort)
    if ((${#append_rows[@]} > 0)); then
      while IFS= read -r line; do
        echo "$line" >> "$new_file"
      done < <(printf '%s\n' "${append_rows[@]}" | sort -u)
    fi

    changed=0
    if ! cmp -s "$old_file" "$new_file"; then
      changed=1
      changed_files=$((changed_files + 1))
      diff -u "$old_file" "$new_file" > "$diff_file" || true
      if [[ "$mode" == "apply" ]]; then
        cp "$new_file" "$file"
      fi
      append_apply_diff "$diff_file" "$file" "1"
    else
      append_apply_diff "$diff_file" "$file" "0"
    fi
    echo "EXPECT_LINT_APPLY: mode=$mode file=$file drop=$drop_count set=$set_count add=$add_count changed=$changed"
  done < <(printf '%s\n' "${file_list[@]}" | sort)
  echo "EXPECT_LINT_APPLY: mode=$mode files=${#file_list[@]} changed=$changed_files actions=$EXPECT_LINT_APPLY_ACTIONS addrow_filter=${EXPECT_LINT_APPLY_ADDROW_FILTER_FILE:-none}"
  return 0
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

sanitize_tsv_field() {
  local s="$1"
  s="${s//$'\t'/\\t}"
  s="${s//$'\n'/\\n}"
  printf '%s' "$s"
}

emit_malformed_fix_suggestion() {
  local file="$1"
  local original="$2"
  local suggested="$3"
  local reason="$4"
  local applied="$5"
  if [[ -z "$EXPECT_FORMAT_MALFORMED_FIX_FILE" ]]; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize_tsv_field "$file")" \
    "$(sanitize_tsv_field "$original")" \
    "$(sanitize_tsv_field "$suggested")" \
    "$(sanitize_tsv_field "$reason")" \
    "$applied" >> "$EXPECT_FORMAT_MALFORMED_FIX_FILE"
  return 0
}

emit_unfixable_format_issue() {
  local file="$1"
  local line="$2"
  local reason="$3"
  local severity="$4"
  local profile="$5"
  local strict_selected="$6"
  if [[ -z "$EXPECT_FORMAT_UNFIXABLE_FILE" ]]; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize_tsv_field "$file")" \
    "$(sanitize_tsv_field "$line")" \
    "$(sanitize_tsv_field "$reason")" \
    "$(sanitize_tsv_field "$severity")" \
    "$(sanitize_tsv_field "$profile")" \
    "$strict_selected" >> "$EXPECT_FORMAT_UNFIXABLE_FILE"
  return 0
}

csv_pattern_match() {
  local value="$1"
  local csv="$2"
  local pattern
  if [[ -z "$csv" ]]; then
    return 0
  fi
  while IFS= read -r pattern; do
    pattern="$(trim_whitespace "$pattern")"
    [[ -z "$pattern" ]] && continue
    if [[ "$value" == $pattern ]]; then
      return 0
    fi
  done < <(printf '%s\n' "$csv" | tr ',' '\n')
  return 1
}

infer_malformed_profile() {
  local raw_line="$1"
  local -a toks=()
  local profile="unknown"
  read -r -a toks <<<"$raw_line"
  if ((${#toks[@]} >= 3)); then
    profile="${toks[2],,}"
    [[ -z "$profile" ]] && profile="unknown"
  fi
  printf '%s' "$profile"
}

classify_unfixable_reason() {
  local raw_line="$1"
  local default_expected="$2"
  local allow_auto_omit="${3:-0}"
  local -a toks=()
  local expected
  read -r -a toks <<<"$raw_line"
  if ((${#toks[@]} < 2)); then
    echo "syntax-too-few-fields"
    return 0
  fi
  if ((${#toks[@]} == 2 || ${#toks[@]} == 3)); then
    if [[ -z "$default_expected" ]]; then
      echo "semantic-missing-default-expected"
      return 0
    fi
  fi
  if ((${#toks[@]} >= 4)); then
    expected="$(trim_whitespace "${toks[3]}")"
    expected="${expected,,}"
    case "$expected" in
      pass|fail|xfail|skip) ;;
      omit|auto)
        if [[ "$allow_auto_omit" != "1" ]]; then
          echo "policy-auto-omit-not-allowed"
          return 0
        fi
        ;;
      *)
        echo "semantic-invalid-expected-token"
        return 0
        ;;
    esac
  fi
  echo "no-canonical-rewrite"
  return 0
}

classify_unfixable_severity() {
  local reason="$1"
  case "$reason" in
    syntax-*|no-canonical-*) echo "error" ;;
    semantic-*|policy-*) echo "warning" ;;
    *) echo "error" ;;
  esac
  return 0
}

strict_unfixable_scope_matches() {
  local file="$1"
  local profile="$2"
  local reason="$3"
  local severity="$4"
  local base
  base="$(basename "$file")"
  if [[ -n "$STRICT_UNFIXABLE_FILE_FILTER" ]]; then
    if ! csv_pattern_match "$file" "$STRICT_UNFIXABLE_FILE_FILTER" && \
       ! csv_pattern_match "$base" "$STRICT_UNFIXABLE_FILE_FILTER"; then
      return 1
    fi
  fi
  if [[ -n "$STRICT_UNFIXABLE_PROFILE_FILTER" ]]; then
    if ! csv_pattern_match "$profile" "$STRICT_UNFIXABLE_PROFILE_FILTER"; then
      return 1
    fi
  fi
  if [[ -n "$STRICT_UNFIXABLE_REASON_FILTER" ]]; then
    if ! csv_pattern_match "$reason" "$STRICT_UNFIXABLE_REASON_FILTER"; then
      return 1
    fi
  fi
  if [[ -n "$STRICT_UNFIXABLE_SEVERITY_FILTER" ]]; then
    if ! csv_pattern_match "$severity" "$STRICT_UNFIXABLE_SEVERITY_FILTER"; then
      return 1
    fi
  fi
  return 0
}

suggest_malformed_expectation_row() {
  local raw_line="$1"
  local default_expected="$2"
  local allow_auto_omit="${3:-0}"
  local -a toks=()
  local test mode_field profile expected reason
  local suggested

  read -r -a toks <<<"$raw_line"
  if ((${#toks[@]} < 2)); then
    echo ""
    return 0
  fi
  test="${toks[0]}"
  mode_field=""
  profile=""
  expected=""
  reason=""

  case ${#toks[@]} in
    2)
      if [[ -z "$default_expected" ]]; then
        echo ""
        return 0
      fi
      mode_field="${toks[1]}"
      profile="*"
      expected="$default_expected"
      reason="fill-profile-expected-default"
      ;;
    3)
      mode_field="${toks[1]}"
      profile="${toks[2]}"
      if [[ -z "$default_expected" ]]; then
        echo ""
        return 0
      fi
      expected="$default_expected"
      reason="fill-expected-default"
      ;;
    *)
      mode_field="${toks[1]}"
      profile="${toks[2]}"
      expected="${toks[3]}"
      reason="trim-extra-fields"
      ;;
  esac

  mode_field="$(trim_whitespace "$mode_field")"
  profile="$(trim_whitespace "$profile")"
  expected="$(trim_whitespace "$expected")"
  [[ -z "$mode_field" ]] && mode_field="*"
  [[ -z "$profile" ]] && profile="*"
  [[ -z "$expected" ]] && expected="$default_expected"
  if [[ -z "$expected" ]]; then
    echo ""
    return 0
  fi
  mode_field="${mode_field,,}"
  profile="${profile,,}"
  expected="${expected,,}"
  case "$expected" in
    pass|fail|xfail|skip) ;;
    omit|auto)
      if [[ "$allow_auto_omit" != "1" ]]; then
        echo ""
        return 0
      fi
      ;;
    *)
      echo ""
      return 0
      ;;
  esac
  suggested="$test"$'\t'"$mode_field"$'\t'"$profile"$'\t'"$expected"
  printf '%s|%s\n' "$suggested" "$reason"
  return 0
}

append_expect_format_diff() {
  local diff_file="$1"
  local source_file="$2"
  local changed="$3"
  if [[ -z "$EXPECT_FORMAT_DIFF_FILE" ]]; then
    return 0
  fi
  if [[ "$changed" == "1" ]]; then
    {
      printf '=== %s ===\n' "$source_file"
      cat "$diff_file"
      printf '\n'
    } >> "$EXPECT_FORMAT_DIFF_FILE"
  fi
  return 0
}

resolve_expect_format_files() {
  local -n out_files="$1"
  local -n out_defaults="$2"
  local target resolved default_expected
  out_files=()
  out_defaults=()

  resolve_target() {
    local name="$1"
    case "$name" in
      EXPECT_FILE) echo "$EXPECT_FILE" ;;
      XFAIL_FILE) echo "$XFAIL_FILE" ;;
      EXPECT_REGEN_OVERRIDE_FILE) echo "$EXPECT_REGEN_OVERRIDE_FILE" ;;
      *) echo "$name" ;;
    esac
  }

  target_default() {
    local name="$1"
    local path="$2"
    if [[ "$name" == "XFAIL_FILE" || "$path" == "$XFAIL_FILE" ]]; then
      echo "xfail"
    else
      echo ""
    fi
  }

  if [[ -n "$EXPECT_FORMAT_FILES" ]]; then
    while IFS= read -r target; do
      [[ -z "$target" ]] && continue
      resolved="$(resolve_target "$target")"
      [[ -z "$resolved" || "$resolved" == "/dev/null" ]] && continue
      default_expected="$(target_default "$target" "$resolved")"
      out_files+=("$resolved")
      out_defaults+=("$default_expected")
    done < <(printf '%s\n' "$EXPECT_FORMAT_FILES" | tr ',' '\n')
    return 0
  fi

  if [[ -n "$EXPECT_FILE" && "$EXPECT_FILE" != "/dev/null" ]]; then
    out_files+=("$EXPECT_FILE")
    out_defaults+=("")
  fi
  if [[ -n "$EXPECT_REGEN_OVERRIDE_FILE" && "$EXPECT_REGEN_OVERRIDE_FILE" != "/dev/null" ]]; then
    out_files+=("$EXPECT_REGEN_OVERRIDE_FILE")
    out_defaults+=("")
  fi
  if [[ -n "$XFAIL_FILE" && "$XFAIL_FILE" != "/dev/null" ]]; then
    out_files+=("$XFAIL_FILE")
    out_defaults+=("xfail")
  fi
  return 0
}

format_expectation_file() {
  local file="$1"
  local default_expected="$2"
  local mode="$3"
  local idx="$4"
  local old_file="$tmpdir/expect-format-old-$idx.tsv"
  local new_file="$tmpdir/expect-format-new-$idx.tsv"
  local diff_file="$tmpdir/expect-format-diff-$idx.txt"
  local line trimmed
  local test mode_field profile expected extra
  local changed=0
  local rows=0
  local malformed=0
  local malformed_suggested=0
  local malformed_fixed=0
  local malformed_unfixable=0
  local malformed_unfixable_strict=0
  local allow_auto_omit=0
  local inferred_profile strict_selected unfix_reason unfix_severity
  local suggestion reason suggest_pair applied comment_blob rec_idx
  local -a pending_comments=()
  local -a row_records=()
  local -a row_comments=()
  local -a malformed_records=()
  local -a malformed_comments=()
  if [[ -n "$EXPECT_REGEN_OVERRIDE_FILE" ]] && [[ "$file" == "$EXPECT_REGEN_OVERRIDE_FILE" ]]; then
    allow_auto_omit=1
  fi

  if [[ ! -f "$file" ]]; then
    echo "SKIP|-1|-1"
    return 0
  fi

  cp "$file" "$old_file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    trimmed="$(trim_whitespace "$line")"
    if [[ -z "$trimmed" ]]; then
      continue
    fi
    if [[ "$trimmed" == \#* ]]; then
      pending_comments+=("$trimmed")
      continue
    fi
    test=""
    mode_field=""
    profile=""
    expected=""
    extra=""
    if [[ "$trimmed" == *$'\t'* ]]; then
      IFS=$'\t' read -r test mode_field profile expected extra <<<"$trimmed"
    else
      read -r test mode_field profile expected extra <<<"$trimmed"
    fi
    test="$(trim_whitespace "${test:-}")"
    mode_field="$(trim_whitespace "${mode_field:-}")"
    profile="$(trim_whitespace "${profile:-}")"
    expected="$(trim_whitespace "${expected:-}")"
    if [[ -n "${extra:-}" || -z "$test" ]]; then
      malformed=$((malformed + 1))
      suggest_pair="$(suggest_malformed_expectation_row "$trimmed" "$default_expected" "$allow_auto_omit")"
      if [[ -n "$suggest_pair" ]]; then
        suggestion="${suggest_pair%|*}"
        reason="${suggest_pair##*|}"
        malformed_suggested=$((malformed_suggested + 1))
        applied=0
        if [[ "$EXPECT_FORMAT_REWRITE_MALFORMED" == "1" ]]; then
          comment_blob=""
          if ((${#pending_comments[@]} > 0)); then
            comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
            comment_blob="${comment_blob%$'\n'}"
            pending_comments=()
          fi
          row_records+=("$suggestion")
          row_comments+=("$comment_blob")
          rows=$((rows + 1))
          malformed_fixed=$((malformed_fixed + 1))
          applied=1
        else
          comment_blob=""
          if ((${#pending_comments[@]} > 0)); then
            comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
            comment_blob="${comment_blob%$'\n'}"
            pending_comments=()
          fi
          malformed_records+=("$trimmed")
          malformed_comments+=("$comment_blob")
        fi
        emit_malformed_fix_suggestion "$file" "$trimmed" "$suggestion" "$reason" "$applied"
      else
        comment_blob=""
        if ((${#pending_comments[@]} > 0)); then
          comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
          comment_blob="${comment_blob%$'\n'}"
          pending_comments=()
        fi
        malformed_records+=("$trimmed")
        malformed_comments+=("$comment_blob")
        malformed_unfixable=$((malformed_unfixable + 1))
        unfix_reason="$(classify_unfixable_reason "$trimmed" "$default_expected" "$allow_auto_omit")"
        unfix_severity="$(classify_unfixable_severity "$unfix_reason")"
        inferred_profile="$(infer_malformed_profile "$trimmed")"
        strict_selected=0
        if strict_unfixable_scope_matches "$file" "$inferred_profile" "$unfix_reason" "$unfix_severity"; then
          strict_selected=1
          malformed_unfixable_strict=$((malformed_unfixable_strict + 1))
        fi
        emit_unfixable_format_issue "$file" "$trimmed" "$unfix_reason" "$unfix_severity" "$inferred_profile" "$strict_selected"
      fi
      continue
    fi
    [[ -z "$mode_field" ]] && mode_field="*"
    [[ -z "$profile" ]] && profile="*"
    [[ -z "$expected" ]] && expected="$default_expected"
    if [[ -z "$expected" ]]; then
      malformed=$((malformed + 1))
      suggest_pair="$(suggest_malformed_expectation_row "$trimmed" "$default_expected" "$allow_auto_omit")"
      if [[ -n "$suggest_pair" ]]; then
        suggestion="${suggest_pair%|*}"
        reason="${suggest_pair##*|}"
        malformed_suggested=$((malformed_suggested + 1))
        applied=0
        if [[ "$EXPECT_FORMAT_REWRITE_MALFORMED" == "1" ]]; then
          comment_blob=""
          if ((${#pending_comments[@]} > 0)); then
            comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
            comment_blob="${comment_blob%$'\n'}"
            pending_comments=()
          fi
          row_records+=("$suggestion")
          row_comments+=("$comment_blob")
          rows=$((rows + 1))
          malformed_fixed=$((malformed_fixed + 1))
          applied=1
        else
          comment_blob=""
          if ((${#pending_comments[@]} > 0)); then
            comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
            comment_blob="${comment_blob%$'\n'}"
            pending_comments=()
          fi
          malformed_records+=("$trimmed")
          malformed_comments+=("$comment_blob")
        fi
        emit_malformed_fix_suggestion "$file" "$trimmed" "$suggestion" "$reason" "$applied"
      else
        comment_blob=""
        if ((${#pending_comments[@]} > 0)); then
          comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
          comment_blob="${comment_blob%$'\n'}"
          pending_comments=()
        fi
        malformed_records+=("$trimmed")
        malformed_comments+=("$comment_blob")
        malformed_unfixable=$((malformed_unfixable + 1))
        unfix_reason="$(classify_unfixable_reason "$trimmed" "$default_expected" "$allow_auto_omit")"
        unfix_severity="$(classify_unfixable_severity "$unfix_reason")"
        inferred_profile="$(infer_malformed_profile "$trimmed")"
        strict_selected=0
        if strict_unfixable_scope_matches "$file" "$inferred_profile" "$unfix_reason" "$unfix_severity"; then
          strict_selected=1
          malformed_unfixable_strict=$((malformed_unfixable_strict + 1))
        fi
        emit_unfixable_format_issue "$file" "$trimmed" "$unfix_reason" "$unfix_severity" "$inferred_profile" "$strict_selected"
      fi
      continue
    fi
    mode_field="${mode_field,,}"
    profile="${profile,,}"
    expected="${expected,,}"
    comment_blob=""
    if ((${#pending_comments[@]} > 0)); then
      comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
      comment_blob="${comment_blob%$'\n'}"
      pending_comments=()
    fi
    row_records+=("$test"$'\t'"$mode_field"$'\t'"$profile"$'\t'"$expected")
    row_comments+=("$comment_blob")
    rows=$((rows + 1))
  done < "$file"

  if ((${#pending_comments[@]} > 0)); then
    comment_blob="$(printf '%s\n' "${pending_comments[@]}")"
    comment_blob="${comment_blob%$'\n'}"
    malformed_records+=("")
    malformed_comments+=("$comment_blob")
    pending_comments=()
  fi

  : > "$new_file"
  if ((${#row_records[@]} > 0)); then
    while IFS= read -r rec; do
      rec_idx="${rec##*$'\x1f'}"
      comment_blob="${row_comments[$((10#$rec_idx))]:-}"
      if [[ -n "$comment_blob" ]]; then
        printf '%s\n' "$comment_blob" >> "$new_file"
      fi
      echo "${row_records[$((10#$rec_idx))]}" >> "$new_file"
    done < <(
      for rec_idx in "${!row_records[@]}"; do
        printf '%s\x1f%08d\n' "${row_records[$rec_idx]}" "$rec_idx"
      done | LC_ALL=C sort
    )
  fi
  if ((${#malformed_records[@]} > 0)); then
    if [[ -s "$new_file" ]]; then
      echo >> "$new_file"
    fi
    for rec_idx in "${!malformed_records[@]}"; do
      comment_blob="${malformed_comments[$rec_idx]:-}"
      if [[ -n "$comment_blob" ]]; then
        printf '%s\n' "$comment_blob" >> "$new_file"
      fi
      if [[ -n "${malformed_records[$rec_idx]}" ]]; then
        printf '%s\n' "${malformed_records[$rec_idx]}" >> "$new_file"
      fi
    done
  fi

  if ! cmp -s "$old_file" "$new_file"; then
    changed=1
    diff -u "$old_file" "$new_file" > "$diff_file" || true
    if [[ "$mode" == "apply" ]]; then
      cp "$new_file" "$file"
    fi
  fi
  append_expect_format_diff "$diff_file" "$file" "$changed"
  echo "$changed|$rows|$malformed|$malformed_suggested|$malformed_fixed|$malformed_unfixable|$malformed_unfixable_strict"
  return 0
}

run_expectation_formatter() {
  local mode="$1"
  local -a files=()
  local -a defaults=()
  local i file default_expected result changed rows malformed malformed_suggested malformed_fixed malformed_unfixable malformed_unfixable_strict
  local changed_files=0
  local file_count=0
  local unfixable_total=0
  local unfixable_strict_total=0

  resolve_expect_format_files files defaults
  if [[ -n "$EXPECT_FORMAT_DIFF_FILE" ]]; then
    : > "$EXPECT_FORMAT_DIFF_FILE"
  fi

  for i in "${!files[@]}"; do
    file="${files[$i]}"
    default_expected="${defaults[$i]}"
    [[ -z "$file" || "$file" == "/dev/null" ]] && continue
    file_count=$((file_count + 1))
    result="$(format_expectation_file "$file" "$default_expected" "$mode" "$((i + 1))")"
    IFS='|' read -r changed rows malformed malformed_suggested malformed_fixed malformed_unfixable malformed_unfixable_strict <<<"$result"
    if [[ "$changed" == "SKIP" ]]; then
      echo "EXPECT_FORMAT: mode=$mode file=$file rows=0 malformed=0 changed=0 missing=1"
      continue
    fi
    if [[ "$changed" == "1" ]]; then
      changed_files=$((changed_files + 1))
    fi
    unfixable_total=$((unfixable_total + ${malformed_unfixable:-0}))
    unfixable_strict_total=$((unfixable_strict_total + ${malformed_unfixable_strict:-0}))
    echo "EXPECT_FORMAT: mode=$mode file=$file rows=$rows malformed=$malformed changed=$changed malformed_suggested=${malformed_suggested:-0} malformed_fixed=${malformed_fixed:-0} malformed_unfixable=${malformed_unfixable:-0} malformed_unfixable_strict=${malformed_unfixable_strict:-0}"
  done
  format_unfixable_issues="$unfixable_total"
  echo "EXPECT_FORMAT: mode=$mode files=$file_count changed=$changed_files malformed_unfixable=$unfixable_total malformed_unfixable_strict=$unfixable_strict_total"
  if [[ "$EXPECT_FORMAT_FAIL_ON_UNFIXABLE" == "1" ]]; then
    if ((unfixable_strict_total > 0)); then
      echo "EXPECT_FORMAT_STRICT: unfixable_total=$unfixable_total scoped=$unfixable_strict_total fail=1 policy=$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY file_filter=${STRICT_UNFIXABLE_FILE_FILTER:-*} profile_filter=${STRICT_UNFIXABLE_PROFILE_FILTER:-*} reason_filter=${STRICT_UNFIXABLE_REASON_FILTER:-*} severity_filter=${STRICT_UNFIXABLE_SEVERITY_FILTER:-*}"
      failures=$((failures + 1))
    else
      echo "EXPECT_FORMAT_STRICT: unfixable_total=$unfixable_total scoped=0 fail=0 policy=$EXPECT_FORMAT_FAIL_ON_UNFIXABLE_POLICY file_filter=${STRICT_UNFIXABLE_FILE_FILTER:-*} profile_filter=${STRICT_UNFIXABLE_PROFILE_FILTER:-*} reason_filter=${STRICT_UNFIXABLE_REASON_FILTER:-*} severity_filter=${STRICT_UNFIXABLE_SEVERITY_FILTER:-*}"
    fi
  fi
  return 0
}

load_expected_cases() {
  local map_name="$1"
  local file="$2"
  local default_expected="${3:-}"
  local -n out_map="$map_name"
  local -A seen_in_file=()
  [[ -f "$file" ]] || return 0
  while IFS=$'\t' read -r test mode profile expected; do
    [[ -z "$test" || "$test" =~ ^# ]] && continue
    [[ -z "$mode" ]] && mode="*"
    [[ -z "$profile" ]] && profile="*"
    [[ -z "$expected" ]] && expected="$default_expected"
    if [[ -z "$expected" ]]; then
      echo "warning: missing expected outcome in $file for $test|$mode|$profile" >&2
      continue
    fi
    expected="${expected,,}"
    case "$expected" in
      pass|fail|xfail|skip) ;;
      *)
        echo "warning: invalid expected outcome '$expected' in $file for $test|$mode|$profile" >&2
        continue
        ;;
    esac
    local key="$test|$mode|$profile"
    if [[ "$EXPECT_LINT" == "1" ]] && [[ -n "${seen_in_file["$key"]+x}" ]]; then
      if [[ "${seen_in_file["$key"]}" == "$expected" ]]; then
        emit_expect_lint "redundant" "$file row duplicates $key = $expected"
        emit_expect_lint_hint "redundant" "$file" "$key" "remove-duplicate-row" \
          "same value repeated"
        emit_expect_lint_fix "redundant" "$file" "drop-row" "$key" "" \
          "remove duplicated row"
      else
        emit_expect_lint "conflict" "$file row redefines $key: ${seen_in_file["$key"]} -> $expected"
        emit_expect_lint_hint "conflict" "$file" "$key" "resolve-key-conflict" \
          "${seen_in_file["$key"]} vs $expected"
        emit_expect_lint_fix "conflict" "$file" "set-row" "$key" \
          "$(format_row_spec "$key" "$expected")" \
          "keep final assignment and remove earlier conflicting rows"
      fi
    fi
    seen_in_file["$key"]="$expected"
    out_map["$test|$mode|$profile"]="$expected"
  done < "$file"
  return 0
}

# Load legacy xfail rows first, then let EXPECT_FILE override where needed.
if [[ "$EXPECT_FORMAT_MODE" != "off" ]]; then
  run_expectation_formatter "$EXPECT_FORMAT_MODE"
fi
if [[ -n "$XFAIL_FILE" ]]; then
  load_expected_cases expected_cases "$XFAIL_FILE" "xfail"
fi
load_expected_cases expected_cases "$EXPECT_FILE"

load_regen_override_cases() {
  local file="$1"
  local -A seen_in_file=()
  [[ -f "$file" ]] || return 0
  while IFS=$'\t' read -r test mode profile expected; do
    [[ -z "$test" || "$test" =~ ^# ]] && continue
    [[ -z "$mode" ]] && mode="*"
    [[ -z "$profile" ]] && profile="*"
    [[ -z "$expected" ]] && continue
    expected="${expected,,}"
    case "$expected" in
      pass|fail|xfail|skip|omit|auto) ;;
      *)
        echo "warning: invalid regen override '$expected' in $file for $test|$mode|$profile" >&2
        continue
        ;;
    esac
    local key="$test|$mode|$profile"
    if [[ "$EXPECT_LINT" == "1" ]] && [[ -n "${seen_in_file["$key"]+x}" ]]; then
      if [[ "${seen_in_file["$key"]}" == "$expected" ]]; then
        emit_expect_lint "redundant" "$file row duplicates $key = $expected"
        emit_expect_lint_hint "redundant" "$file" "$key" "remove-duplicate-row" \
          "same value repeated"
        emit_expect_lint_fix "redundant" "$file" "drop-row" "$key" "" \
          "remove duplicated row"
      else
        emit_expect_lint "conflict" "$file row redefines $key: ${seen_in_file["$key"]} -> $expected"
        emit_expect_lint_hint "conflict" "$file" "$key" "resolve-key-conflict" \
          "${seen_in_file["$key"]} vs $expected"
        emit_expect_lint_fix "conflict" "$file" "set-row" "$key" \
          "$(format_row_spec "$key" "$expected")" \
          "keep final assignment and remove earlier conflicting rows"
      fi
    fi
    seen_in_file["$key"]="$expected"
    regen_override_cases["$test|$mode|$profile"]="$expected"
  done < "$file"
  return 0
}

if [[ -n "$EXPECT_REGEN_OVERRIDE_FILE" ]]; then
  load_regen_override_cases "$EXPECT_REGEN_OVERRIDE_FILE"
fi

populate_suite_tests() {
  local sv
  for sv in "$YOSYS_SVA_DIR"/*.sv; do
    [[ -f "$sv" ]] || continue
    suite_tests["$(basename "$sv" .sv)"]=1
  done
}

lint_unknown_tests_for_map() {
  local map_name="$1"
  local label="$2"
  local -n map_ref="$map_name"
  local -a keys=()
  local key test mode profile
  for key in "${!map_ref[@]}"; do
    keys+=("$key")
  done
  if ((${#keys[@]} == 0)); then
    return
  fi
  while IFS= read -r key; do
    IFS='|' read -r test mode profile <<<"$key"
    [[ "$test" == "*" ]] && continue
    if [[ -z "${suite_tests["$test"]+x}" ]]; then
      emit_expect_lint "unknown-test" "$label references missing test '$test' via key $key"
      emit_expect_lint_hint "unknown-test" "$label" "$key" "rename-or-remove-row" \
        "test missing from suite"
      emit_expect_lint_fix "unknown-test" "$label" "drop-row" "$key" "" \
        "row targets a non-existent suite test"
    fi
  done < <(printf '%s\n' "${keys[@]}" | sort)
  return 0
}

lookup_match_key_for_tuple() {
  local map_name="$1"
  local test="$2"
  local mode="$3"
  local profile="$4"
  local -n map_ref="$map_name"
  local key
  for key in \
    "$test|$mode|$profile" \
    "$test|$mode|*" \
    "$test|*|$profile" \
    "$test|*|*" \
    "*|$mode|$profile" \
    "*|$mode|*" \
    "*|*|$profile" \
    "*|*|*"; do
    if [[ -n "${map_ref["$key"]+x}" ]]; then
      echo "$key"
      return
    fi
  done
  echo ""
}

lookup_match_keys_for_tuple() {
  local map_name="$1"
  local test="$2"
  local mode="$3"
  local profile="$4"
  local -n map_ref="$map_name"
  local key
  for key in \
    "$test|$mode|$profile" \
    "$test|$mode|*" \
    "$test|*|$profile" \
    "$test|*|*" \
    "*|$mode|$profile" \
    "*|$mode|*" \
    "*|*|$profile" \
    "*|*|*"; do
    if [[ -n "${map_ref["$key"]+x}" ]]; then
      echo "$key"
    fi
  done
}

lint_shadowed_patterns_for_map() {
  local map_name="$1"
  local label="$2"
  local -n map_ref="$map_name"
  local -a tests=()
  local -a keys=()
  local -A possible_hits=()
  local test mode profile matched key

  while IFS= read -r test; do
    tests+=("$test")
  done < <(printf '%s\n' "${!suite_tests[@]}" | sort)
  if ((${#tests[@]} == 0)); then
    return
  fi
  for key in "${!map_ref[@]}"; do
    keys+=("$key")
    possible_hits["$key"]=0
  done
  if ((${#keys[@]} == 0)); then
    return
  fi

  for test in "${tests[@]}"; do
    for mode in pass fail; do
      for profile in known xprop; do
        matched="$(lookup_match_key_for_tuple "$map_name" "$test" "$mode" "$profile")"
        if [[ -n "$matched" ]]; then
          possible_hits["$matched"]=$((possible_hits["$matched"] + 1))
        fi
      done
    done
  done

  while IFS= read -r key; do
    if [[ "$key" == *"*"* ]] && [[ "${possible_hits["$key"]}" -eq 0 ]]; then
      emit_expect_lint "shadowed" "$label wildcard key $key is never selected for suite matrix"
      emit_expect_lint_hint "shadowed" "$label" "$key" "remove-or-tighten-key" \
        "wildcard row is dead under precedence"
      emit_expect_lint_fix "shadowed" "$label" "drop-row" "$key" "" \
        "wildcard row is never selected under precedence"
    fi
  done < <(printf '%s\n' "${keys[@]}" | sort)
  return 0
}

lint_precedence_ambiguity_for_map() {
  local map_name="$1"
  local label="$2"
  local -n map_ref="$map_name"
  local -a tests=()
  local test mode profile winner winner_val other other_val tuple pair
  local -a matches=()
  local -A winner_hits=()
  local -A winner_first_tuple=()
  local -A pair_counts=()
  local -A pair_examples=()

  while IFS= read -r test; do
    tests+=("$test")
  done < <(printf '%s\n' "${!suite_tests[@]}" | sort)
  if ((${#tests[@]} == 0)); then
    return
  fi

  for test in "${tests[@]}"; do
    for mode in pass fail; do
      for profile in known xprop; do
        mapfile -t matches < <(lookup_match_keys_for_tuple "$map_name" "$test" "$mode" "$profile")
        if ((${#matches[@]} == 0)); then
          continue
        fi
        winner="${matches[0]}"
        winner_hits["$winner"]=$((winner_hits["$winner"] + 1))
        winner_val="${map_ref["$winner"]}"
        tuple="$test|$mode|$profile"
        if [[ -z "${winner_first_tuple["$winner"]+x}" ]]; then
          winner_first_tuple["$winner"]="$tuple"
        fi
        if ((${#matches[@]} < 2)); then
          continue
        fi
        for other in "${matches[@]:1}"; do
          other_val="${map_ref["$other"]}"
          if [[ "$winner_val" == "$other_val" ]]; then
            continue
          fi
          pair="$winner||$other"
          pair_counts["$pair"]=$((pair_counts["$pair"] + 1))
          if [[ -z "${pair_examples["$pair"]+x}" ]]; then
            pair_examples["$pair"]="$tuple"
          fi
        done
      done
    done
  done

  if ((${#pair_counts[@]} == 0)); then
    return
  fi
  while IFS= read -r pair; do
    local cnt
    cnt="${pair_counts["$pair"]}"
    winner="${pair%%||*}"
    other="${pair#*||}"
    if [[ "${winner_hits["$other"]:-0}" -eq 0 ]]; then
      continue
    fi
    emit_expect_lint "ambiguity" \
      "$label precedence-sensitive overlap: $winner (${map_ref["$winner"]}) overrides $other (${map_ref["$other"]}) for $cnt tuple(s), e.g. ${pair_examples["$pair"]}"
    emit_expect_lint_hint "ambiguity" "$label" "$winner||$other" "split-or-align-overlap" \
      "${pair_examples["$pair"]}"
    winner_val="${map_ref["$winner"]}"
    other_val="${map_ref["$other"]}"
    if [[ -n "${winner_first_tuple["$other"]+x}" ]]; then
      emit_expect_lint_fix "ambiguity" "$label" "add-row" "$other" \
        "$(format_row_spec "${winner_first_tuple["$other"]}" "$other_val")" \
        "preserve lower-precedence intent on explicit tuple"
    fi
    emit_expect_lint_fix "ambiguity" "$label" "set-row" "$other" \
      "$(format_row_spec "$other" "$winner_val")" \
      "align lower-precedence row with overriding row $winner"
  done < <(printf '%s\n' "${!pair_counts[@]}" | sort)
  return 0
}

if [[ "$EXPECT_LINT" == "1" ]]; then
  populate_suite_tests
  lint_unknown_tests_for_map expected_cases "EXPECT_FILE"
  lint_unknown_tests_for_map regen_override_cases "EXPECT_REGEN_OVERRIDE_FILE"
  lint_shadowed_patterns_for_map expected_cases "EXPECT_FILE"
  lint_shadowed_patterns_for_map regen_override_cases "EXPECT_REGEN_OVERRIDE_FILE"
  lint_precedence_ambiguity_for_map expected_cases "EXPECT_FILE"
  lint_precedence_ambiguity_for_map regen_override_cases "EXPECT_REGEN_OVERRIDE_FILE"
  if [[ "$EXPECT_LINT_APPLY_MODE" != "off" ]]; then
    apply_expect_lint_fixes "$EXPECT_LINT_FIXES_FILE" "$EXPECT_LINT_APPLY_MODE"
  fi
  echo "EXPECT_LINT_SUMMARY: issues=$lint_issues"
fi

write_expect_diff_line() {
  local line="$1"
  echo "$line"
  if [[ -n "$EXPECT_DIFF_FILE" ]]; then
    echo "$line" >> "$EXPECT_DIFF_FILE"
  fi
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  printf '%s' "$s"
}

emit_expectation_diff() {
  local baseline_file="$1"
  local baseline_default="$2"
  local -A baseline_cases=()
  local -a added_lines=()
  local -a removed_lines=()
  local -a changed_lines=()
  local -a added_keys=()
  local -a removed_keys=()
  local -a changed_keys=()

  if [[ ! -f "$baseline_file" ]]; then
    echo "warning: EXPECT_DIFF_BASELINE file not found: $baseline_file" >&2
    expect_diff_changed=1
    return
  fi
  load_expected_cases baseline_cases "$baseline_file" "$baseline_default"

  local key
  for key in "${!expected_cases[@]}"; do
    if [[ -z "${baseline_cases["$key"]+x}" ]]; then
      added_lines+=("$key -> ${expected_cases["$key"]}")
      added_keys+=("$key")
    elif [[ "${baseline_cases["$key"]}" != "${expected_cases["$key"]}" ]]; then
      changed_lines+=("$key ${baseline_cases["$key"]} -> ${expected_cases["$key"]}")
      changed_keys+=("$key")
    fi
  done
  for key in "${!baseline_cases[@]}"; do
    if [[ -z "${expected_cases["$key"]+x}" ]]; then
      removed_lines+=("$key (was ${baseline_cases["$key"]})")
      removed_keys+=("$key")
    fi
  done

  expect_diff_added=${#added_lines[@]}
  expect_diff_removed=${#removed_lines[@]}
  expect_diff_changed=${#changed_lines[@]}

  if [[ -n "$EXPECT_DIFF_FILE" ]]; then
    : > "$EXPECT_DIFF_FILE"
  fi
  write_expect_diff_line "expect-diff summary: added=$expect_diff_added, removed=$expect_diff_removed, changed=$expect_diff_changed"
  if ((${#added_lines[@]} > 0)); then
    while IFS= read -r line; do
      write_expect_diff_line "EXPECT_DIFF_ADDED: $line"
    done < <(printf '%s\n' "${added_lines[@]}" | sort)
  fi
  if ((${#removed_lines[@]} > 0)); then
    while IFS= read -r line; do
      write_expect_diff_line "EXPECT_DIFF_REMOVED: $line"
    done < <(printf '%s\n' "${removed_lines[@]}" | sort)
  fi
  if ((${#changed_lines[@]} > 0)); then
    while IFS= read -r line; do
      write_expect_diff_line "EXPECT_DIFF_CHANGED: $line"
    done < <(printf '%s\n' "${changed_lines[@]}" | sort)
  fi

  if [[ -n "$EXPECT_DIFF_TSV_FILE" ]]; then
    : > "$EXPECT_DIFF_TSV_FILE"
    printf 'kind\ttest\tmode\tprofile\told\tnew\n' >> "$EXPECT_DIFF_TSV_FILE"
    if ((${#added_keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile
        IFS='|' read -r test mode profile <<<"$key"
        printf 'added\t%s\t%s\t%s\t\t%s\n' \
          "$test" "$mode" "$profile" "${expected_cases["$key"]}" >> "$EXPECT_DIFF_TSV_FILE"
      done < <(printf '%s\n' "${added_keys[@]}" | sort)
    fi
    if ((${#removed_keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile
        IFS='|' read -r test mode profile <<<"$key"
        printf 'removed\t%s\t%s\t%s\t%s\t\n' \
          "$test" "$mode" "$profile" "${baseline_cases["$key"]}" >> "$EXPECT_DIFF_TSV_FILE"
      done < <(printf '%s\n' "${removed_keys[@]}" | sort)
    fi
    if ((${#changed_keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile
        IFS='|' read -r test mode profile <<<"$key"
        printf 'changed\t%s\t%s\t%s\t%s\t%s\n' \
          "$test" "$mode" "$profile" "${baseline_cases["$key"]}" "${expected_cases["$key"]}" \
          >> "$EXPECT_DIFF_TSV_FILE"
      done < <(printf '%s\n' "${changed_keys[@]}" | sort)
    fi
  fi

  if [[ -n "$EXPECT_DIFF_JSON_FILE" ]]; then
    : > "$EXPECT_DIFF_JSON_FILE"
    local -a json_entries=()
    local test mode profile old_val new_val
    if ((${#added_keys[@]} > 0)); then
      while IFS= read -r key; do
        IFS='|' read -r test mode profile <<<"$key"
        old_val=""
        new_val="${expected_cases["$key"]}"
        json_entries+=("    {\"kind\":\"added\",\"test\":\"$(json_escape "$test")\",\"mode\":\"$(json_escape "$mode")\",\"profile\":\"$(json_escape "$profile")\",\"old\":\"$(json_escape "$old_val")\",\"new\":\"$(json_escape "$new_val")\"}")
      done < <(printf '%s\n' "${added_keys[@]}" | sort)
    fi
    if ((${#removed_keys[@]} > 0)); then
      while IFS= read -r key; do
        IFS='|' read -r test mode profile <<<"$key"
        old_val="${baseline_cases["$key"]}"
        new_val=""
        json_entries+=("    {\"kind\":\"removed\",\"test\":\"$(json_escape "$test")\",\"mode\":\"$(json_escape "$mode")\",\"profile\":\"$(json_escape "$profile")\",\"old\":\"$(json_escape "$old_val")\",\"new\":\"$(json_escape "$new_val")\"}")
      done < <(printf '%s\n' "${removed_keys[@]}" | sort)
    fi
    if ((${#changed_keys[@]} > 0)); then
      while IFS= read -r key; do
        IFS='|' read -r test mode profile <<<"$key"
        old_val="${baseline_cases["$key"]}"
        new_val="${expected_cases["$key"]}"
        json_entries+=("    {\"kind\":\"changed\",\"test\":\"$(json_escape "$test")\",\"mode\":\"$(json_escape "$mode")\",\"profile\":\"$(json_escape "$profile")\",\"old\":\"$(json_escape "$old_val")\",\"new\":\"$(json_escape "$new_val")\"}")
      done < <(printf '%s\n' "${changed_keys[@]}" | sort)
    fi

    {
      printf '{\n'
      printf '  "summary": {"added": %d, "removed": %d, "changed": %d},\n' \
        "$expect_diff_added" "$expect_diff_removed" "$expect_diff_changed"
      printf '  "entries": [\n'
      local i
      for i in "${!json_entries[@]}"; do
        if (( i + 1 < ${#json_entries[@]} )); then
          printf '%s,\n' "${json_entries[$i]}"
        else
          printf '%s\n' "${json_entries[$i]}"
        fi
      done
      printf '  ]\n'
      printf '}\n'
    } > "$EXPECT_DIFF_JSON_FILE"
  fi
}

case_profile() {
  if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    echo "known"
  else
    echo "xprop"
  fi
}

lookup_expected_case() {
  local test="$1"
  local mode="$2"
  local profile="$3"
  local key
  for key in \
    "$test|$mode|$profile" \
    "$test|$mode|*" \
    "$test|*|$profile" \
    "$test|*|*" \
    "*|$mode|$profile" \
    "*|$mode|*" \
    "*|*|$profile" \
    "*|*|*"; do
    if [[ -n "${expected_cases["$key"]+x}" ]]; then
      echo "${expected_cases["$key"]}"
      return
    fi
  done
  echo "pass"
}

lookup_regen_override_case() {
  local test="$1"
  local mode="$2"
  local profile="$3"
  local key
  for key in \
    "$test|$mode|$profile" \
    "$test|$mode|*" \
    "$test|*|$profile" \
    "$test|*|*" \
    "*|$mode|$profile" \
    "*|$mode|*" \
    "*|*|$profile" \
    "*|*|*"; do
    if [[ -n "${regen_override_cases["$key"]+x}" ]]; then
      echo "${regen_override_cases["$key"]}"
      return
    fi
  done
  echo ""
}

record_observed_case() {
  local test="$1"
  local mode="$2"
  local profile="$3"
  local passed="$4"
  local observed="fail"
  if [[ "$passed" == "1" ]]; then
    observed="pass"
  fi
  observed_cases["$test|$mode|$profile"]="$observed"
}

record_skipped_case() {
  local test="$1"
  local mode="$2"
  local profile="$3"
  observed_cases["$test|$mode|$profile"]="skip"
}

map_observed_to_regen_expected() {
  local test="$1"
  local mode="$2"
  local profile="$3"
  local observed="$4"
  local override
  override="$(lookup_regen_override_case "$test" "$mode" "$profile")"
  if [[ -n "$override" && "$override" != "auto" ]]; then
    case "$override" in
      pass|fail|xfail|skip)
        echo "$override"
        return
        ;;
      omit)
        echo ""
        return
        ;;
    esac
  fi
  local policy
  case "$observed" in
    pass)
      echo "pass"
      return
      ;;
    fail)
      policy="$EXPECT_REGEN_FAIL_POLICY"
      ;;
    skip)
      policy="$EXPECT_REGEN_SKIP_POLICY"
      ;;
    *)
      echo "warning: unexpected observed outcome '$observed'" >&2
      echo ""
      return
      ;;
  esac
  case "$policy" in
    pass|fail|xfail|skip)
      echo "$policy"
      ;;
    omit)
      echo ""
      ;;
    *)
      echo "warning: invalid regen policy '$policy' for observed '$observed'" >&2
      echo ""
      ;;
  esac
}

emit_observed_outputs() {
  local -a keys=()
  local key
  for key in "${!observed_cases[@]}"; do
    keys+=("$key")
  done
  if [[ -n "$EXPECT_OBSERVED_FILE" ]]; then
    : > "$EXPECT_OBSERVED_FILE"
    printf '# test_name\tmode\tprofile\tobserved\n' >> "$EXPECT_OBSERVED_FILE"
    if ((${#keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile observed
        IFS='|' read -r test mode profile <<<"$key"
        observed="${observed_cases["$key"]}"
        if [[ "$observed" == "skip" && "$EXPECT_OBSERVED_INCLUDE_SKIPPED" != "1" ]]; then
          continue
        fi
        printf '%s\t%s\t%s\t%s\n' \
          "$test" "$mode" "$profile" "$observed" >> "$EXPECT_OBSERVED_FILE"
      done < <(printf '%s\n' "${keys[@]}" | sort)
    fi
  fi
  if [[ -n "$EXPECT_REGEN_FILE" ]]; then
    : > "$EXPECT_REGEN_FILE"
    printf '# test_name\tmode\tprofile\texpected\n' >> "$EXPECT_REGEN_FILE"
    printf '# generated from observed outcomes with policies:\n' >> "$EXPECT_REGEN_FILE"
    printf '#   EXPECT_REGEN_FAIL_POLICY=%s\n' "$EXPECT_REGEN_FAIL_POLICY" >> "$EXPECT_REGEN_FILE"
    printf '#   EXPECT_REGEN_SKIP_POLICY=%s\n' "$EXPECT_REGEN_SKIP_POLICY" >> "$EXPECT_REGEN_FILE"
    if ((${#keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile observed expected
        IFS='|' read -r test mode profile <<<"$key"
        observed="${observed_cases["$key"]}"
        expected="$(map_observed_to_regen_expected "$test" "$mode" "$profile" "$observed")"
        if [[ -z "$expected" ]]; then
          continue
        fi
        printf '%s\t%s\t%s\t%s\n' \
          "$test" "$mode" "$profile" "$expected" >> "$EXPECT_REGEN_FILE"
      done < <(printf '%s\n' "${keys[@]}" | sort)
    fi
  fi
}

emit_mode_summary_outputs() {
  local generated_at
  generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local run_id="$YOSYS_SVA_MODE_SUMMARY_RUN_ID"
  if [[ -z "$run_id" ]]; then
    run_id="$generated_at"
  fi
  local tsv_header
  tsv_header='schema_version	run_id	generated_at_utc	test_total	test_failures	test_xfail	test_xpass	test_skipped	mode_total	mode_pass	mode_fail	mode_xfail	mode_xpass	mode_epass	mode_efail	mode_unskip	mode_skipped	mode_skip_pass	mode_skip_fail	mode_skip_expected	mode_skip_unexpected	skip_reason_vhdl	skip_reason_fail-no-macro	skip_reason_no-property	skip_reason_other'
  local legacy_tsv_header
  legacy_tsv_header='test_total	test_failures	test_xfail	test_xpass	test_skipped	mode_total	mode_pass	mode_fail	mode_xfail	mode_xpass	mode_epass	mode_efail	mode_unskip	mode_skipped	mode_skip_pass	mode_skip_fail	mode_skip_expected	mode_skip_unexpected	skip_reason_vhdl	skip_reason_fail-no-macro	skip_reason_no-property	skip_reason_other'
  local tsv_columns=25
  local legacy_tsv_columns=22
  local json_validator_mode="$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR"
  if [[ "$json_validator_mode" == "auto" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      json_validator_mode="python"
    else
      echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR=auto requires python3 in PATH" >&2
      exit 1
    fi
  fi
  if [[ "$json_validator_mode" == "python" ]] && ! command -v python3 >/dev/null 2>&1; then
    echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_JSON_VALIDATOR=python requires python3 in PATH" >&2
    exit 1
  fi
  local jsonl_migration_mode="$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE"
  if [[ "$jsonl_migration_mode" == "auto" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      jsonl_migration_mode="python"
    else
      jsonl_migration_mode="shell"
    fi
  fi
  if [[ "$jsonl_migration_mode" == "python" ]] && ! command -v python3 >/dev/null 2>&1; then
    echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_MIGRATION_MODE=python requires python3 in PATH" >&2
    exit 1
  fi
  local jsonl_migration_mode_resolved="$jsonl_migration_mode"
  local tsv_row
  local history_drop_future_tsv=0
  local history_drop_future_jsonl=0
  local history_drop_age_tsv=0
  local history_drop_age_jsonl=0
  local history_drop_max_entries_tsv=0
  local history_drop_max_entries_jsonl=0
  local drop_events_lock_file="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_FILE"
  local drop_events_lock_backend="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND"
  local drop_events_lock_timeout_secs="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_TIMEOUT_SECS"
  local drop_events_lock_stale_secs="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_STALE_SECS"
  local drop_events_id_hash_mode="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH"
  local drop_events_event_id_policy="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY"
  local drop_events_id_metadata_policy="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY"
  local drop_events_rewrite_run_id_regex="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_RUN_ID_REGEX"
  local drop_events_rewrite_reason_regex="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_REASON_REGEX"
  local drop_events_rewrite_schema_version_regex="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_REGEX"
  local drop_events_rewrite_history_file_regex="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_REGEX"
  local drop_events_rewrite_schema_version_list="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_LIST"
  local drop_events_rewrite_history_file_list="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_LIST"
  local drop_events_rewrite_selector_mode="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MODE"
  local drop_events_rewrite_selector_clauses_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_CLAUSES_JSON"
  local drop_events_rewrite_selector_macros_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON"
  local drop_events_rewrite_selector_profiles_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILES_JSON"
  local drop_events_rewrite_selector_profile_list="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_LIST"
  local drop_events_rewrite_selector_profile_default_list="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_DEFAULT_LIST"
  local drop_events_rewrite_selector_profile_overlay_list="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_OVERLAY_LIST"
  local drop_events_rewrite_selector_profile_route="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE"
  local drop_events_rewrite_selector_profile_routes_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON"
  local drop_events_rewrite_selector_profile_route_auto_mode="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE"
  local drop_events_rewrite_selector_profile_route_context_ci_provider="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_PROVIDER"
  local drop_events_rewrite_selector_profile_route_context_ci_job="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_JOB"
  local drop_events_rewrite_selector_profile_route_context_ci_branch="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_BRANCH"
  local drop_events_rewrite_selector_profile_route_context_ci_target="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_CI_TARGET"
  local drop_events_rewrite_selector_profile_route_context_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON"
  local drop_events_rewrite_selector_profile_route_context_schema_json="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_JSON"
  local drop_events_rewrite_selector_profile_route_context_schema_version="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_VERSION"
  local drop_events_rewrite_row_generated_at_utc_min="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MIN"
  local drop_events_rewrite_row_generated_at_utc_max="$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MAX"
  local drop_events_route_context_ci_provider=""
  local drop_events_route_context_ci_job=""
  local drop_events_route_context_ci_branch=""
  local drop_events_route_context_ci_target=""
  local drop_events_id_hash_mode_effective
  local drop_events_id_hash_algorithm
  local drop_events_id_hash_version
  local drop_events_lock_warned=0
  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE" ]] && [[ -z "$drop_events_lock_file" ]]; then
    drop_events_lock_file="${YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE}.lock"
  fi
  resolve_drop_events_id_hash_mode_effective() {
    case "$drop_events_id_hash_mode" in
      cksum)
        printf 'cksum\n'
        ;;
      crc32)
        printf 'crc32\n'
        ;;
      auto)
        if command -v cksum >/dev/null 2>&1; then
          printf 'cksum\n'
        elif command -v python3 >/dev/null 2>&1; then
          printf 'crc32\n'
        else
          printf 'unavailable\n'
        fi
        ;;
      *)
        printf 'unavailable\n'
        ;;
    esac
  }
  drop_events_id_hash_mode_effective="$(resolve_drop_events_id_hash_mode_effective)"
  resolve_profile_route_context_ci_provider() {
    if [[ -n "$drop_events_rewrite_selector_profile_route_context_ci_provider" ]]; then
      printf '%s\n' "$drop_events_rewrite_selector_profile_route_context_ci_provider"
      return
    fi
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
      printf 'github\n'
      return
    fi
    if [[ "${GITLAB_CI:-}" == "true" ]]; then
      printf 'gitlab\n'
      return
    fi
    if [[ "${BUILDKITE:-}" == "true" ]]; then
      printf 'buildkite\n'
      return
    fi
    if [[ -n "${JENKINS_URL:-}" ]]; then
      printf 'jenkins\n'
      return
    fi
    if [[ -n "${CI:-}" ]]; then
      printf 'ci\n'
      return
    fi
    printf '\n'
  }
  resolve_profile_route_context_ci_job() {
    if [[ -n "$drop_events_rewrite_selector_profile_route_context_ci_job" ]]; then
      printf '%s\n' "$drop_events_rewrite_selector_profile_route_context_ci_job"
      return
    fi
    local candidate
    for candidate in "${GITHUB_JOB:-}" "${GITHUB_WORKFLOW:-}" "${CI_JOB_NAME:-}" "${BUILDKITE_LABEL:-}" "${JOB_NAME:-}"; do
      if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
      fi
    done
    printf '\n'
  }
  resolve_profile_route_context_ci_branch() {
    if [[ -n "$drop_events_rewrite_selector_profile_route_context_ci_branch" ]]; then
      printf '%s\n' "$drop_events_rewrite_selector_profile_route_context_ci_branch"
      return
    fi
    local candidate
    for candidate in "${GITHUB_REF_NAME:-}" "${CI_COMMIT_REF_NAME:-}" "${BUILDKITE_BRANCH:-}" "${GIT_BRANCH:-}" "${BRANCH_NAME:-}"; do
      if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
      fi
    done
    printf '\n'
  }
  resolve_profile_route_context_ci_target() {
    if [[ -n "$drop_events_rewrite_selector_profile_route_context_ci_target" ]]; then
      printf '%s\n' "$drop_events_rewrite_selector_profile_route_context_ci_target"
      return
    fi
    if [[ -n "$TEST_FILTER" ]]; then
      printf '%s\n' "$TEST_FILTER"
      return
    fi
    printf '\n'
  }
  drop_events_route_context_ci_provider="$(resolve_profile_route_context_ci_provider)"
  drop_events_route_context_ci_job="$(resolve_profile_route_context_ci_job)"
  drop_events_route_context_ci_branch="$(resolve_profile_route_context_ci_branch)"
  drop_events_route_context_ci_target="$(resolve_profile_route_context_ci_target)"
  case "$drop_events_id_hash_mode_effective" in
    cksum)
      drop_events_id_hash_algorithm="cksum"
      drop_events_id_hash_version=1
      ;;
    crc32)
      drop_events_id_hash_algorithm="crc32"
      drop_events_id_hash_version=1
      ;;
    *)
      drop_events_id_hash_algorithm="unavailable"
      drop_events_id_hash_version=0
      ;;
  esac
  printf -v tsv_row '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d' \
    "$YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION" "$run_id" "$generated_at" \
    "$total" "$failures" "$xfails" "$xpasses" "$skipped" \
    "$mode_total" "$mode_out_pass" "$mode_out_fail" "$mode_out_xfail" \
    "$mode_out_xpass" "$mode_out_epass" "$mode_out_efail" "$mode_out_unskip" \
    "$mode_skipped" "$mode_skipped_pass" "$mode_skipped_fail" \
    "$mode_skipped_expected" "$mode_skipped_unexpected" \
    "$mode_skip_reason_vhdl" "$mode_skip_reason_fail_no_macro" \
    "$mode_skip_reason_no_property" "$mode_skip_reason_other"

  is_non_negative_int() {
    local value="$1"
    [[ "$value" =~ ^[0-9]+$ ]]
  }

  validate_history_tsv_row() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    local -a cols=()
    local i
    IFS=$'\t' read -r -a cols <<<"$line"
    if ((${#cols[@]} != tsv_columns)); then
      echo "error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE row in $file at line $lineno: expected $tsv_columns columns" >&2
      exit 1
    fi
    if ! is_non_negative_int "${cols[0]}"; then
      echo "error: invalid schema_version in $file at line $lineno: ${cols[0]}" >&2
      exit 1
    fi
    if [[ -z "${cols[1]}" ]]; then
      echo "error: empty run_id in $file at line $lineno" >&2
      exit 1
    fi
    if [[ ! "${cols[2]}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
      echo "error: invalid generated_at_utc in $file at line $lineno: ${cols[2]}" >&2
      exit 1
    fi
    for ((i = 3; i < tsv_columns; ++i)); do
      if ! is_non_negative_int "${cols[$i]}"; then
        echo "error: invalid numeric field in $file at line $lineno: column $((i + 1)) value '${cols[$i]}'" >&2
        exit 1
      fi
    done
  }

  validate_legacy_tsv_row() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    local -a cols=()
    local i
    IFS=$'\t' read -r -a cols <<<"$line"
    if ((${#cols[@]} != legacy_tsv_columns)); then
      echo "error: invalid legacy YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE row in $file at line $lineno: expected $legacy_tsv_columns columns" >&2
      exit 1
    fi
    for ((i = 0; i < legacy_tsv_columns; ++i)); do
      if ! is_non_negative_int "${cols[$i]}"; then
        echo "error: invalid numeric field in legacy row $file line $lineno: column $((i + 1)) value '${cols[$i]}'" >&2
        exit 1
      fi
    done
  }

  validate_history_jsonl_line_python() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    python3 - "$file" "$lineno" "$line" <<'PY'
import json
import re
import sys

file = sys.argv[1]
lineno = sys.argv[2]
line = sys.argv[3]

def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)

try:
    def no_duplicate_object_pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key '{key}'")
            result[key] = value
        return result

    obj = json.loads(line, object_pairs_hook=no_duplicate_object_pairs_hook)
except ValueError as ex:
    if "duplicate key" in str(ex):
        fail(f"error: duplicate key in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}: {ex}")
    fail(f"error: invalid JSON object in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
except Exception:
    fail(f"error: invalid JSON object in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")

if not isinstance(obj, dict):
    fail(f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line in {file} at line {lineno}: expected JSON object")

schema = obj.get("schema_version")
if isinstance(schema, bool):
    fail(f"error: invalid JSONL schema_version in {file} at line {lineno}")
if isinstance(schema, int):
    if schema < 0:
        fail(f"error: invalid JSONL schema_version in {file} at line {lineno}")
elif isinstance(schema, str):
    if not schema.isdigit():
        fail(f"error: invalid JSONL schema_version in {file} at line {lineno}")
else:
    fail(f"error: invalid JSONL schema_version in {file} at line {lineno}")

run_id = obj.get("run_id")
if not isinstance(run_id, str) or not run_id:
    fail(f"error: invalid run_id in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")

generated_at_utc = obj.get("generated_at_utc")
if not isinstance(generated_at_utc, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", generated_at_utc):
    fail(f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")

required_top = ["run_id", "generated_at_utc", "test_summary", "mode_summary", "skip_reasons"]
for key in required_top:
    if key not in obj:
        fail(f"error: missing key '{key}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")

required_sections = {
    "test_summary": ["total", "failures", "xfail", "xpass", "skipped"],
    "mode_summary": ["total", "pass", "fail", "xfail", "xpass", "epass", "efail", "unskip", "skipped", "skip_pass", "skip_fail", "skip_expected", "skip_unexpected"],
    "skip_reasons": ["vhdl", "fail_no_macro", "no_property", "other"],
}

for section, fields in required_sections.items():
    payload = obj.get(section)
    if not isinstance(payload, dict):
        fail(f"error: invalid object '{section}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    for field in fields:
        field_name = f"{section}.{field}"
        if field not in payload:
            fail(f"error: missing summary field '{field_name}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"error: invalid numeric summary field '{field_name}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")

drop_events_summary = obj.get("drop_events_summary")
if drop_events_summary is not None:
    if not isinstance(drop_events_summary, dict):
        fail(f"error: invalid object 'drop_events_summary' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    if "total" not in drop_events_summary:
        fail(f"error: missing summary field 'drop_events_summary.total' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    total = drop_events_summary["total"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        fail(f"error: invalid numeric summary field 'drop_events_summary.total' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    reasons = drop_events_summary.get("reasons")
    if not isinstance(reasons, dict):
        fail(f"error: invalid object 'drop_events_summary.reasons' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    future_skew = reasons.get("future_skew")
    if isinstance(future_skew, bool) or not isinstance(future_skew, int) or future_skew < 0:
        fail(f"error: invalid numeric summary field 'drop_events_summary.reasons.future_skew' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    for field in ("age_retention", "max_entries"):
        value = reasons.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"error: invalid numeric summary field 'drop_events_summary.reasons.{field}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    history_format = drop_events_summary.get("history_format")
    if not isinstance(history_format, dict):
        fail(f"error: invalid object 'drop_events_summary.history_format' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    for field in ("tsv", "jsonl"):
        value = history_format.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"error: invalid numeric summary field 'drop_events_summary.history_format.{field}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    id_hash_mode = drop_events_summary.get("id_hash_mode")
    if id_hash_mode is not None:
        if not isinstance(id_hash_mode, str) or id_hash_mode not in ("auto", "cksum", "crc32", "unavailable"):
            fail(f"error: invalid summary field 'drop_events_summary.id_hash_mode' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    id_hash_algorithm = drop_events_summary.get("id_hash_algorithm")
    if id_hash_algorithm is not None:
        if not isinstance(id_hash_algorithm, str) or id_hash_algorithm not in ("cksum", "crc32", "unavailable"):
            fail(f"error: invalid summary field 'drop_events_summary.id_hash_algorithm' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
    id_hash_version = drop_events_summary.get("id_hash_version")
    if id_hash_version is not None:
        if isinstance(id_hash_version, bool) or not isinstance(id_hash_version, int) or id_hash_version < 0:
            fail(f"error: invalid summary field 'drop_events_summary.id_hash_version' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}")
PY
  }

  validate_history_jsonl_line() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    validate_history_jsonl_line_python "$line" "$file" "$lineno"
  }

  extract_history_jsonl_generated_at_and_run_id() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    python3 - "$line" "$file" "$lineno" <<'PY'
import json
import sys

line = sys.argv[1]
file = sys.argv[2]
lineno = sys.argv[3]

try:
    obj = json.loads(line)
except Exception:
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

if not isinstance(obj, dict):
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

generated_at = obj.get("generated_at_utc")
if not isinstance(generated_at, str) or not generated_at:
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

run_id = obj.get("run_id")
if not isinstance(run_id, str):
    run_id = ""

print(generated_at)
print(run_id)
PY
  }

  migrate_history_jsonl_line() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    local payload migrated

    if [[ "$jsonl_migration_mode_resolved" == "python" ]]; then
      python3 - "$line" "$file" "$lineno" <<'PY'
import json
import sys

line = sys.argv[1]
file = sys.argv[2]
lineno = sys.argv[3]

invalid_json_msg = (
    f"error: invalid JSON object in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE "
    f"{file} at line {lineno}"
)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def no_duplicate_object_pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key '{key}'")
        result[key] = value
    return result


try:
    obj = json.loads(line, object_pairs_hook=no_duplicate_object_pairs_hook)
except ValueError as ex:
    if "duplicate key" in str(ex):
        fail(
            f"error: duplicate key in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE {file} at line {lineno}: {ex}"
        )
    fail(invalid_json_msg)
except Exception:
    fail(invalid_json_msg)

if not isinstance(obj, dict):
    fail(
        f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line in {file} at line {lineno}: expected JSON object"
    )

if "schema_version" in obj:
    print(line)
    sys.exit(0)

migrated = {
    "schema_version": "0",
    "run_id": f"legacy-{lineno}",
    "generated_at_utc": "1970-01-01T00:00:00Z",
}
for key, value in obj.items():
    if key not in migrated:
        migrated[key] = value

print(json.dumps(migrated, separators=(",", ":")))
PY
      return
    fi

    jsonl_line_has_top_level_key_shell() {
      local line="$1"
      local key="$2"
      awk -v target="$key" '
BEGIN {
  depth = 0
  in_string = 0
  escaped = 0
  token = ""
  candidate = 0
  found = 0
  last_sig = ""
}
{
  line = $0
  n = length(line)
  for (i = 1; i <= n; ++i) {
    c = substr(line, i, 1)
    if (in_string) {
      if (escaped) {
        token = token c
        escaped = 0
        continue
      }
      if (c == "\\") {
        token = token c
        escaped = 1
        continue
      }
      if (c == "\"") {
        in_string = 0
        if (candidate) {
          j = i + 1
          while (j <= n) {
            d = substr(line, j, 1)
            if (d ~ /[[:space:]]/) {
              ++j
              continue
            }
            break
          }
          if (j <= n && substr(line, j, 1) == ":" && token == target) {
            found = 1
            break
          }
        }
        candidate = 0
        continue
      }
      token = token c
      continue
    }
    if (c ~ /[[:space:]]/)
      continue
    if (c == "\"") {
      in_string = 1
      token = ""
      candidate = (depth == 1 && (last_sig == "{" || last_sig == ","))
      continue
    }
    if (c == "{") {
      ++depth
      if (depth == 1)
        last_sig = "{"
      continue
    }
    if (c == "}") {
      if (depth == 1)
        last_sig = "}"
      if (depth > 0)
        --depth
      continue
    }
    if (depth == 1 && c == ",") {
      last_sig = ","
      continue
    }
    if (depth == 1 && c == ":") {
      last_sig = ":"
      continue
    }
    if (depth == 1)
      last_sig = c
  }
}
END {
  exit(found ? 0 : 1)
}' <<<"$line"
    }

    if jsonl_line_has_top_level_key_shell "$line" "schema_version"; then
      printf '%s\n' "$line"
      return
    fi
    if [[ "$line" != *"{"* || "$line" != *"}"* ]]; then
      echo "error: unsupported YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line format in $file at line $lineno" >&2
      exit 1
    fi
    payload="$(printf '%s' "$line" | sed -E 's/^[[:space:]]*\{//; s/\}[[:space:]]*$//')"
    if [[ -n "$payload" ]]; then
      printf -v migrated '{"schema_version":"0","run_id":"legacy-%d","generated_at_utc":"1970-01-01T00:00:00Z",%s}' "$lineno" "$payload"
    else
      printf -v migrated '{"schema_version":"0","run_id":"legacy-%d","generated_at_utc":"1970-01-01T00:00:00Z"}' "$lineno"
    fi
    printf '%s\n' "$migrated"
  }

  utc_to_epoch() {
    local timestamp="$1"
    local epoch
    if epoch="$(date -u -d "$timestamp" +%s 2>/dev/null)"; then
      printf '%s\n' "$epoch"
      return 0
    fi
    if epoch="$(date -ju -f '%Y-%m-%dT%H:%M:%SZ' "$timestamp" +%s 2>/dev/null)"; then
      printf '%s\n' "$epoch"
      return 0
    fi
    return 1
  }

  json_escape() {
    local value="$1"
    printf '%s' "$value" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
  }

  extract_drop_event_generated_at_utc() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    python3 - "$line" "$file" "$lineno" <<'PY'
import json
import sys

line = sys.argv[1]
file = sys.argv[2]
lineno = sys.argv[3]

try:
    obj = json.loads(line)
except Exception:
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

if not isinstance(obj, dict):
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

value = obj.get("generated_at_utc")
if not isinstance(value, str) or not value:
    print(
        f"error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}",
        file=sys.stderr,
    )
    sys.exit(1)

print(value)
PY
  }

  stable_event_id() {
    local key="$1"
    local digest
    case "$drop_events_id_hash_mode" in
      cksum)
        if ! command -v cksum >/dev/null 2>&1; then
          echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH=cksum requires cksum in PATH" >&2
          exit 1
        fi
        if ! digest="$(printf '%s' "$key" | cksum | awk '{print $1}')"; then
          echo "error: failed to derive drop-event id via cksum" >&2
          exit 1
        fi
        ;;
      crc32)
        if ! command -v python3 >/dev/null 2>&1; then
          echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH=crc32 requires python3 in PATH" >&2
          exit 1
        fi
        digest="$(python3 - "$key" <<'PY'
import sys
import zlib

key = sys.argv[1]
print(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
PY
)"
        ;;
      auto)
        if command -v cksum >/dev/null 2>&1; then
          if ! digest="$(printf '%s' "$key" | cksum | awk '{print $1}')"; then
            echo "error: failed to derive drop-event id via cksum" >&2
            exit 1
          fi
        elif command -v python3 >/dev/null 2>&1; then
          digest="$(python3 - "$key" <<'PY'
import sys
import zlib

key = sys.argv[1]
print(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
PY
)"
        else
          echo "error: stable drop-event IDs require cksum or python3 in PATH" >&2
          exit 1
        fi
        ;;
      *)
        echo "error: unsupported drop-event id hash mode: $drop_events_id_hash_mode" >&2
        exit 1
        ;;
    esac
    printf 'drop-%s\n' "$digest"
    return 0
  }

  pid_start_jiffies() {
    local pid="$1"
    local stat_line
    if [[ -r "/proc/${pid}/stat" ]]; then
      stat_line="$(cat "/proc/${pid}/stat" 2>/dev/null || true)"
      if [[ -n "$stat_line" ]]; then
        # /proc/<pid>/stat field 22 is starttime in clock ticks since boot.
        printf '%s\n' "$stat_line" | awk '{print $22}'
        return 0
      fi
    fi
    return 1
  }

  generate_lock_owner_nonce() {
    if [[ -r /proc/sys/kernel/random/uuid ]]; then
      cat /proc/sys/kernel/random/uuid 2>/dev/null && return 0
    fi
    printf '%s-%s-%s\n' "$(date -u +%s 2>/dev/null || echo 0)" "$$" "$RANDOM"
    return 0
  }

  lockdir_mtime_epoch() {
    local path="$1"
    local epoch
    if epoch="$(stat -c %Y "$path" 2>/dev/null)"; then
      printf '%s\n' "$epoch"
      return 0
    fi
    if epoch="$(stat -f %m "$path" 2>/dev/null)"; then
      printf '%s\n' "$epoch"
      return 0
    fi
    return 1
  }

  maybe_reclaim_stale_lockdir() {
    local lock_dir="$1"
    local stale_secs="$2"
    ((stale_secs > 0)) || return 1
    [[ -d "$lock_dir" ]] || return 1
    local now_epoch
    local dir_epoch
    local age_secs
    local owner_file owner_pid
    if ! now_epoch="$(date -u +%s 2>/dev/null)"; then
      return 1
    fi
    if ! dir_epoch="$(lockdir_mtime_epoch "$lock_dir")"; then
      return 1
    fi
    age_secs=$((now_epoch - dir_epoch))
    ((age_secs >= stale_secs)) || return 1
    owner_file="${lock_dir}/owner"
    owner_pid=""
    local owner_pid_start
    local live_pid_start
    owner_pid_start=""
    live_pid_start=""
    if [[ -f "$owner_file" ]]; then
      owner_pid="$(sed -nE 's/^pid=([0-9]+)$/\1/p' "$owner_file" | head -n 1)"
      owner_pid_start="$(sed -nE 's/^pid_start_jiffies=([0-9]+)$/\1/p' "$owner_file" | head -n 1)"
    fi
    if [[ -n "$owner_pid" ]] && kill -0 "$owner_pid" 2>/dev/null; then
      if [[ -n "$owner_pid_start" ]] && live_pid_start="$(pid_start_jiffies "$owner_pid")"; then
        if [[ "$live_pid_start" == "$owner_pid_start" ]]; then
          return 1
        fi
      else
        # Conservatively keep lockdir if we cannot disambiguate PID identity.
        return 1
      fi
    fi
    rm -rf "$lock_dir" 2>/dev/null || return 1
    if ((drop_events_lock_warned == 0)); then
      echo "warning: reclaimed stale drop-event lock directory $lock_dir (age=${age_secs}s)" >&2
      drop_events_lock_warned=1
    fi
    return 0
  }

  with_drop_events_lock() {
    local lock_file="$1"
    shift
    if [[ -z "$lock_file" || "$drop_events_lock_backend" == "none" ]]; then
      "$@"
      return $?
    fi
    local selected_backend="$drop_events_lock_backend"
    : >> "$lock_file"
    if [[ "$selected_backend" == "auto" ]]; then
      if command -v flock >/dev/null 2>&1; then
        selected_backend="flock"
      else
        selected_backend="mkdir"
        if ((drop_events_lock_warned == 0)); then
          echo "warning: flock not found; using mkdir lock backend for $lock_file" >&2
          drop_events_lock_warned=1
        fi
      fi
    fi
    case "$selected_backend" in
      flock)
        if ! command -v flock >/dev/null 2>&1; then
          echo "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_LOCK_BACKEND=flock requires flock in PATH" >&2
          return 1
        fi
        local lock_fd status
        exec {lock_fd}>>"$lock_file"
        flock -x "$lock_fd"
        "$@"
        status=$?
        flock -u "$lock_fd" || true
        exec {lock_fd}>&-
        return $status
        ;;
      mkdir)
        local lock_dir start_epoch now_epoch status owner_nonce owner_pid_start
        lock_dir="${lock_file}.d"
        start_epoch="$(date -u +%s)"
        while ! mkdir "$lock_dir" 2>/dev/null; do
          maybe_reclaim_stale_lockdir "$lock_dir" "$drop_events_lock_stale_secs" || true
          now_epoch="$(date -u +%s)"
          if ((now_epoch - start_epoch >= drop_events_lock_timeout_secs)); then
            echo "error: timed out acquiring drop-event lock directory $lock_dir after ${drop_events_lock_timeout_secs}s" >&2
            return 1
          fi
          sleep 0.1
        done
        owner_nonce="$(generate_lock_owner_nonce)"
        owner_pid_start=""
        if owner_pid_start="$(pid_start_jiffies "$$" 2>/dev/null)"; then
          :
        else
          owner_pid_start=""
        fi
        {
          printf 'pid=%d\n' "$$"
          if [[ -n "$owner_pid_start" ]]; then
            printf 'pid_start_jiffies=%s\n' "$owner_pid_start"
          fi
          printf 'owner_nonce=%s\n' "$owner_nonce"
          printf 'acquired_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "${lock_dir}/owner" 2>/dev/null || true
        "$@"
        status=$?
        rm -f "${lock_dir}/owner" 2>/dev/null || true
        rmdir "$lock_dir" 2>/dev/null || true
        return $status
        ;;
      *)
        echo "error: unsupported drop-event lock backend: $selected_backend" >&2
        return 1
        ;;
    esac
  }

  append_drop_event_line() {
    local line="$1"
    printf '%s\n' "$line" >> "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE"
  }

  emit_history_drop_event() {
    local history_file="$1"
    local history_format="$2"
    local lineno="$3"
    local row_generated_at="$4"
    local run_id="$5"
    local reason="$6"
    [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE" ]] || return 0
    local now_utc
    local escaped_file
    local escaped_format
    local escaped_generated
    local escaped_run_id
    local escaped_reason
    local escaped_event_id
    local escaped_schema_version
    local escaped_id_hash_mode
    local escaped_id_hash_algorithm
    local id_hash_mode_effective_event
    local id_hash_algorithm_event
    local id_hash_version_event
    local event_id
    local event_line
    now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    escaped_file="$(json_escape "$history_file")"
    escaped_format="$(json_escape "$history_format")"
    escaped_generated="$(json_escape "$row_generated_at")"
    escaped_run_id="$(json_escape "$run_id")"
    escaped_reason="$(json_escape "$reason")"
    escaped_schema_version="$(json_escape "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION")"
    id_hash_mode_effective_event="${drop_events_id_hash_mode_effective:-unavailable}"
    id_hash_algorithm_event="${drop_events_id_hash_algorithm:-unavailable}"
    id_hash_version_event="${drop_events_id_hash_version:-0}"
    escaped_id_hash_mode="$(json_escape "$id_hash_mode_effective_event")"
    escaped_id_hash_algorithm="$(json_escape "$id_hash_algorithm_event")"
    event_id="$(stable_event_id "${reason}|${history_format}|${run_id}|${row_generated_at}")"
    escaped_event_id="$(json_escape "$event_id")"
    printf -v event_line '{"schema_version":"%s","event_id":"%s","generated_at_utc":"%s","reason":"%s","history_file":"%s","history_format":"%s","line":%d,"row_generated_at_utc":"%s","run_id":"%s","id_hash_mode":"%s","id_hash_algorithm":"%s","id_hash_version":%d}' \
      "$escaped_schema_version" "$escaped_event_id" "$now_utc" "$escaped_reason" "$escaped_file" "$escaped_format" "$lineno" "$escaped_generated" "$escaped_run_id" "$escaped_id_hash_mode" "$escaped_id_hash_algorithm" "$id_hash_version_event"
    with_drop_events_lock "$drop_events_lock_file" append_drop_event_line "$event_line"
  }

  trim_history_tsv() {
    local file="$1"
    local max_entries="$2"
    local max_age_days="$3"
    local max_future_skew_secs="$4"
    local -a lines=()
    local -a cols=()
    local line
    local header
    local generated_at
    local row_epoch
    local now_epoch
    local cutoff_epoch
    local max_future_epoch
    local start i rows
    [[ -f "$file" ]] || return 0
    if ((max_age_days > 0 || max_future_skew_secs > 0)); then
      mapfile -t lines < "$file"
      ((${#lines[@]} > 0)) || return 0
      header="${lines[0]}"
      if ! now_epoch="$(date -u +%s 2>/dev/null)"; then
        echo "error: failed to compute current UTC epoch for history age retention" >&2
        exit 1
      fi
      cutoff_epoch=$((now_epoch - max_age_days * 86400))
      max_future_epoch=$((now_epoch + max_future_skew_secs))
      : > "$file"
      printf '%s\n' "$header" >> "$file"
      for ((i = 1; i < ${#lines[@]}; ++i)); do
        line="${lines[$i]}"
        [[ -z "$line" ]] && continue
        IFS=$'\t' read -r -a cols <<<"$line"
        generated_at="${cols[2]:-}"
        if ! row_epoch="$(utc_to_epoch "$generated_at")"; then
          echo "error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE $file at line $((i + 1))" >&2
          exit 1
        fi
        if ((max_future_skew_secs > 0 && row_epoch > max_future_epoch)); then
          if [[ "$YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY" == "warn" ]]; then
            echo "warning: generated_at_utc exceeds YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS in YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE $file at line $((i + 1)); dropping row due YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY=warn" >&2
            history_drop_future_tsv=$((history_drop_future_tsv + 1))
            emit_history_drop_event \
              "$file" "tsv" "$((i + 1))" "$generated_at" "${cols[1]:-}" "future_skew"
            continue
          fi
          echo "error: generated_at_utc exceeds YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS in YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE $file at line $((i + 1))" >&2
          exit 1
        fi
        if ((max_age_days == 0 || row_epoch >= cutoff_epoch)); then
          printf '%s\n' "$line" >> "$file"
        else
          history_drop_age_tsv=$((history_drop_age_tsv + 1))
          emit_history_drop_event \
            "$file" "tsv" "$((i + 1))" "$generated_at" "${cols[1]:-}" "age_retention"
        fi
      done
    fi
    ((max_entries > 0)) || return 0
    mapfile -t lines < "$file"
    ((${#lines[@]} > 0)) || return 0
    header="${lines[0]}"
    rows=$(( ${#lines[@]} - 1 ))
    if ((rows <= max_entries)); then
      return 0
    fi
    start=$(( ${#lines[@]} - max_entries ))
    for ((i = 1; i < start; ++i)); do
      line="${lines[$i]}"
      [[ -z "$line" ]] && continue
      IFS=$'\t' read -r -a cols <<<"$line"
      generated_at="${cols[2]:-}"
      history_drop_max_entries_tsv=$((history_drop_max_entries_tsv + 1))
      emit_history_drop_event \
        "$file" "tsv" "$((i + 1))" "$generated_at" "${cols[1]:-}" "max_entries"
    done
    : > "$file"
    printf '%s\n' "$header" >> "$file"
    for ((i = start; i < ${#lines[@]}; ++i)); do
      printf '%s\n' "${lines[$i]}" >> "$file"
    done
  }

  prepare_history_tsv_file() {
    local file="$1"
    local -a lines=()
    local i
    [[ -f "$file" ]] || return 0
    [[ -s "$file" ]] || return 0
    mapfile -t lines < "$file"
    ((${#lines[@]} > 0)) || return 0
    if [[ "${lines[0]}" == "$tsv_header" ]]; then
      for ((i = 1; i < ${#lines[@]}; ++i)); do
        [[ -z "${lines[$i]}" ]] && continue
        validate_history_tsv_row "${lines[$i]}" "$file" "$((i + 1))"
      done
      return 0
    fi
    if [[ "${lines[0]}" == "$legacy_tsv_header" ]]; then
      : > "$file"
      printf '%s\n' "$tsv_header" >> "$file"
      for ((i = 1; i < ${#lines[@]}; ++i)); do
        [[ -z "${lines[$i]}" ]] && continue
        validate_legacy_tsv_row "${lines[$i]}" "$file" "$((i + 1))"
        printf '0\tlegacy-%d\t1970-01-01T00:00:00Z\t%s\n' "$i" "${lines[$i]}" >> "$file"
      done
      return 0
    fi
    echo "error: unsupported YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE header in $file" >&2
    exit 1
  }

  prepare_history_jsonl_file() {
    local file="$1"
    local -a lines=()
    local i
    local line migrated
    [[ -f "$file" ]] || return 0
    [[ -s "$file" ]] || return 0
    mapfile -t lines < "$file"
    ((${#lines[@]} > 0)) || return 0
    : > "$file"
    for ((i = 0; i < ${#lines[@]}; ++i)); do
      line="${lines[$i]}"
      [[ -z "$line" ]] && continue
      migrated="$(migrate_history_jsonl_line "$line" "$file" "$((i + 1))")"
      validate_history_jsonl_line "$migrated" "$file" "$((i + 1))"
      printf '%s\n' "$migrated" >> "$file"
    done
  }

  trim_history_jsonl() {
    local file="$1"
    local max_entries="$2"
    local max_age_days="$3"
    local max_future_skew_secs="$4"
    local -a lines=()
    local line
    local generated_at
    local run_id
    local -a parsed_fields=()
    local row_epoch
    local now_epoch
    local cutoff_epoch
    local max_future_epoch
    local start i
    [[ -f "$file" ]] || return 0
    if ((max_age_days > 0 || max_future_skew_secs > 0)); then
      mapfile -t lines < "$file"
      if ! now_epoch="$(date -u +%s 2>/dev/null)"; then
        echo "error: failed to compute current UTC epoch for history age retention" >&2
        exit 1
      fi
      cutoff_epoch=$((now_epoch - max_age_days * 86400))
      max_future_epoch=$((now_epoch + max_future_skew_secs))
      : > "$file"
      for ((i = 0; i < ${#lines[@]}; ++i)); do
        line="${lines[$i]}"
        [[ -z "$line" ]] && continue
        mapfile -t parsed_fields < <(extract_history_jsonl_generated_at_and_run_id "$line" "$file" "$((i + 1))")
        generated_at="${parsed_fields[0]:-}"
        run_id="${parsed_fields[1]:-}"
        if ! row_epoch="$(utc_to_epoch "$generated_at")"; then
          echo "error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE $file at line $((i + 1))" >&2
          exit 1
        fi
        if ((max_future_skew_secs > 0 && row_epoch > max_future_epoch)); then
          if [[ "$YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY" == "warn" ]]; then
            echo "warning: generated_at_utc exceeds YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE $file at line $((i + 1)); dropping row due YOSYS_SVA_MODE_SUMMARY_HISTORY_FUTURE_POLICY=warn" >&2
            history_drop_future_jsonl=$((history_drop_future_jsonl + 1))
            emit_history_drop_event \
              "$file" "jsonl" "$((i + 1))" "$generated_at" "$run_id" "future_skew"
            continue
          fi
          echo "error: generated_at_utc exceeds YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE $file at line $((i + 1))" >&2
          exit 1
        fi
        if ((max_age_days == 0 || row_epoch >= cutoff_epoch)); then
          printf '%s\n' "$line" >> "$file"
        else
          history_drop_age_jsonl=$((history_drop_age_jsonl + 1))
          emit_history_drop_event \
            "$file" "jsonl" "$((i + 1))" "$generated_at" "$run_id" "age_retention"
        fi
      done
    fi
    ((max_entries > 0)) || return 0
    mapfile -t lines < "$file"
    if ((${#lines[@]} <= max_entries)); then
      return 0
    fi
    : > "$file"
    start=$(( ${#lines[@]} - max_entries ))
    for ((i = 0; i < start; ++i)); do
      line="${lines[$i]}"
      [[ -z "$line" ]] && continue
      mapfile -t parsed_fields < <(extract_history_jsonl_generated_at_and_run_id "$line" "$file" "$((i + 1))")
      generated_at="${parsed_fields[0]:-}"
      run_id="${parsed_fields[1]:-}"
      history_drop_max_entries_jsonl=$((history_drop_max_entries_jsonl + 1))
      emit_history_drop_event \
        "$file" "jsonl" "$((i + 1))" "$generated_at" "$run_id" "max_entries"
    done
    for ((i = start; i < ${#lines[@]}; ++i)); do
      printf '%s\n' "${lines[$i]}" >> "$file"
    done
  }

  trim_drop_events_jsonl_unlocked() {
    local file="$1"
    local max_entries="$2"
    local max_age_days="$3"
    local -a lines=()
    local line
    local generated_at
    local row_epoch
    local now_epoch
    local cutoff_epoch
    local start i
    [[ -f "$file" ]] || return 0

    prepare_drop_events_jsonl_file() {
      local migrate_file="$1"
      python3 - "$migrate_file" "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_SCHEMA_VERSION" "$drop_events_id_hash_mode" "$drop_events_id_hash_mode_effective" "$drop_events_id_hash_algorithm" "$drop_events_id_hash_version" "$drop_events_event_id_policy" "$drop_events_id_metadata_policy" "$drop_events_rewrite_run_id_regex" "$drop_events_rewrite_reason_regex" "$drop_events_rewrite_schema_version_regex" "$drop_events_rewrite_history_file_regex" "$drop_events_rewrite_schema_version_list" "$drop_events_rewrite_history_file_list" "$drop_events_rewrite_selector_mode" "$drop_events_rewrite_selector_clauses_json" "$drop_events_rewrite_selector_macros_json" "$drop_events_rewrite_selector_profiles_json" "$drop_events_rewrite_selector_profile_list" "$drop_events_rewrite_selector_profile_default_list" "$drop_events_rewrite_selector_profile_overlay_list" "$drop_events_rewrite_selector_profile_route" "$drop_events_rewrite_selector_profile_routes_json" "$drop_events_rewrite_selector_profile_route_auto_mode" "$YOSYS_SVA_DIR" "$TEST_FILTER" "$SCRIPT_DIR" "$drop_events_route_context_ci_provider" "$drop_events_route_context_ci_job" "$drop_events_route_context_ci_branch" "$drop_events_route_context_ci_target" "$drop_events_rewrite_selector_profile_route_context_json" "$drop_events_rewrite_selector_profile_route_context_schema_json" "$drop_events_rewrite_selector_profile_route_context_schema_version" "$drop_events_rewrite_row_generated_at_utc_min" "$drop_events_rewrite_row_generated_at_utc_max" <<'PY'
from datetime import datetime, timezone
import csv
import json
import re
import shutil
import subprocess
import sys
import zlib
from collections import OrderedDict

file = sys.argv[1]
default_schema = sys.argv[2]
id_hash_mode = sys.argv[3]
effective_id_hash_mode = sys.argv[4]
effective_id_hash_algorithm = sys.argv[5]
effective_id_hash_version_raw = sys.argv[6]
event_id_policy = sys.argv[7]
id_metadata_policy = sys.argv[8]
rewrite_run_id_regex = sys.argv[9]
rewrite_reason_regex = sys.argv[10]
rewrite_schema_version_regex = sys.argv[11]
rewrite_history_file_regex = sys.argv[12]
rewrite_schema_version_list_raw = sys.argv[13]
rewrite_history_file_list_raw = sys.argv[14]
rewrite_selector_mode = sys.argv[15]
rewrite_selector_clauses_json_raw = sys.argv[16]
rewrite_selector_macros_json_raw = sys.argv[17]
rewrite_selector_profiles_json_raw = sys.argv[18]
rewrite_selector_profile_list_raw = sys.argv[19]
rewrite_selector_profile_default_list_raw = sys.argv[20]
rewrite_selector_profile_overlay_list_raw = sys.argv[21]
rewrite_selector_profile_route_raw = sys.argv[22]
rewrite_selector_profile_routes_json_raw = sys.argv[23]
rewrite_selector_profile_route_auto_mode_raw = sys.argv[24]
rewrite_selector_context_suite_dir = sys.argv[25]
rewrite_selector_context_test_filter = sys.argv[26]
rewrite_selector_context_script_dir = sys.argv[27]
rewrite_selector_context_ci_provider = sys.argv[28]
rewrite_selector_context_ci_job = sys.argv[29]
rewrite_selector_context_ci_branch = sys.argv[30]
rewrite_selector_context_ci_target = sys.argv[31]
rewrite_selector_context_json_raw = sys.argv[32]
rewrite_selector_context_schema_json_raw = sys.argv[33]
rewrite_selector_context_schema_version_raw = sys.argv[34]
rewrite_row_generated_at_utc_min = sys.argv[35]
rewrite_row_generated_at_utc_max = sys.argv[36]

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

def parse_utc_epoch(value: str, field_name: str) -> int:
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        fail(
            f"error: invalid {field_name}: {value} (expected YYYY-MM-DDTHH:MM:SSZ)"
        )
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def parse_selector_list(raw: str, field_name: str):
    if not raw:
        return None
    if "\n" in raw or "\r" in raw:
        fail(
            f"error: invalid {field_name}: newline is not allowed in comma-separated list"
        )
    try:
        rows = list(csv.reader([raw], skipinitialspace=True, strict=True))
    except csv.Error as ex:
        fail(
            f"error: invalid {field_name}: malformed comma-separated list ({ex})"
        )
    if len(rows) != 1:
        fail(
            f"error: invalid {field_name}: malformed comma-separated list"
        )
    values = []
    for token in rows[0]:
        value = token.strip()
        if not value:
            fail(
                f"error: invalid {field_name}: empty entry in comma-separated list"
            )
        values.append(value)
    if not values:
        fail(
            f"error: invalid {field_name}: empty entry in comma-separated list"
        )
    return set(values)


def parse_selector_list_value(value, field_name: str):
    if value is None:
        return None
    if isinstance(value, str):
        return parse_selector_list(value, field_name)
    if isinstance(value, list):
        values = []
        for token in value:
            if not isinstance(token, str) or not token.strip():
                fail(
                    f"error: invalid {field_name}: expected non-empty string entries"
                )
            values.append(token.strip())
        if not values:
            fail(
                f"error: invalid {field_name}: expected at least one entry"
            )
        return set(values)
    fail(
        f"error: invalid {field_name}: expected string or array of strings"
    )

try:
    effective_id_hash_version = int(effective_id_hash_version_raw)
except Exception:
    effective_id_hash_version = 0
if effective_id_hash_version < 0:
    effective_id_hash_version = 0

if event_id_policy not in {"preserve", "infer", "rewrite"}:
    print(
        f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_EVENT_ID_POLICY mode: {event_id_policy}",
        file=sys.stderr,
    )
    sys.exit(1)

if id_metadata_policy not in {"preserve", "infer", "rewrite"}:
    print(
        f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_METADATA_POLICY mode: {id_metadata_policy}",
        file=sys.stderr,
    )
    sys.exit(1)

if rewrite_selector_mode not in {"all", "any"}:
    fail(
        "error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MODE: "
        f"{rewrite_selector_mode} (expected all|any)"
    )

rewrite_run_id_pattern = None
if rewrite_run_id_regex:
    try:
        rewrite_run_id_pattern = re.compile(rewrite_run_id_regex)
    except re.error as ex:
        fail(
            f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_RUN_ID_REGEX: {rewrite_run_id_regex} ({ex})"
        )

rewrite_reason_pattern = None
if rewrite_reason_regex:
    try:
        rewrite_reason_pattern = re.compile(rewrite_reason_regex)
    except re.error as ex:
        fail(
            f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_REASON_REGEX: {rewrite_reason_regex} ({ex})"
        )

rewrite_schema_version_pattern = None
if rewrite_schema_version_regex:
    try:
        rewrite_schema_version_pattern = re.compile(rewrite_schema_version_regex)
    except re.error as ex:
        fail(
            "error: invalid "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_REGEX: "
            f"{rewrite_schema_version_regex} ({ex})"
        )

rewrite_history_file_pattern = None
if rewrite_history_file_regex:
    try:
        rewrite_history_file_pattern = re.compile(rewrite_history_file_regex)
    except re.error as ex:
        fail(
            "error: invalid "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_REGEX: "
            f"{rewrite_history_file_regex} ({ex})"
        )

rewrite_schema_version_set = parse_selector_list(
    rewrite_schema_version_list_raw,
    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SCHEMA_VERSION_LIST",
)
rewrite_history_file_set = parse_selector_list(
    rewrite_history_file_list_raw,
    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_HISTORY_FILE_LIST",
)


def compile_clause_regex(value, field_name: str):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        fail(f"error: invalid {field_name}: expected non-empty string")
    try:
        return re.compile(value)
    except re.error as ex:
        fail(f"error: invalid {field_name}: {value} ({ex})")


def parse_selector_clause_array(payload, field_name: str, macro_specs=None, macro_field_name=None):
    if not isinstance(payload, list) or not payload:
        fail(
            f"error: invalid {field_name}: expected non-empty JSON array"
        )
    base_keys = {
        "run_id_regex",
        "reason_regex",
        "schema_version_regex",
        "history_file_regex",
        "schema_version_list",
        "history_file_list",
        "row_generated_at_utc_min",
        "row_generated_at_utc_max",
    }
    combinator_keys = {"all_of", "any_of", "not", "at_least", "at_most", "exactly", "macro"}
    macro_specs = macro_specs or {}
    if macro_field_name is None:
        macro_field_name = (
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON"
        )
    macro_cache = {}
    macro_state = {}
    macro_stack = []

    def parse_selector_expression(expr, expr_field_name: str):
        def resolve_macro(macro_name: str, ref_field_name: str):
            state = macro_state.get(macro_name, 0)
            if state == 2:
                return macro_cache[macro_name]
            if state == 1:
                cycle_start = 0
                for i, name in enumerate(macro_stack):
                    if name == macro_name:
                        cycle_start = i
                        break
                cycle_path = macro_stack[cycle_start:] + [macro_name]
                fail(
                    f"error: invalid {macro_field_name}: macro cycle detected ({' -> '.join(cycle_path)})"
                )
            raw_expr = macro_specs.get(macro_name)
            if raw_expr is None:
                fail(
                    f"error: invalid {ref_field_name}: unknown macro '{macro_name}'"
                )
            macro_state[macro_name] = 1
            macro_stack.append(macro_name)
            parsed_expr = parse_selector_expression(raw_expr, f"{macro_field_name}.{macro_name}")
            macro_stack.pop()
            macro_state[macro_name] = 2
            macro_cache[macro_name] = parsed_expr
            return parsed_expr

        def parse_cardinality_spec(value, spec_field_name: str):
            if not isinstance(value, dict) or not value:
                fail(
                    f"error: invalid {spec_field_name}: expected non-empty object"
                )
            unknown_keys = sorted(set(value.keys()) - {"count", "of"})
            if unknown_keys:
                fail(
                    f"error: invalid {spec_field_name}: unknown key '{unknown_keys[0]}'"
                )
            if "count" not in value or "of" not in value:
                fail(
                    f"error: invalid {spec_field_name}: expected keys 'count' and 'of'"
                )
            count = value["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                fail(
                    f"error: invalid {spec_field_name}.count: expected non-negative integer"
                )
            of_payload = value["of"]
            if not isinstance(of_payload, list) or not of_payload:
                fail(
                    f"error: invalid {spec_field_name}.of: expected non-empty JSON array"
                )
            expressions = []
            for of_index, of_expr in enumerate(of_payload, start=1):
                expressions.append(
                    parse_selector_expression(of_expr, f"{spec_field_name}.of[{of_index}]")
                )
            if count > len(expressions):
                fail(
                    f"error: invalid {spec_field_name}.count: exceeds number of expressions"
                )
            return {"count": count, "expressions": expressions}

        if not isinstance(expr, dict) or not expr:
            fail(
                f"error: invalid {expr_field_name}: expected non-empty object"
            )
        unknown_keys = sorted(set(expr.keys()) - base_keys - combinator_keys)
        if unknown_keys:
            fail(
                f"error: invalid {expr_field_name}: unknown key '{unknown_keys[0]}'"
            )

        min_epoch = None
        if "row_generated_at_utc_min" in expr:
            if not isinstance(expr["row_generated_at_utc_min"], str):
                fail(
                    f"error: invalid {expr_field_name}.row_generated_at_utc_min: expected string"
                )
            min_epoch = parse_utc_epoch(
                expr["row_generated_at_utc_min"],
                f"{expr_field_name}.row_generated_at_utc_min",
            )
        max_epoch = None
        if "row_generated_at_utc_max" in expr:
            if not isinstance(expr["row_generated_at_utc_max"], str):
                fail(
                    f"error: invalid {expr_field_name}.row_generated_at_utc_max: expected string"
                )
            max_epoch = parse_utc_epoch(
                expr["row_generated_at_utc_max"],
                f"{expr_field_name}.row_generated_at_utc_max",
            )
        if min_epoch is not None and max_epoch is not None and min_epoch > max_epoch:
            fail(
                f"error: invalid {expr_field_name}: row_generated_at_utc_min exceeds row_generated_at_utc_max"
            )

        all_of_expressions = None
        if "all_of" in expr:
            if not isinstance(expr["all_of"], list) or not expr["all_of"]:
                fail(
                    f"error: invalid {expr_field_name}.all_of: expected non-empty JSON array"
                )
            all_of_expressions = []
            for all_of_index, all_of_expr in enumerate(expr["all_of"], start=1):
                all_of_expressions.append(
                    parse_selector_expression(
                        all_of_expr, f"{expr_field_name}.all_of[{all_of_index}]"
                    )
                )

        any_of_expressions = None
        if "any_of" in expr:
            if not isinstance(expr["any_of"], list) or not expr["any_of"]:
                fail(
                    f"error: invalid {expr_field_name}.any_of: expected non-empty JSON array"
                )
            any_of_expressions = []
            for any_of_index, any_of_expr in enumerate(expr["any_of"], start=1):
                any_of_expressions.append(
                    parse_selector_expression(
                        any_of_expr, f"{expr_field_name}.any_of[{any_of_index}]"
                    )
                )

        not_expression = None
        if "not" in expr:
            if not isinstance(expr["not"], dict) or not expr["not"]:
                fail(
                    f"error: invalid {expr_field_name}.not: expected non-empty object"
                )
            not_expression = parse_selector_expression(
                expr["not"], f"{expr_field_name}.not"
            )

        at_least_spec = None
        if "at_least" in expr:
            at_least_spec = parse_cardinality_spec(
                expr["at_least"], f"{expr_field_name}.at_least"
            )

        at_most_spec = None
        if "at_most" in expr:
            at_most_spec = parse_cardinality_spec(
                expr["at_most"], f"{expr_field_name}.at_most"
            )

        exactly_spec = None
        if "exactly" in expr:
            exactly_spec = parse_cardinality_spec(
                expr["exactly"], f"{expr_field_name}.exactly"
            )

        macro_expression = None
        if "macro" in expr:
            macro_value = expr["macro"]
            if not isinstance(macro_value, str) or not macro_value.strip():
                fail(
                    f"error: invalid {expr_field_name}.macro: expected non-empty string"
                )
            macro_name = macro_value.strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", macro_name):
                fail(
                    f"error: invalid {expr_field_name}.macro: invalid macro name '{macro_name}'"
                )
            macro_expression = resolve_macro(macro_name, f"{expr_field_name}.macro")

        has_base_predicate = bool(base_keys.intersection(expr.keys()))
        if (
            not has_base_predicate
            and all_of_expressions is None
            and any_of_expressions is None
            and not_expression is None
            and at_least_spec is None
            and at_most_spec is None
            and exactly_spec is None
            and macro_expression is None
        ):
            fail(
                f"error: invalid {expr_field_name}: expected selector predicates or combinators"
            )

        return {
            "run_id_pattern": compile_clause_regex(
                expr.get("run_id_regex"), f"{expr_field_name}.run_id_regex"
            ),
            "reason_pattern": compile_clause_regex(
                expr.get("reason_regex"), f"{expr_field_name}.reason_regex"
            ),
            "schema_version_pattern": compile_clause_regex(
                expr.get("schema_version_regex"),
                f"{expr_field_name}.schema_version_regex",
            ),
            "history_file_pattern": compile_clause_regex(
                expr.get("history_file_regex"),
                f"{expr_field_name}.history_file_regex",
            ),
            "schema_version_set": parse_selector_list_value(
                expr.get("schema_version_list"),
                f"{expr_field_name}.schema_version_list",
            ),
            "history_file_set": parse_selector_list_value(
                expr.get("history_file_list"),
                f"{expr_field_name}.history_file_list",
            ),
            "row_generated_at_min_epoch": min_epoch,
            "row_generated_at_max_epoch": max_epoch,
            "all_of_expressions": all_of_expressions,
            "any_of_expressions": any_of_expressions,
            "not_expression": not_expression,
            "at_least_spec": at_least_spec,
            "at_most_spec": at_most_spec,
            "exactly_spec": exactly_spec,
            "macro_expression": macro_expression,
        }

    clauses = []
    for clause_index, clause in enumerate(payload, start=1):
        clauses.append(parse_selector_expression(clause, f"{field_name}[{clause_index}]"))
    return clauses


def parse_selector_macros(raw: str):
    if not raw:
        return {}
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON"
    )
    try:
        payload = json.loads(raw)
    except Exception:
        fail(f"error: invalid {field_name}: expected JSON object")
    if not isinstance(payload, dict) or not payload:
        fail(f"error: invalid {field_name}: expected non-empty JSON object")
    macros = {}
    for macro_name, macro_expr in payload.items():
        if not isinstance(macro_name, str) or not macro_name:
            fail(
                f"error: invalid {field_name}: macro names must be non-empty strings"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", macro_name):
            fail(
                f"error: invalid {field_name}: invalid macro name '{macro_name}'"
            )
        if not isinstance(macro_expr, dict) or not macro_expr:
            fail(
                f"error: invalid {field_name}.{macro_name}: expected non-empty object"
            )
        macros[macro_name] = macro_expr
    return macros


def parse_selector_clauses(raw: str, macro_specs):
    if not raw:
        return None
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_CLAUSES_JSON"
    )
    try:
        payload = json.loads(raw)
    except Exception:
        fail(f"error: invalid {field_name}: expected JSON array")
    return parse_selector_clause_array(
        payload,
        field_name,
        macro_specs,
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON",
    )


def parse_selector_profile_name_list(raw: str, field_name: str):
    if not raw:
        return []
    if "\n" in raw or "\r" in raw:
        fail(
            f"error: invalid {field_name}: newline is not allowed in comma-separated list"
        )
    try:
        rows = list(csv.reader([raw], skipinitialspace=True, strict=True))
    except csv.Error as ex:
        fail(
            f"error: invalid {field_name}: malformed comma-separated list ({ex})"
        )
    if len(rows) != 1:
        fail(
            f"error: invalid {field_name}: malformed comma-separated list"
        )
    names = []
    seen = set()
    for token in rows[0]:
        name = token.strip()
        if not name:
            fail(
                f"error: invalid {field_name}: empty entry in comma-separated list"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            fail(
                f"error: invalid {field_name}: invalid profile name '{name}'"
            )
        if name in seen:
            fail(
                f"error: invalid {field_name}: duplicate profile name '{name}'"
            )
        seen.add(name)
        names.append(name)
    if not names:
        fail(
            f"error: invalid {field_name}: empty entry in comma-separated list"
        )
    return names


def parse_selector_profile_name_array(payload, field_name: str):
    if not isinstance(payload, list) or not payload:
        fail(
            f"error: invalid {field_name}: expected non-empty array of profile names"
        )
    names = []
    seen = set()
    for token in payload:
        if not isinstance(token, str) or not token.strip():
            fail(
                f"error: invalid {field_name}: expected non-empty string entries"
            )
        name = token.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            fail(
                f"error: invalid {field_name}: invalid profile name '{name}'"
            )
        if name in seen:
            fail(
                f"error: invalid {field_name}: duplicate profile name '{name}'"
            )
        seen.add(name)
        names.append(name)
    return names


def format_context_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return value


def validate_context_key_name(key: str, field_name: str):
    if not isinstance(key, str) or not key:
        fail(
            f"error: invalid {field_name}: context keys must be non-empty strings"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
        fail(
            f"error: invalid {field_name}: invalid context key '{key}'"
        )


def validate_context_value(value, key_spec, field_name: str, format_output: bool):
    expected_type = key_spec["type"]
    if expected_type == "string":
        if not isinstance(value, str):
            fail(
                f"error: invalid {field_name}: expected string value"
            )
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            fail(
                f"error: invalid {field_name}: expected integer value"
            )
    else:
        if not isinstance(value, bool):
            fail(
                f"error: invalid {field_name}: expected boolean value"
            )
    if key_spec["min_value"] is not None and value < key_spec["min_value"]:
        fail(
            f"error: invalid {field_name}: value below minimum {key_spec['min_value']}"
        )
    if key_spec["max_value"] is not None and value > key_spec["max_value"]:
        fail(
            f"error: invalid {field_name}: value above maximum {key_spec['max_value']}"
        )
    formatted_value = format_context_scalar(value)
    if key_spec["enum_values"] is not None and formatted_value not in key_spec["enum_values"]:
        fail(
            f"error: invalid {field_name}: value is not in configured enum"
        )
    if key_spec["regex"] is not None and not key_spec["regex"].search(formatted_value):
        fail(
            f"error: invalid {field_name}: value does not match configured regex"
        )
    if format_output:
        return formatted_value
    return value


def parse_selector_profile_route_context_schema(raw: str, expected_version_raw: str):
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_JSON"
    )
    expected_version_field = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_VERSION"
    )
    if not raw:
        return None
    expected_version = expected_version_raw.strip()
    if not expected_version:
        fail(
            f"error: invalid {expected_version_field}: expected non-empty schema version"
        )
    try:
        payload = json.loads(raw)
    except Exception:
        fail(f"error: invalid {field_name}: expected JSON object")
    if not isinstance(payload, dict) or not payload:
        fail(f"error: invalid {field_name}: expected non-empty JSON object")
    unknown_keys = sorted(
        set(payload.keys())
        - {"schema_version", "allow_unknown_keys", "validate_merged_context", "keys"}
    )
    if unknown_keys:
        fail(
            f"error: invalid {field_name}: unknown key '{unknown_keys[0]}'"
        )
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        fail(
            f"error: invalid {field_name}.schema_version: expected non-empty string"
        )
    if schema_version != expected_version:
        fail(
            f"error: invalid {field_name}.schema_version: expected '{expected_version}', got '{schema_version}'"
        )
    allow_unknown_keys = payload.get("allow_unknown_keys", True)
    if not isinstance(allow_unknown_keys, bool):
        fail(
            f"error: invalid {field_name}.allow_unknown_keys: expected boolean"
        )
    validate_merged_context = payload.get("validate_merged_context", False)
    if not isinstance(validate_merged_context, bool):
        fail(
            f"error: invalid {field_name}.validate_merged_context: expected boolean"
        )
    keys_payload = payload.get("keys")
    if not isinstance(keys_payload, dict) or not keys_payload:
        fail(
            f"error: invalid {field_name}.keys: expected non-empty object"
        )
    key_specs = {}
    for context_key, key_spec in keys_payload.items():
        validate_context_key_name(context_key, f"{field_name}.keys")
        key_field = f"{field_name}.keys.{context_key}"
        if not isinstance(key_spec, dict):
            fail(
                f"error: invalid {key_field}: expected object"
            )
        unknown_spec_keys = sorted(
            set(key_spec.keys()) - {"type", "required", "regex", "enum", "min", "max"}
        )
        if unknown_spec_keys:
            fail(
                f"error: invalid {key_field}: unknown key '{unknown_spec_keys[0]}'"
            )
        value_type = key_spec.get("type", "string")
        if value_type not in {"string", "integer", "boolean"}:
            fail(
                f"error: invalid {key_field}.type: expected one of string, integer, boolean"
            )
        required = key_spec.get("required", False)
        if not isinstance(required, bool):
            fail(
                f"error: invalid {key_field}.required: expected boolean"
            )
        compiled_regex = None
        if "regex" in key_spec:
            compiled_regex = compile_clause_regex(
                key_spec["regex"], f"{key_field}.regex"
            )
        enum_values = None
        if "enum" in key_spec:
            enum_raw = key_spec["enum"]
            if not isinstance(enum_raw, list) or not enum_raw:
                fail(
                    f"error: invalid {key_field}.enum: expected non-empty array"
                )
            enum_values = set()
            for enum_value in enum_raw:
                enum_values.add(
                    validate_context_value(
                        enum_value,
                        {
                            "type": value_type,
                            "min_value": None,
                            "max_value": None,
                            "enum_values": None,
                            "regex": None,
                        },
                        f"{key_field}.enum",
                        True,
                    )
                )
        min_value = None
        max_value = None
        if "min" in key_spec:
            if value_type != "integer":
                fail(
                    f"error: invalid {key_field}.min: only supported for integer type"
                )
            min_raw = key_spec["min"]
            if not isinstance(min_raw, int) or isinstance(min_raw, bool):
                fail(
                    f"error: invalid {key_field}.min: expected integer"
                )
            min_value = min_raw
        if "max" in key_spec:
            if value_type != "integer":
                fail(
                    f"error: invalid {key_field}.max: only supported for integer type"
                )
            max_raw = key_spec["max"]
            if not isinstance(max_raw, int) or isinstance(max_raw, bool):
                fail(
                    f"error: invalid {key_field}.max: expected integer"
                )
            max_value = max_raw
        if min_value is not None and max_value is not None and min_value > max_value:
            fail(
                f"error: invalid {key_field}: min must be <= max"
            )
        key_specs[context_key] = {
            "type": value_type,
            "required": required,
            "regex": compiled_regex,
            "enum_values": enum_values,
            "min_value": min_value,
            "max_value": max_value,
        }
    return {
        "allow_unknown_keys": allow_unknown_keys,
        "validate_merged_context": validate_merged_context,
        "key_specs": key_specs,
    }


def validate_selector_profile_route_context_map(
    payload,
    field_name: str,
    schema_spec,
    enforce_required: bool,
    enforce_unknown: bool,
    format_output: bool,
):
    if not isinstance(payload, dict):
        fail(f"error: invalid {field_name}: expected JSON object")
    context = {}
    key_specs = {}
    allow_unknown_keys = True
    if schema_spec is not None:
        key_specs = schema_spec["key_specs"]
        allow_unknown_keys = schema_spec["allow_unknown_keys"]
    for key, value in payload.items():
        validate_context_key_name(key, field_name)
        key_spec = key_specs.get(key)
        if key_spec is None:
            if schema_spec is not None and enforce_unknown and not allow_unknown_keys:
                fail(
                    f"error: invalid {field_name}.{key}: unknown context key"
                )
            if isinstance(value, str):
                context[key] = value
                continue
            if isinstance(value, bool) or isinstance(value, int):
                if format_output:
                    context[key] = format_context_scalar(value)
                else:
                    context[key] = value
                continue
            fail(
                f"error: invalid {field_name}.{key}: expected scalar value"
            )
        context[key] = validate_context_value(
            value, key_spec, f"{field_name}.{key}", format_output
        )
    if schema_spec is not None and enforce_required:
        for key, key_spec in key_specs.items():
            if key_spec["required"] and key not in context:
                fail(
                    f"error: invalid {field_name}: missing required context key '{key}'"
                )
    return context


def parse_selector_profile_route_context(raw: str, schema_spec):
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON"
    )
    if not raw:
        payload = {}
    else:
        try:
            payload = json.loads(raw)
        except Exception:
            fail(f"error: invalid {field_name}: expected JSON object")
    enforce_required = True
    if schema_spec is not None and schema_spec["validate_merged_context"]:
        # Built-in context fields are only available after merge, so required
        # checks must be deferred to merged-context validation.
        enforce_required = False
    format_output = True
    if schema_spec is not None and schema_spec["validate_merged_context"]:
        # Preserve typed scalars until merged-context validation applies schema.
        format_output = False
    return validate_selector_profile_route_context_map(
        payload,
        field_name,
        schema_spec,
        enforce_required,
        True,
        format_output,
    )


def parse_selector_profile_routes(raw: str):
    if not raw:
        return None
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON"
    )
    try:
        payload = json.loads(raw)
    except Exception:
        fail(f"error: invalid {field_name}: expected JSON object")
    if not isinstance(payload, dict) or not payload:
        fail(f"error: invalid {field_name}: expected non-empty JSON object")

    routes = {}
    for route_name, route_spec in payload.items():
        if not isinstance(route_name, str) or not route_name:
            fail(
                f"error: invalid {field_name}: route names must be non-empty strings"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", route_name):
            fail(
                f"error: invalid {field_name}: invalid route name '{route_name}'"
            )
        route_field = f"{field_name}.{route_name}"
        if not isinstance(route_spec, dict) or not route_spec:
            fail(
                f"error: invalid {route_field}: expected non-empty object"
            )
        unknown_keys = sorted(
            set(route_spec.keys())
            - {"default_list", "profile_list", "overlay_list", "when", "priority"}
        )
        if unknown_keys:
            fail(
                f"error: invalid {route_field}: unknown key '{unknown_keys[0]}'"
            )

        def parse_route_name_list(route_value, route_field_name: str):
            if isinstance(route_value, str):
                return parse_selector_profile_name_list(route_value, route_field_name)
            return parse_selector_profile_name_array(route_value, route_field_name)

        default_names = []
        if "default_list" in route_spec:
            default_names = parse_route_name_list(
                route_spec["default_list"], f"{route_field}.default_list"
            )
        profile_names = []
        if "profile_list" in route_spec:
            profile_names = parse_route_name_list(
                route_spec["profile_list"], f"{route_field}.profile_list"
            )
        overlay_names = []
        if "overlay_list" in route_spec:
            overlay_names = parse_route_name_list(
                route_spec["overlay_list"], f"{route_field}.overlay_list"
            )
        route_priority = 0
        if "priority" in route_spec:
            priority_value = route_spec["priority"]
            if (
                not isinstance(priority_value, int)
                or isinstance(priority_value, bool)
                or priority_value < 0
            ):
                fail(
                    f"error: invalid {route_field}.priority: expected non-negative integer"
                )
            route_priority = priority_value
        when_conditions = []
        if "when" in route_spec:
            when_value = route_spec["when"]
            if not isinstance(when_value, dict) or not when_value:
                fail(
                    f"error: invalid {route_field}.when: expected non-empty object"
                )
            unknown_when_keys = sorted(
                set(when_value.keys())
                - {
                    "suite_dir_regex",
                    "test_filter_regex",
                    "script_dir_regex",
                    "ci_provider_regex",
                    "ci_job_regex",
                    "ci_branch_regex",
                    "ci_target_regex",
                    "context_regex",
                }
            )
            if unknown_when_keys:
                fail(
                    f"error: invalid {route_field}.when: unknown key '{unknown_when_keys[0]}'"
                )
            if "suite_dir_regex" in when_value:
                when_conditions.append(
                    (
                        "suite_dir",
                        compile_clause_regex(
                            when_value["suite_dir_regex"],
                            f"{route_field}.when.suite_dir_regex",
                        ),
                    )
                )
            if "test_filter_regex" in when_value:
                when_conditions.append(
                    (
                        "test_filter",
                        compile_clause_regex(
                            when_value["test_filter_regex"],
                            f"{route_field}.when.test_filter_regex",
                        ),
                    )
                )
            if "script_dir_regex" in when_value:
                when_conditions.append(
                    (
                        "script_dir",
                        compile_clause_regex(
                            when_value["script_dir_regex"],
                            f"{route_field}.when.script_dir_regex",
                        ),
                    )
                )
            if "ci_provider_regex" in when_value:
                when_conditions.append(
                    (
                        "ci_provider",
                        compile_clause_regex(
                            when_value["ci_provider_regex"],
                            f"{route_field}.when.ci_provider_regex",
                        ),
                    )
                )
            if "ci_job_regex" in when_value:
                when_conditions.append(
                    (
                        "ci_job",
                        compile_clause_regex(
                            when_value["ci_job_regex"],
                            f"{route_field}.when.ci_job_regex",
                        ),
                    )
                )
            if "ci_branch_regex" in when_value:
                when_conditions.append(
                    (
                        "ci_branch",
                        compile_clause_regex(
                            when_value["ci_branch_regex"],
                            f"{route_field}.when.ci_branch_regex",
                        ),
                    )
                )
            if "ci_target_regex" in when_value:
                when_conditions.append(
                    (
                        "ci_target",
                        compile_clause_regex(
                            when_value["ci_target_regex"],
                            f"{route_field}.when.ci_target_regex",
                        ),
                    )
                )
            if "context_regex" in when_value:
                context_regex_value = when_value["context_regex"]
                if not isinstance(context_regex_value, dict) or not context_regex_value:
                    fail(
                        f"error: invalid {route_field}.when.context_regex: expected non-empty object"
                    )
                for context_key, context_pattern in context_regex_value.items():
                    if not isinstance(context_key, str) or not context_key:
                        fail(
                            f"error: invalid {route_field}.when.context_regex: keys must be non-empty strings"
                        )
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", context_key):
                        fail(
                            f"error: invalid {route_field}.when.context_regex: invalid context key '{context_key}'"
                        )
                    when_conditions.append(
                        (
                            context_key,
                            compile_clause_regex(
                                context_pattern,
                                f"{route_field}.when.context_regex.{context_key}",
                            ),
                        )
                    )
        if not default_names and not profile_names and not overlay_names:
            fail(
                f"error: invalid {route_field}: expected at least one list key"
            )
        routes[route_name] = {
            "default_names": default_names,
            "profile_names": profile_names,
            "overlay_names": overlay_names,
            "priority": route_priority,
            "when_conditions": when_conditions,
        }
    return routes


def parse_selector_profiles(raw: str, macro_specs):
    if not raw:
        return None
    field_name = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILES_JSON"
    )
    try:
        payload = json.loads(raw)
    except Exception:
        fail(f"error: invalid {field_name}: expected JSON object")
    if not isinstance(payload, dict) or not payload:
        fail(f"error: invalid {field_name}: expected non-empty JSON object")

    profile_specs = {}
    for profile_name, profile_value in payload.items():
        if not isinstance(profile_name, str) or not profile_name:
            fail(
                f"error: invalid {field_name}: profile names must be non-empty strings"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_name):
            fail(
                f"error: invalid {field_name}: invalid profile name '{profile_name}'"
            )
        profile_field = f"{field_name}.{profile_name}"
        if isinstance(profile_value, list):
            profile_specs[profile_name] = {
                "extends": [],
                "clauses": parse_selector_clause_array(
                    profile_value,
                    profile_field,
                    macro_specs,
                    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON",
                ),
            }
            continue
        if not isinstance(profile_value, dict) or not profile_value:
            fail(
                f"error: invalid {profile_field}: expected non-empty array or object"
            )
        unknown_keys = sorted(set(profile_value.keys()) - {"extends", "clauses"})
        if unknown_keys:
            fail(
                f"error: invalid {profile_field}: unknown key '{unknown_keys[0]}'"
            )

        extends = []
        if "extends" in profile_value:
            extends_value = profile_value["extends"]
            if isinstance(extends_value, str):
                extends_tokens = [extends_value]
            elif isinstance(extends_value, list):
                extends_tokens = extends_value
            else:
                fail(
                    f"error: invalid {profile_field}.extends: expected string or array of strings"
                )
            if not extends_tokens:
                fail(
                    f"error: invalid {profile_field}.extends: expected at least one profile name"
                )
            seen_extends = set()
            for token in extends_tokens:
                if not isinstance(token, str) or not token.strip():
                    fail(
                        f"error: invalid {profile_field}.extends: expected non-empty string entries"
                    )
                name = token.strip()
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                    fail(
                        f"error: invalid {profile_field}.extends: invalid profile name '{name}'"
                    )
                if name in seen_extends:
                    fail(
                        f"error: invalid {profile_field}.extends: duplicate profile name '{name}'"
                    )
                seen_extends.add(name)
                extends.append(name)

        clauses = []
        if "clauses" in profile_value:
            clauses = parse_selector_clause_array(
                profile_value["clauses"],
                f"{profile_field}.clauses",
                macro_specs,
                "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_MACROS_JSON",
            )
        if not extends and not clauses:
            fail(
                f"error: invalid {profile_field}: expected 'clauses' and/or 'extends'"
            )
        profile_specs[profile_name] = {"extends": extends, "clauses": clauses}

    resolved = {}
    resolve_stack = []
    resolve_state = {}

    def resolve_profile(profile_name: str):
        state = resolve_state.get(profile_name, 0)
        if state == 2:
            return resolved[profile_name]
        if state == 1:
            cycle_start = 0
            for i, name in enumerate(resolve_stack):
                if name == profile_name:
                    cycle_start = i
                    break
            cycle_path = resolve_stack[cycle_start:] + [profile_name]
            fail(
                f"error: invalid {field_name}: profile cycle detected ({' -> '.join(cycle_path)})"
            )
        spec = profile_specs.get(profile_name)
        if spec is None:
            parent = resolve_stack[-1] if resolve_stack else profile_name
            fail(
                f"error: invalid {field_name}.{parent}.extends: unknown profile '{profile_name}'"
            )
        resolve_state[profile_name] = 1
        resolve_stack.append(profile_name)
        clauses = []
        for base_profile_name in spec["extends"]:
            clauses.extend(resolve_profile(base_profile_name))
        clauses.extend(spec["clauses"])
        resolve_stack.pop()
        resolve_state[profile_name] = 2
        resolved[profile_name] = clauses
        return clauses

    for profile_name in profile_specs:
        resolve_profile(profile_name)

    return resolved


rewrite_selector_macros = parse_selector_macros(rewrite_selector_macros_json_raw)
rewrite_selector_clauses = parse_selector_clauses(
    rewrite_selector_clauses_json_raw, rewrite_selector_macros
)
rewrite_selector_profile_names = parse_selector_profile_name_list(
    rewrite_selector_profile_list_raw,
    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_LIST",
)
rewrite_selector_profile_default_names = parse_selector_profile_name_list(
    rewrite_selector_profile_default_list_raw,
    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_DEFAULT_LIST",
)
rewrite_selector_profile_overlay_names = parse_selector_profile_name_list(
    rewrite_selector_profile_overlay_list_raw,
    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_OVERLAY_LIST",
)
rewrite_selector_profile_route_context_schema = parse_selector_profile_route_context_schema(
    rewrite_selector_context_schema_json_raw,
    rewrite_selector_context_schema_version_raw,
)
rewrite_selector_profile_route_context = parse_selector_profile_route_context(
    rewrite_selector_context_json_raw,
    rewrite_selector_profile_route_context_schema,
)
rewrite_selector_profile_routes = parse_selector_profile_routes(
    rewrite_selector_profile_routes_json_raw
)
rewrite_selector_profile_route = rewrite_selector_profile_route_raw.strip()
rewrite_selector_profile_route_auto_mode = (
    rewrite_selector_profile_route_auto_mode_raw.strip().lower()
)
if rewrite_selector_profile_route_auto_mode == "auto":
    rewrite_selector_profile_route_auto_mode = "optional"
if not rewrite_selector_profile_route_auto_mode:
    rewrite_selector_profile_route_auto_mode = "off"
if rewrite_selector_profile_route_auto_mode not in {"off", "optional", "required"}:
    fail(
        "error: invalid "
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE: "
        f"expected one of off, optional, required; got '{rewrite_selector_profile_route_auto_mode_raw}'"
    )
if rewrite_selector_profile_route:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rewrite_selector_profile_route):
        fail(
            "error: invalid "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE: "
            f"invalid route name '{rewrite_selector_profile_route}'"
        )
    if rewrite_selector_profile_routes is None:
        fail(
            "error: "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE "
            "requires "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON"
        )
selected_route_spec = None
if rewrite_selector_profile_route:
    selected_route_spec = rewrite_selector_profile_routes.get(
        rewrite_selector_profile_route
    )
    if selected_route_spec is None:
        fail(
            "error: invalid "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE: "
            f"unknown route '{rewrite_selector_profile_route}'"
        )
elif rewrite_selector_profile_route_auto_mode != "off":
    if rewrite_selector_profile_routes is None:
        if rewrite_selector_profile_route_auto_mode == "required":
            fail(
                "error: "
                "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE=required "
                "requires "
                "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON"
            )
    else:
        auto_context = {
            "suite_dir": rewrite_selector_context_suite_dir,
            "test_filter": rewrite_selector_context_test_filter,
            "script_dir": rewrite_selector_context_script_dir,
            "ci_provider": rewrite_selector_context_ci_provider,
            "ci_job": rewrite_selector_context_ci_job,
            "ci_branch": rewrite_selector_context_ci_branch,
            "ci_target": rewrite_selector_context_ci_target,
        }
        for context_key, context_value in rewrite_selector_profile_route_context.items():
            auto_context[context_key] = context_value
        if (
            rewrite_selector_profile_route_context_schema is not None
            and rewrite_selector_profile_route_context_schema["validate_merged_context"]
        ):
            auto_context = validate_selector_profile_route_context_map(
                auto_context,
                (
                    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON.effective"
                ),
                rewrite_selector_profile_route_context_schema,
                True,
                True,
                True,
            )
        auto_matches = []
        for route_name, route_spec in rewrite_selector_profile_routes.items():
            when_conditions = route_spec["when_conditions"]
            if not when_conditions:
                continue
            matched = True
            for context_field, compiled_regex in when_conditions:
                if not compiled_regex.search(auto_context.get(context_field, "")):
                    matched = False
                    break
            if matched:
                auto_matches.append((route_name, route_spec["priority"]))
        if auto_matches:
            max_priority = max(priority for _, priority in auto_matches)
            top_matches = sorted(
                route_name
                for route_name, priority in auto_matches
                if priority == max_priority
            )
            if len(top_matches) == 1:
                rewrite_selector_profile_route = top_matches[0]
                selected_route_spec = rewrite_selector_profile_routes[
                    rewrite_selector_profile_route
                ]
            else:
                fail(
                    "error: invalid "
                    "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE: "
                    f"ambiguous auto route matches ({', '.join(top_matches)})"
                )
        elif rewrite_selector_profile_route_auto_mode == "required":
            fail(
                "error: invalid "
                "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_AUTO_MODE: "
                "no route matched auto context"
            )
rewrite_selector_profiles = parse_selector_profiles(
    rewrite_selector_profiles_json_raw, rewrite_selector_macros
)
rewrite_selector_profile_requests = []
if selected_route_spec is not None:
    route_base_field = (
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTES_JSON"
        f".{rewrite_selector_profile_route}"
    )
    for profile_name in selected_route_spec["default_names"]:
        rewrite_selector_profile_requests.append(
            (profile_name, f"{route_base_field}.default_list")
        )
    for profile_name in selected_route_spec["profile_names"]:
        rewrite_selector_profile_requests.append(
            (profile_name, f"{route_base_field}.profile_list")
        )
    for profile_name in selected_route_spec["overlay_names"]:
        rewrite_selector_profile_requests.append(
            (profile_name, f"{route_base_field}.overlay_list")
        )
for profile_name in rewrite_selector_profile_default_names:
    rewrite_selector_profile_requests.append(
        (
            profile_name,
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_DEFAULT_LIST",
        )
    )
for profile_name in rewrite_selector_profile_names:
    rewrite_selector_profile_requests.append(
        (
            profile_name,
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_LIST",
        )
    )
for profile_name in rewrite_selector_profile_overlay_names:
    rewrite_selector_profile_requests.append(
        (
            profile_name,
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_OVERLAY_LIST",
        )
    )

if rewrite_selector_profile_requests:
    if rewrite_selector_profiles is None:
        fail(
            "error: "
            "selector profile lists require "
            "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILES_JSON"
        )
    selected_profile_clauses = []
    for profile_name, source_field_name in rewrite_selector_profile_requests:
        clauses = rewrite_selector_profiles.get(profile_name)
        if clauses is None:
            fail(
                f"error: invalid {source_field_name}: "
                f"unknown profile '{profile_name}'"
            )
        selected_profile_clauses.extend(clauses)
    if rewrite_selector_clauses is None:
        rewrite_selector_clauses = selected_profile_clauses
    else:
        rewrite_selector_clauses = rewrite_selector_clauses + selected_profile_clauses

rewrite_row_generated_at_min_epoch = None
if rewrite_row_generated_at_utc_min:
    rewrite_row_generated_at_min_epoch = parse_utc_epoch(
        rewrite_row_generated_at_utc_min,
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MIN",
    )

rewrite_row_generated_at_max_epoch = None
if rewrite_row_generated_at_utc_max:
    rewrite_row_generated_at_max_epoch = parse_utc_epoch(
        rewrite_row_generated_at_utc_max,
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MAX",
    )

if (
    rewrite_row_generated_at_min_epoch is not None
    and rewrite_row_generated_at_max_epoch is not None
    and rewrite_row_generated_at_min_epoch > rewrite_row_generated_at_max_epoch
):
    fail(
        "error: invalid rewrite timestamp window: "
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MIN exceeds "
        "YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_ROW_GENERATED_AT_UTC_MAX"
    )

REQUIRED_STRING_KEYS = (
    "generated_at_utc",
    "reason",
    "history_file",
    "history_format",
    "row_generated_at_utc",
    "run_id",
)
CANONICAL_KEYS = (
    "schema_version",
    "event_id",
    "generated_at_utc",
    "reason",
    "history_file",
    "history_format",
    "line",
    "row_generated_at_utc",
    "run_id",
    "id_hash_mode",
    "id_hash_algorithm",
    "id_hash_version",
)


def parse_json_object(line: str, lineno: int) -> dict:
    def no_duplicate_object_pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key '{key}'")
            result[key] = value
        return result

    try:
        obj = json.loads(line, object_pairs_hook=no_duplicate_object_pairs_hook)
    except ValueError as ex:
        if "duplicate key" in str(ex):
            fail(
                f"error: duplicate key in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}: {ex}"
            )
        fail(
            f"error: invalid JSON object in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
        )
    except Exception:
        fail(
            f"error: invalid JSON object in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
        )
    if not isinstance(obj, dict):
        fail(
            f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE line in {file} at line {lineno}: expected JSON object"
        )
    return obj


def normalize_schema(schema, lineno: int) -> str:
    if schema is None:
        schema = default_schema
    if isinstance(schema, bool):
        fail(
            f"error: invalid key 'schema_version' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
        )
    if isinstance(schema, int):
        if schema < 0:
            fail(
                f"error: invalid key 'schema_version' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
        return str(schema)
    if isinstance(schema, str) and schema.isdigit():
        return schema
    fail(
        f"error: invalid key 'schema_version' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
    )


def cksum_digest(key: str):
    if shutil.which("cksum") is None:
        return None
    proc = subprocess.run(["cksum"], input=key, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().split()
    if not parts:
        return None
    return parts[0]


def derive_event_id(reason: str, history_format: str, run_id: str, row_generated_at: str) -> str:
    key = f"{reason}|{history_format}|{run_id}|{row_generated_at}"
    if id_hash_mode not in {"auto", "cksum", "crc32"}:
        fail(
            f"error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH mode: {id_hash_mode}"
        )

    if id_hash_mode in {"auto", "cksum"}:
        digest = cksum_digest(key)
        if digest is not None:
            return f"drop-{digest}"
        if id_hash_mode == "cksum":
            fail(
                "error: YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_ID_HASH=cksum requires functional cksum in PATH"
            )

    digest = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
    return f"drop-{digest}"


def parse_row_generated_at_epoch(row_generated_at: str, lineno: int) -> int:
    return parse_utc_epoch(
        row_generated_at,
        f"row_generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}",
    )


def selected_for_rewrite(
    reason: str,
    run_id: str,
    schema_version: str,
    history_file: str,
    row_generated_at_epoch,
) -> bool:
    def selector_expression_matches(expr) -> bool:
        checks = []
        if expr["run_id_pattern"] is not None:
            checks.append(expr["run_id_pattern"].search(run_id) is not None)
        if expr["reason_pattern"] is not None:
            checks.append(expr["reason_pattern"].search(reason) is not None)
        if expr["schema_version_pattern"] is not None:
            checks.append(expr["schema_version_pattern"].search(schema_version) is not None)
        if expr["history_file_pattern"] is not None:
            checks.append(expr["history_file_pattern"].search(history_file) is not None)
        if expr["schema_version_set"] is not None:
            checks.append(schema_version in expr["schema_version_set"])
        if expr["history_file_set"] is not None:
            checks.append(history_file in expr["history_file_set"])
        if expr["row_generated_at_min_epoch"] is not None:
            checks.append(
                row_generated_at_epoch is not None
                and row_generated_at_epoch >= expr["row_generated_at_min_epoch"]
            )
        if expr["row_generated_at_max_epoch"] is not None:
            checks.append(
                row_generated_at_epoch is not None
                and row_generated_at_epoch <= expr["row_generated_at_max_epoch"]
            )
        if expr["all_of_expressions"] is not None:
            checks.append(all(selector_expression_matches(child) for child in expr["all_of_expressions"]))
        if expr["any_of_expressions"] is not None:
            checks.append(any(selector_expression_matches(child) for child in expr["any_of_expressions"]))
        if expr["not_expression"] is not None:
            checks.append(not selector_expression_matches(expr["not_expression"]))
        if expr["at_least_spec"] is not None:
            match_count = sum(
                1
                for child in expr["at_least_spec"]["expressions"]
                if selector_expression_matches(child)
            )
            checks.append(match_count >= expr["at_least_spec"]["count"])
        if expr["at_most_spec"] is not None:
            match_count = sum(
                1
                for child in expr["at_most_spec"]["expressions"]
                if selector_expression_matches(child)
            )
            checks.append(match_count <= expr["at_most_spec"]["count"])
        if expr["exactly_spec"] is not None:
            match_count = sum(
                1
                for child in expr["exactly_spec"]["expressions"]
                if selector_expression_matches(child)
            )
            checks.append(match_count == expr["exactly_spec"]["count"])
        if expr["macro_expression"] is not None:
            checks.append(selector_expression_matches(expr["macro_expression"]))
        return bool(checks) and all(checks)

    if rewrite_selector_clauses is not None:
        return any(selector_expression_matches(clause) for clause in rewrite_selector_clauses)

    checks = []
    if rewrite_run_id_pattern is not None:
        checks.append(rewrite_run_id_pattern.search(run_id) is not None)
    if rewrite_reason_pattern is not None:
        checks.append(rewrite_reason_pattern.search(reason) is not None)
    if rewrite_schema_version_pattern is not None:
        checks.append(rewrite_schema_version_pattern.search(schema_version) is not None)
    if rewrite_history_file_pattern is not None:
        checks.append(rewrite_history_file_pattern.search(history_file) is not None)
    if rewrite_schema_version_set is not None:
        checks.append(schema_version in rewrite_schema_version_set)
    if rewrite_history_file_set is not None:
        checks.append(history_file in rewrite_history_file_set)
    if rewrite_row_generated_at_min_epoch is not None:
        checks.append(row_generated_at_epoch >= rewrite_row_generated_at_min_epoch)
    if rewrite_row_generated_at_max_epoch is not None:
        checks.append(row_generated_at_epoch <= rewrite_row_generated_at_max_epoch)
    if not checks:
        return True
    if rewrite_selector_mode == "any":
        return any(checks)
    return all(checks)


with open(file, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

def selector_expression_uses_row_generated_at(expr) -> bool:
    if (
        expr["row_generated_at_min_epoch"] is not None
        or expr["row_generated_at_max_epoch"] is not None
    ):
        return True
    if expr["all_of_expressions"] is not None:
        for child in expr["all_of_expressions"]:
            if selector_expression_uses_row_generated_at(child):
                return True
    if expr["any_of_expressions"] is not None:
        for child in expr["any_of_expressions"]:
            if selector_expression_uses_row_generated_at(child):
                return True
    if expr["not_expression"] is not None:
        if selector_expression_uses_row_generated_at(expr["not_expression"]):
            return True
    if expr["at_least_spec"] is not None:
        for child in expr["at_least_spec"]["expressions"]:
            if selector_expression_uses_row_generated_at(child):
                return True
    if expr["at_most_spec"] is not None:
        for child in expr["at_most_spec"]["expressions"]:
            if selector_expression_uses_row_generated_at(child):
                return True
    if expr["exactly_spec"] is not None:
        for child in expr["exactly_spec"]["expressions"]:
            if selector_expression_uses_row_generated_at(child):
                return True
    if expr["macro_expression"] is not None:
        if selector_expression_uses_row_generated_at(expr["macro_expression"]):
            return True
    return False


needs_row_generated_at_epoch = (
    rewrite_row_generated_at_min_epoch is not None
    or rewrite_row_generated_at_max_epoch is not None
)
if rewrite_selector_clauses is not None:
    for clause in rewrite_selector_clauses:
        if selector_expression_uses_row_generated_at(clause):
            needs_row_generated_at_epoch = True
            break

rewritten = []
for lineno, line in enumerate(lines, start=1):
    if not line:
        continue
    obj = parse_json_object(line, lineno)
    for key in REQUIRED_STRING_KEYS:
        value = obj.get(key)
        if not isinstance(value, str) or not value:
            fail(
                f"error: missing key '{key}' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
    line_value = obj.get("line")
    if isinstance(line_value, bool) or not isinstance(line_value, int) or line_value < 0:
        fail(
            f"error: missing key 'line' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
        )

    schema_version = normalize_schema(obj.get("schema_version"), lineno)
    row_generated_at_epoch = None
    if needs_row_generated_at_epoch:
        row_generated_at_epoch = parse_row_generated_at_epoch(
            obj["row_generated_at_utc"], lineno
        )
    rewrite_selected = selected_for_rewrite(
        obj["reason"],
        obj["run_id"],
        schema_version,
        obj["history_file"],
        row_generated_at_epoch,
    )
    event_id = obj.get("event_id")
    had_event_id = isinstance(event_id, str) and bool(event_id)
    if event_id_policy == "preserve":
        if not had_event_id:
            fail(
                f"error: missing key 'event_id' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
    if event_id_policy == "rewrite" and rewrite_selected:
        event_id = derive_event_id(
            obj["reason"], obj["history_format"], obj["run_id"], obj["row_generated_at_utc"]
        )
    elif not had_event_id:
        event_id = derive_event_id(
            obj["reason"], obj["history_format"], obj["run_id"], obj["row_generated_at_utc"]
        )

    id_hash_mode_value = obj.get("id_hash_mode")
    if id_hash_mode_value is not None:
        if (
            not isinstance(id_hash_mode_value, str)
            or id_hash_mode_value not in ("auto", "cksum", "crc32", "unavailable")
        ):
            fail(
                f"error: invalid key 'id_hash_mode' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
    elif (not had_event_id) or id_metadata_policy in {"infer", "rewrite"}:
        id_hash_mode_value = effective_id_hash_mode

    id_hash_algorithm_value = obj.get("id_hash_algorithm")
    if id_hash_algorithm_value is not None:
        if (
            not isinstance(id_hash_algorithm_value, str)
            or id_hash_algorithm_value not in ("cksum", "crc32", "unavailable")
        ):
            fail(
                f"error: invalid key 'id_hash_algorithm' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
    elif (not had_event_id) or id_metadata_policy in {"infer", "rewrite"}:
        id_hash_algorithm_value = effective_id_hash_algorithm

    id_hash_version_value = obj.get("id_hash_version")
    if id_hash_version_value is not None:
        if (
            isinstance(id_hash_version_value, bool)
            or not isinstance(id_hash_version_value, int)
            or id_hash_version_value < 0
        ):
            fail(
                f"error: invalid key 'id_hash_version' in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE {file} at line {lineno}"
            )
    elif (not had_event_id) or id_metadata_policy in {"infer", "rewrite"}:
        id_hash_version_value = effective_id_hash_version

    if had_event_id and id_metadata_policy == "rewrite" and rewrite_selected:
        id_hash_mode_value = effective_id_hash_mode
        id_hash_algorithm_value = effective_id_hash_algorithm
        id_hash_version_value = effective_id_hash_version

    canonical = OrderedDict()
    canonical["schema_version"] = schema_version
    canonical["event_id"] = event_id
    canonical["generated_at_utc"] = obj["generated_at_utc"]
    canonical["reason"] = obj["reason"]
    canonical["history_file"] = obj["history_file"]
    canonical["history_format"] = obj["history_format"]
    canonical["line"] = line_value
    canonical["row_generated_at_utc"] = obj["row_generated_at_utc"]
    canonical["run_id"] = obj["run_id"]
    if id_hash_mode_value is not None:
        canonical["id_hash_mode"] = id_hash_mode_value
    if id_hash_algorithm_value is not None:
        canonical["id_hash_algorithm"] = id_hash_algorithm_value
    if id_hash_version_value is not None:
        canonical["id_hash_version"] = id_hash_version_value
    for key, value in obj.items():
        if key not in canonical:
            canonical[key] = value
    rewritten.append(json.dumps(canonical, separators=(",", ":")))

with open(file, "w", encoding="utf-8") as f:
    for line in rewritten:
        f.write(line + "\n")
PY
    }

    prepare_drop_events_jsonl_file "$file"

    if ((max_age_days > 0)); then
      mapfile -t lines < "$file"
      if ! now_epoch="$(date -u +%s 2>/dev/null)"; then
        echo "error: failed to compute current UTC epoch for drop-event age retention" >&2
        exit 1
      fi
      cutoff_epoch=$((now_epoch - max_age_days * 86400))
      : > "$file"
      for ((i = 0; i < ${#lines[@]}; ++i)); do
        line="${lines[$i]}"
        [[ -z "$line" ]] && continue
        generated_at="$(extract_drop_event_generated_at_utc "$line" "$file" "$((i + 1))")"
        if ! row_epoch="$(utc_to_epoch "$generated_at")"; then
          echo "error: invalid generated_at_utc in YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE $file at line $((i + 1))" >&2
          exit 1
        fi
        if ((row_epoch >= cutoff_epoch)); then
          printf '%s\n' "$line" >> "$file"
        fi
      done
    fi
    ((max_entries > 0)) || return 0
    mapfile -t lines < "$file"
    if ((${#lines[@]} <= max_entries)); then
      return 0
    fi
    : > "$file"
    start=$(( ${#lines[@]} - max_entries ))
    for ((i = start; i < ${#lines[@]}; ++i)); do
      printf '%s\n' "${lines[$i]}" >> "$file"
    done
  }

  trim_drop_events_jsonl() {
    local file="$1"
    local max_entries="$2"
    local max_age_days="$3"
    with_drop_events_lock \
      "$drop_events_lock_file" \
      trim_drop_events_jsonl_unlocked \
      "$file" \
      "$max_entries" \
      "$max_age_days"
  }

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_TSV_FILE" ]]; then
    : > "$YOSYS_SVA_MODE_SUMMARY_TSV_FILE"
    printf '%s\n' "$tsv_header" >> "$YOSYS_SVA_MODE_SUMMARY_TSV_FILE"
    printf '%s\n' "$tsv_row" >> "$YOSYS_SVA_MODE_SUMMARY_TSV_FILE"
  fi

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" ]]; then
    prepare_history_tsv_file "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE"
    if [[ ! -s "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" ]]; then
      printf '%s\n' "$tsv_header" >> "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE"
    fi
    printf '%s\n' "$tsv_row" >> "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE"
    trim_history_tsv \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS"
  fi

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" ]]; then
    prepare_history_jsonl_file "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE"
    # First trim existing rows to accumulate drop counters before appending this run.
    trim_history_jsonl \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_AGE_DAYS" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_FUTURE_SKEW_SECS"
    {
      printf '{"schema_version":"%s","run_id":"%s","generated_at_utc":"%s","test_summary":{"total":%d,"failures":%d,"xfail":%d,"xpass":%d,"skipped":%d},"mode_summary":{"total":%d,"pass":%d,"fail":%d,"xfail":%d,"xpass":%d,"epass":%d,"efail":%d,"unskip":%d,"skipped":%d,"skip_pass":%d,"skip_fail":%d,"skip_expected":%d,"skip_unexpected":%d},"skip_reasons":{"vhdl":%d,"fail_no_macro":%d,"no_property":%d,"other":%d},"drop_events_summary":{"total":%d,"reasons":{"future_skew":%d,"age_retention":%d,"max_entries":%d},"history_format":{"tsv":%d,"jsonl":%d},"id_hash_mode":"%s","id_hash_algorithm":"%s","id_hash_version":%d}}\n' \
        "$YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION" "$run_id" "$generated_at" \
        "$total" "$failures" "$xfails" "$xpasses" "$skipped" \
        "$mode_total" "$mode_out_pass" "$mode_out_fail" "$mode_out_xfail" \
        "$mode_out_xpass" "$mode_out_epass" "$mode_out_efail" "$mode_out_unskip" \
        "$mode_skipped" "$mode_skipped_pass" "$mode_skipped_fail" \
        "$mode_skipped_expected" "$mode_skipped_unexpected" \
        "$mode_skip_reason_vhdl" "$mode_skip_reason_fail_no_macro" \
        "$mode_skip_reason_no_property" "$mode_skip_reason_other" \
        "$((history_drop_future_tsv + history_drop_future_jsonl + history_drop_age_tsv + history_drop_age_jsonl + history_drop_max_entries_tsv + history_drop_max_entries_jsonl))" \
        "$((history_drop_future_tsv + history_drop_future_jsonl))" \
        "$((history_drop_age_tsv + history_drop_age_jsonl))" \
        "$((history_drop_max_entries_tsv + history_drop_max_entries_jsonl))" \
        "$((history_drop_future_tsv + history_drop_age_tsv + history_drop_max_entries_tsv))" \
        "$((history_drop_future_jsonl + history_drop_age_jsonl + history_drop_max_entries_jsonl))" \
        "$drop_events_id_hash_mode_effective" \
        "$drop_events_id_hash_algorithm" \
        "$drop_events_id_hash_version"
    } >> "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE"
    # Re-apply entry cap after appending this run row.
    trim_history_jsonl \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES" \
      0 \
      0
  fi

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_JSON_FILE" ]]; then
    : > "$YOSYS_SVA_MODE_SUMMARY_JSON_FILE"
    {
      printf '{\n'
      printf '  "schema_version": "%s",\n' "$YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION"
      printf '  "run_id": "%s",\n' "$run_id"
      printf '  "generated_at_utc": "%s",\n' "$generated_at"
      printf '  "test_summary": {\n'
      printf '    "total": %d,\n' "$total"
      printf '    "failures": %d,\n' "$failures"
      printf '    "xfail": %d,\n' "$xfails"
      printf '    "xpass": %d,\n' "$xpasses"
      printf '    "skipped": %d\n' "$skipped"
      printf '  },\n'
      printf '  "mode_summary": {\n'
      printf '    "total": %d,\n' "$mode_total"
      printf '    "pass": %d,\n' "$mode_out_pass"
      printf '    "fail": %d,\n' "$mode_out_fail"
      printf '    "xfail": %d,\n' "$mode_out_xfail"
      printf '    "xpass": %d,\n' "$mode_out_xpass"
      printf '    "epass": %d,\n' "$mode_out_epass"
      printf '    "efail": %d,\n' "$mode_out_efail"
      printf '    "unskip": %d,\n' "$mode_out_unskip"
      printf '    "skipped": %d,\n' "$mode_skipped"
      printf '    "skip_pass": %d,\n' "$mode_skipped_pass"
      printf '    "skip_fail": %d,\n' "$mode_skipped_fail"
      printf '    "skip_expected": %d,\n' "$mode_skipped_expected"
      printf '    "skip_unexpected": %d\n' "$mode_skipped_unexpected"
      printf '  },\n'
      printf '  "skip_reasons": {\n'
      printf '    "vhdl": %d,\n' "$mode_skip_reason_vhdl"
      printf '    "fail_no_macro": %d,\n' "$mode_skip_reason_fail_no_macro"
      printf '    "no_property": %d,\n' "$mode_skip_reason_no_property"
      printf '    "other": %d\n' "$mode_skip_reason_other"
      printf '  },\n'
      printf '  "drop_events_summary": {\n'
      printf '    "total": %d,\n' "$((history_drop_future_tsv + history_drop_future_jsonl + history_drop_age_tsv + history_drop_age_jsonl + history_drop_max_entries_tsv + history_drop_max_entries_jsonl))"
      printf '    "reasons": {\n'
      printf '      "future_skew": %d\n' "$((history_drop_future_tsv + history_drop_future_jsonl))"
      printf '      ,"age_retention": %d\n' "$((history_drop_age_tsv + history_drop_age_jsonl))"
      printf '      ,"max_entries": %d\n' "$((history_drop_max_entries_tsv + history_drop_max_entries_jsonl))"
      printf '    },\n'
      printf '    "history_format": {\n'
      printf '      "tsv": %d,\n' "$((history_drop_future_tsv + history_drop_age_tsv + history_drop_max_entries_tsv))"
      printf '      "jsonl": %d\n' "$((history_drop_future_jsonl + history_drop_age_jsonl + history_drop_max_entries_jsonl))"
      printf '    },\n'
      printf '    "id_hash_mode": "%s",\n' "$drop_events_id_hash_mode_effective"
      printf '    "id_hash_algorithm": "%s",\n' "$drop_events_id_hash_algorithm"
      printf '    "id_hash_version": %d\n' "$drop_events_id_hash_version"
      printf '  }\n'
      printf '}\n'
    } > "$YOSYS_SVA_MODE_SUMMARY_JSON_FILE"
  fi

  if ((history_drop_future_tsv > 0 || history_drop_future_jsonl > 0 || \
        history_drop_age_tsv > 0 || history_drop_age_jsonl > 0 || \
        history_drop_max_entries_tsv > 0 || history_drop_max_entries_jsonl > 0)); then
    echo "warning: dropped history rows (future_skew: tsv=${history_drop_future_tsv}, jsonl=${history_drop_future_jsonl}; age_retention: tsv=${history_drop_age_tsv}, jsonl=${history_drop_age_jsonl}; max_entries: tsv=${history_drop_max_entries_tsv}, jsonl=${history_drop_max_entries_jsonl})" >&2
  fi

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE" ]]; then
    trim_drop_events_jsonl \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_ENTRIES" \
      "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_MAX_AGE_DAYS"
  fi
}

report_case_outcome() {
  local base="$1"
  local mode="$2"
  local passed="$3"
  local profile="$4"
  mode_total=$((mode_total + 1))
  record_observed_case "$base" "$mode" "$profile" "$passed"
  local expected
  expected="$(lookup_expected_case "$base" "$mode" "$profile")"
  case "$expected" in
    skip)
      echo "UNSKIP($mode): $base [$profile]"
      mode_out_unskip=$((mode_out_unskip + 1))
      failures=$((failures + 1))
      ;;
    xfail)
      if [[ "$passed" == "1" ]]; then
        echo "XPASS($mode): $base [$profile]"
        mode_out_xpass=$((mode_out_xpass + 1))
        xpasses=$((xpasses + 1))
        if [[ "$ALLOW_XPASS" != "1" ]]; then
          failures=$((failures + 1))
        fi
      else
        echo "XFAIL($mode): $base [$profile]"
        mode_out_xfail=$((mode_out_xfail + 1))
        xfails=$((xfails + 1))
      fi
      ;;
    fail)
      if [[ "$passed" == "1" ]]; then
        echo "EPASS($mode): $base [$profile]"
        mode_out_epass=$((mode_out_epass + 1))
        failures=$((failures + 1))
      else
        echo "EFAIL($mode): $base [$profile]"
        mode_out_efail=$((mode_out_efail + 1))
      fi
      ;;
    pass)
      if [[ "$passed" == "1" ]]; then
        echo "PASS($mode): $base"
        mode_out_pass=$((mode_out_pass + 1))
      else
        echo "FAIL($mode): $base"
        mode_out_fail=$((mode_out_fail + 1))
        failures=$((failures + 1))
      fi
      ;;
  esac
}

report_skipped_case() {
  local base="$1"
  local mode="$2"
  local profile="$3"
  local reason="$4"
  local emit_line="${5:-1}"
  mode_total=$((mode_total + 1))
  mode_skipped=$((mode_skipped + 1))
  case "$mode" in
    pass) mode_skipped_pass=$((mode_skipped_pass + 1)) ;;
    fail) mode_skipped_fail=$((mode_skipped_fail + 1)) ;;
  esac
  case "$reason" in
    vhdl) mode_skip_reason_vhdl=$((mode_skip_reason_vhdl + 1)) ;;
    fail-no-macro) mode_skip_reason_fail_no_macro=$((mode_skip_reason_fail_no_macro + 1)) ;;
    no-property) mode_skip_reason_no_property=$((mode_skip_reason_no_property + 1)) ;;
    *) mode_skip_reason_other=$((mode_skip_reason_other + 1)) ;;
  esac
  record_skipped_case "$base" "$mode" "$profile"
  local expected
  expected="$(lookup_expected_case "$base" "$mode" "$profile")"
  if [[ "$emit_line" == "1" ]]; then
    echo "SKIP($reason): $base"
  fi
  if [[ "$expected" == "skip" ]]; then
    mode_skipped_expected=$((mode_skipped_expected + 1))
    if [[ "$emit_line" == "1" ]]; then
      echo "SKIP_EXPECTED($mode): $base [$profile]"
    fi
    return
  fi
  mode_skipped_unexpected=$((mode_skipped_unexpected + 1))
  if [[ "$EXPECT_SKIP_STRICT" == "1" ]]; then
    echo "UNEXPECTED_SKIP($mode): $base [$profile]"
    failures=$((failures + 1))
  fi
}

run_case() {
  local sv="$1"
  local mode="$2"
  local base
  base="$(basename "$sv" .sv)"
  if [[ "$mode" == "fail" && "$SKIP_FAIL_WITHOUT_MACRO" == "1" ]]; then
    if ! grep -qE '^\s*`(ifn?def|if)\s+FAIL\b' "$sv"; then
      report_skipped_case "$base" "$mode" "$(case_profile)" "fail-no-macro"
      return
    fi
  fi
  local extra_def=()
  if [[ "$mode" == "fail" ]]; then
    extra_def=(-DFAIL)
  fi
  local log_tag="$base"
  local rel_path="${sv#"$YOSYS_SVA_DIR/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"
  local mlir="$tmpdir/${base}_${mode}.mlir"
  local bmc_log="$tmpdir/${base}_${mode}.circt-bmc.log"

  local verilog_args=()
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
    verilog_args+=("--no-uvm-auto-include")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
    verilog_args+=("${extra_args[@]}")
  fi
  if ! run_limited "$CIRCT_VERILOG" --ir-llhd "${verilog_args[@]}" \
      "${extra_def[@]}" "$sv" > "$mlir"; then
    report_case_outcome "$base" "$mode" 0 "$(case_profile)"
    return
  fi
  local out
  bmc_args=("-b" "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" \
      "--module" "$TOP" "--shared-libs=$Z3_LIB")
  if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
    bmc_args+=("--rising-clocks-only")
  fi
  if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
    bmc_args+=("--allow-multi-clock")
  fi
  if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    bmc_args+=("--assume-known-inputs")
  fi
  if [[ -n "$CIRCT_BMC_ARGS" ]]; then
    read -r -a extra_bmc_args <<<"$CIRCT_BMC_ARGS"
    bmc_args+=("${extra_bmc_args[@]}")
  fi
  out=""
  if out="$(run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" \
      2> "$bmc_log")"; then
    bmc_status=0
  else
    bmc_status=$?
  fi
  if [[ "$NO_PROPERTY_AS_SKIP" == "1" ]] && \
      grep -q "no property provided to check in module" "$bmc_log"; then
    report_skipped_case "$base" "$mode" "$(case_profile)" "no-property"
    skipped=$((skipped + 1))
    return
  fi

  if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
    if [[ "$bmc_status" -eq 0 ]]; then
      report_case_outcome "$base" "$mode" 1 "$(case_profile)"
    else
      report_case_outcome "$base" "$mode" 0 "$(case_profile)"
    fi
  else
    if [[ "$mode" == "pass" ]]; then
      if ! grep -q "Bound reached with no violations!" <<<"$out"; then
        report_case_outcome "$base" "$mode" 0 "$(case_profile)"
      else
        report_case_outcome "$base" "$mode" 1 "$(case_profile)"
      fi
    else
      if ! grep -q "Assertion can be violated!" <<<"$out"; then
        report_case_outcome "$base" "$mode" 0 "$(case_profile)"
      else
        report_case_outcome "$base" "$mode" 1 "$(case_profile)"
      fi
    fi
  fi

  if [[ -n "$KEEP_LOGS_DIR" ]]; then
    mkdir -p "$KEEP_LOGS_DIR"
    cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}_${mode}.mlir" 2>/dev/null || true
  fi
}

for sv in "$YOSYS_SVA_DIR"/*.sv; do
  if [[ ! -f "$sv" ]]; then
    continue
  fi
  if [[ -n "$TEST_FILTER" ]]; then
    base="$(basename "$sv" .sv)"
    if [[ ! "$base" =~ $TEST_FILTER ]]; then
      continue
    fi
  fi
  base="$(basename "$sv" .sv)"
  if [[ "$SKIP_VHDL" == "1" && -f "$YOSYS_SVA_DIR/$base.vhd" ]]; then
    profile="$(case_profile)"
    report_skipped_case "$base" pass "$profile" "vhdl" 1
    report_skipped_case "$base" fail "$profile" "vhdl" 0
    skipped=$((skipped + 1))
    continue
  fi
  total=$((total + 1))
  run_case "$sv" pass
  run_case "$sv" fail
done

if [[ -n "$EXPECT_DIFF_BASELINE" ]]; then
  emit_expectation_diff "$EXPECT_DIFF_BASELINE" "$EXPECT_DIFF_BASELINE_DEFAULT_EXPECTED"
  if [[ "$EXPECT_DIFF_FAIL_ON_CHANGE" == "1" ]] && \
      ((expect_diff_added > 0 || expect_diff_removed > 0 || expect_diff_changed > 0)); then
    failures=$((failures + 1))
  fi
fi

if [[ -n "$EXPECT_OBSERVED_FILE" || -n "$EXPECT_REGEN_FILE" ]]; then
  emit_observed_outputs
fi

if [[ "$EXPECT_LINT" == "1" ]] && [[ "$EXPECT_LINT_FAIL_ON_ISSUES" == "1" ]] && ((lint_issues > 0)); then
  failures=$((failures + 1))
fi

if [[ -n "$YOSYS_SVA_MODE_SUMMARY_TSV_FILE" || -n "$YOSYS_SVA_MODE_SUMMARY_JSON_FILE" || \
      -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" || -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" || \
      -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_JSONL_FILE" ]]; then
  emit_mode_summary_outputs
fi

echo "yosys SVA summary: $total tests, failures=$failures, xfail=$xfails, xpass=$xpasses, skipped=$skipped"
echo "yosys SVA mode summary: total=$mode_total pass=$mode_out_pass fail=$mode_out_fail xfail=$mode_out_xfail xpass=$mode_out_xpass epass=$mode_out_epass efail=$mode_out_efail unskip=$mode_out_unskip skipped=$mode_skipped skip_pass=$mode_skipped_pass skip_fail=$mode_skipped_fail skip_expected=$mode_skipped_expected skip_unexpected=$mode_skipped_unexpected skip_reason_vhdl=$mode_skip_reason_vhdl skip_reason_fail-no-macro=$mode_skip_reason_fail_no_macro skip_reason_no-property=$mode_skip_reason_no_property skip_reason_other=$mode_skip_reason_other"
exit "$failures"
