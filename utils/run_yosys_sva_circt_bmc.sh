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
  local tsv_row
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

  validate_history_jsonl_line() {
    local line="$1"
    local file="$2"
    local lineno="$3"
    local key
    if [[ "$line" != \{* || "$line" != *\} ]]; then
      echo "error: invalid YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line in $file at line $lineno: expected JSON object" >&2
      exit 1
    fi
    if ! printf '%s\n' "$line" | grep -Eq '"schema_version"[[:space:]]*:[[:space:]]*"[0-9]+"'; then
      echo "error: invalid JSONL schema_version in $file at line $lineno" >&2
      exit 1
    fi
    for key in run_id generated_at_utc test_summary mode_summary skip_reasons; do
      if ! printf '%s\n' "$line" | grep -Eq "\"$key\"[[:space:]]*:"; then
        echo "error: missing key '$key' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE $file at line $lineno" >&2
        exit 1
      fi
    done
    for key in total failures xfail xpass skipped pass fail epass efail unskip skip_pass skip_fail skip_expected skip_unexpected vhdl fail_no_macro no_property other; do
      if ! printf '%s\n' "$line" | grep -Eq "\"$key\"[[:space:]]*:"; then
        echo "error: missing summary field '$key' in YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE $file at line $lineno" >&2
        exit 1
      fi
    done
  }

  trim_history_tsv() {
    local file="$1"
    local max_entries="$2"
    local -a lines=()
    local header
    local start i rows
    ((max_entries > 0)) || return 0
    [[ -f "$file" ]] || return 0
    mapfile -t lines < "$file"
    ((${#lines[@]} > 0)) || return 0
    header="${lines[0]}"
    rows=$(( ${#lines[@]} - 1 ))
    if ((rows <= max_entries)); then
      return 0
    fi
    : > "$file"
    printf '%s\n' "$header" >> "$file"
    start=$(( ${#lines[@]} - max_entries ))
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
    local line payload migrated
    [[ -f "$file" ]] || return 0
    [[ -s "$file" ]] || return 0
    mapfile -t lines < "$file"
    ((${#lines[@]} > 0)) || return 0
    : > "$file"
    for ((i = 0; i < ${#lines[@]}; ++i)); do
      line="${lines[$i]}"
      [[ -z "$line" ]] && continue
      if [[ "$line" == *\"schema_version\"* ]]; then
        validate_history_jsonl_line "$line" "$file" "$((i + 1))"
        printf '%s\n' "$line" >> "$file"
        continue
      fi
      if [[ "$line" != *"{"* || "$line" != *"}"* ]]; then
        echo "error: unsupported YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line format in $file at line $((i + 1))" >&2
        exit 1
      fi
      payload="$(printf '%s' "$line" | sed -E 's/^[[:space:]]*\{//; s/\}[[:space:]]*$//')"
      if [[ -n "$payload" ]]; then
        printf -v migrated '{"schema_version":"0","run_id":"legacy-%d","generated_at_utc":"1970-01-01T00:00:00Z",%s}' "$((i + 1))" "$payload"
      else
        printf -v migrated '{"schema_version":"0","run_id":"legacy-%d","generated_at_utc":"1970-01-01T00:00:00Z"}' "$((i + 1))"
      fi
      validate_history_jsonl_line "$migrated" "$file" "$((i + 1))"
      printf '%s\n' "$migrated" >> "$file"
    done
  }

  trim_history_jsonl() {
    local file="$1"
    local max_entries="$2"
    local -a lines=()
    local start i
    ((max_entries > 0)) || return 0
    [[ -f "$file" ]] || return 0
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
    trim_history_tsv "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES"
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
      printf '  }\n'
      printf '}\n'
    } > "$YOSYS_SVA_MODE_SUMMARY_JSON_FILE"
  fi

  if [[ -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" ]]; then
    prepare_history_jsonl_file "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE"
    {
      printf '{"schema_version":"%s","run_id":"%s","generated_at_utc":"%s","test_summary":{"total":%d,"failures":%d,"xfail":%d,"xpass":%d,"skipped":%d},"mode_summary":{"total":%d,"pass":%d,"fail":%d,"xfail":%d,"xpass":%d,"epass":%d,"efail":%d,"unskip":%d,"skipped":%d,"skip_pass":%d,"skip_fail":%d,"skip_expected":%d,"skip_unexpected":%d},"skip_reasons":{"vhdl":%d,"fail_no_macro":%d,"no_property":%d,"other":%d}}\n' \
        "$YOSYS_SVA_MODE_SUMMARY_SCHEMA_VERSION" "$run_id" "$generated_at" \
        "$total" "$failures" "$xfails" "$xpasses" "$skipped" \
        "$mode_total" "$mode_out_pass" "$mode_out_fail" "$mode_out_xfail" \
        "$mode_out_xpass" "$mode_out_epass" "$mode_out_efail" "$mode_out_unskip" \
        "$mode_skipped" "$mode_skipped_pass" "$mode_skipped_fail" \
        "$mode_skipped_expected" "$mode_skipped_unexpected" \
        "$mode_skip_reason_vhdl" "$mode_skip_reason_fail_no_macro" \
        "$mode_skip_reason_no_property" "$mode_skip_reason_other"
    } >> "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE"
    trim_history_jsonl "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" "$YOSYS_SVA_MODE_SUMMARY_HISTORY_MAX_ENTRIES"
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
      -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE" || -n "$YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE" ]]; then
  emit_mode_summary_outputs
fi

echo "yosys SVA summary: $total tests, failures=$failures, xfail=$xfails, xpass=$xpasses, skipped=$skipped"
echo "yosys SVA mode summary: total=$mode_total pass=$mode_out_pass fail=$mode_out_fail xfail=$mode_out_xfail xpass=$mode_out_xpass epass=$mode_out_epass efail=$mode_out_efail unskip=$mode_out_unskip skipped=$mode_skipped skip_pass=$mode_skipped_pass skip_fail=$mode_skipped_fail skip_expected=$mode_skipped_expected skip_unexpected=$mode_skipped_unexpected skip_reason_vhdl=$mode_skip_reason_vhdl skip_reason_fail-no-macro=$mode_skip_reason_fail_no_macro skip_reason_no-property=$mode_skip_reason_no_property skip_reason_other=$mode_skip_reason_other"
exit "$failures"
