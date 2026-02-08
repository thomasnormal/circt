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
expect_diff_added=0
expect_diff_removed=0
expect_diff_changed=0
lint_issues=0

declare -A expected_cases
declare -A observed_cases
declare -A regen_override_cases
declare -A suite_tests

case "$EXPECT_LINT_APPLY_MODE" in
  off|dry-run|apply) ;;
  *)
    echo "invalid EXPECT_LINT_APPLY_MODE: $EXPECT_LINT_APPLY_MODE (expected off|dry-run|apply)" >&2
    exit 1
    ;;
esac
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
    echo "EXPECT_LINT_APPLY: mode=$mode files=0 changed=0 actions=$EXPECT_LINT_APPLY_ACTIONS"
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
  echo "EXPECT_LINT_APPLY: mode=$mode files=${#file_list[@]} changed=$changed_files actions=$EXPECT_LINT_APPLY_ACTIONS"
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

report_case_outcome() {
  local base="$1"
  local mode="$2"
  local passed="$3"
  local profile="$4"
  record_observed_case "$base" "$mode" "$profile" "$passed"
  local expected
  expected="$(lookup_expected_case "$base" "$mode" "$profile")"
  case "$expected" in
    skip)
      echo "UNSKIP($mode): $base [$profile]"
      failures=$((failures + 1))
      ;;
    xfail)
      if [[ "$passed" == "1" ]]; then
        echo "XPASS($mode): $base [$profile]"
        xpasses=$((xpasses + 1))
        if [[ "$ALLOW_XPASS" != "1" ]]; then
          failures=$((failures + 1))
        fi
      else
        echo "XFAIL($mode): $base [$profile]"
        xfails=$((xfails + 1))
      fi
      ;;
    fail)
      if [[ "$passed" == "1" ]]; then
        echo "EPASS($mode): $base [$profile]"
        failures=$((failures + 1))
      else
        echo "EFAIL($mode): $base [$profile]"
      fi
      ;;
    pass)
      if [[ "$passed" == "1" ]]; then
        echo "PASS($mode): $base"
      else
        echo "FAIL($mode): $base"
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
  record_skipped_case "$base" "$mode" "$profile"
  local expected
  expected="$(lookup_expected_case "$base" "$mode" "$profile")"
  if [[ "$emit_line" == "1" ]]; then
    echo "SKIP($reason): $base"
  fi
  if [[ "$expected" == "skip" ]]; then
    if [[ "$emit_line" == "1" ]]; then
      echo "SKIP_EXPECTED($mode): $base [$profile]"
    fi
    return
  fi
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

echo "yosys SVA summary: $total tests, failures=$failures, xfail=$xfails, xpass=$xpasses, skipped=$skipped"
exit "$failures"
