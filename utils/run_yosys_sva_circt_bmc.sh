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

declare -A expected_cases
declare -A observed_cases
load_expected_cases() {
  local map_name="$1"
  local file="$2"
  local default_expected="${3:-}"
  local -n out_map="$map_name"
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
      pass|fail|xfail) ;;
      *)
        echo "warning: invalid expected outcome '$expected' in $file for $test|$mode|$profile" >&2
        continue
        ;;
    esac
    out_map["$test|$mode|$profile"]="$expected"
  done < "$file"
  return 0
}

# Load legacy xfail rows first, then let EXPECT_FILE override where needed.
if [[ -n "$XFAIL_FILE" ]]; then
  load_expected_cases expected_cases "$XFAIL_FILE" "xfail"
fi
load_expected_cases expected_cases "$EXPECT_FILE"

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
        local test mode profile
        IFS='|' read -r test mode profile <<<"$key"
        printf '%s\t%s\t%s\t%s\n' \
          "$test" "$mode" "$profile" "${observed_cases["$key"]}" >> "$EXPECT_OBSERVED_FILE"
      done < <(printf '%s\n' "${keys[@]}" | sort)
    fi
  fi
  if [[ -n "$EXPECT_REGEN_FILE" ]]; then
    : > "$EXPECT_REGEN_FILE"
    printf '# test_name\tmode\tprofile\texpected\n' >> "$EXPECT_REGEN_FILE"
    printf '# generated from observed outcomes (pass->pass, fail->xfail)\n' >> "$EXPECT_REGEN_FILE"
    if ((${#keys[@]} > 0)); then
      while IFS= read -r key; do
        local test mode profile expected
        IFS='|' read -r test mode profile <<<"$key"
        expected="xfail"
        if [[ "${observed_cases["$key"]}" == "pass" ]]; then
          expected="pass"
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

run_case() {
  local sv="$1"
  local mode="$2"
  if [[ "$mode" == "fail" && "$SKIP_FAIL_WITHOUT_MACRO" == "1" ]]; then
    if ! grep -qE '^\s*`(ifn?def|if)\s+FAIL\b' "$sv"; then
      echo "SKIP(fail-no-macro): $(basename "$sv" .sv)"
      return
    fi
  fi
  local extra_def=()
  if [[ "$mode" == "fail" ]]; then
    extra_def=(-DFAIL)
  fi
  local base
  base="$(basename "$sv" .sv)"
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
    echo "SKIP(no-property): $base"
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
    echo "SKIP(vhdl): $base"
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

echo "yosys SVA summary: $total tests, failures=$failures, xfail=$xfails, xpass=$xpasses, skipped=$skipped"
exit "$failures"
