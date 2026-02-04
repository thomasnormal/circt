#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"
FORCE_BMC="${FORCE_BMC:-0}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
BMC_RUN_SMTLIB="${BMC_RUN_SMTLIB:-0}"
Z3_BIN="${Z3_BIN:-}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
# NOTE: NO_PROPERTY_AS_SKIP defaults to 0 because the "no property provided to check"
# warning is SPURIOUS - it's emitted before LTLToCore and LowerClockedAssertLike passes
# run, which convert verif.clocked_assert (!ltl.property type) to verif.assert (i1 type).
# After these passes complete, the actual assertions are present and checked correctly.
# Setting this to 1 would cause false SKIP results (e.g., 9/26 instead of 23/26 pass rate).
NO_PROPERTY_AS_SKIP="${NO_PROPERTY_AS_SKIP:-0}"
TAG_REGEX="${TAG_REGEX:-(^| )16\\.|(^| )9\\.4\\.4}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/sv-tests-bmc-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECT_FILE="${EXPECT_FILE:-$SCRIPT_DIR/sv-tests-bmc-expect.txt}"
UVM_PATH="${UVM_PATH:-$SCRIPT_DIR/../lib/Runtime/uvm}"
UVM_TAG_REGEX="${UVM_TAG_REGEX:-(^| )uvm( |$)}"
INCLUDE_UVM_TAGS="${INCLUDE_UVM_TAGS:-0}"
TAG_REGEX_EFFECTIVE="$TAG_REGEX"
if [[ "$INCLUDE_UVM_TAGS" == "1" ]]; then
  TAG_REGEX_EFFECTIVE="($TAG_REGEX_EFFECTIVE)|$UVM_TAG_REGEX"
fi

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
fi

if [[ "$BMC_RUN_SMTLIB" == "1" && "$BMC_SMOKE_ONLY" != "1" ]]; then
  if [[ -z "$Z3_BIN" ]]; then
    if command -v z3 >/dev/null 2>&1; then
      Z3_BIN="z3"
    elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
    elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3/build/z3"
    fi
  fi
  if [[ -z "$Z3_BIN" ]]; then
    echo "z3 not found; set Z3_BIN or disable BMC_RUN_SMTLIB" >&2
    exit 1
  fi
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

results_tmp="$tmpdir/results.txt"
touch "$results_tmp"

pass=0
fail=0
xfail=0
xpass=0
error=0
skip=0
total=0

declare -A expect_mode
if [[ -f "$EXPECT_FILE" ]]; then
  while IFS=$'\t' read -r name mode reason; do
    if [[ -z "$name" || "$name" =~ ^# ]]; then
      continue
    fi
    if [[ -z "$mode" ]]; then
      mode="compile-only"
    fi
    expect_mode["$name"]="$mode"
  done < "$EXPECT_FILE"
fi

read_meta() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*:${key}:[[:space:]]*//p" "$file" | head -n 1
}

normalize_paths() {
  local root="$1"
  shift
  local out=()
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      continue
    fi
    if [[ "$item" = /* ]]; then
      out+=("$item")
    else
      out+=("$root/$item")
    fi
  done
  printf '%s\n' "${out[@]}"
}

while IFS= read -r -d '' sv; do
  tags="$(read_meta tags "$sv")"
  if [[ -z "$tags" ]]; then
    skip=$((skip + 1))
    continue
  fi
  if ! [[ "$tags" =~ $TAG_REGEX_EFFECTIVE ]]; then
    skip=$((skip + 1))
    continue
  fi

  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  total=$((total + 1))

  type="$(read_meta type "$sv")"
  run_bmc=1
  if [[ "$type" =~ [Pp]arsing ]]; then
    run_bmc=0
    if [[ "$FORCE_BMC" == "1" ]]; then
      run_bmc=1
    fi
  fi
  use_uvm=0
  if [[ "$tags" =~ $UVM_TAG_REGEX ]]; then
    use_uvm=1
  fi

  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi
  expect="${expect_mode[$base]-}"
  case "$expect" in
    skip)
      skip=$((skip + 1))
      continue
      ;;
    compile-only|parse-only)
      run_bmc=0
      ;;
    xfail)
      should_fail="1"
      ;;
  esac

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
  if [[ -z "$top_module" ]]; then
    top_module="top"
  fi

  test_root="$SV_TESTS_DIR/tests"
  if [[ -z "$files_line" ]]; then
    files=("$sv")
  else
    mapfile -t files < <(normalize_paths "$test_root" $files_line)
  fi
  if [[ -z "$incdirs_line" ]]; then
    incdirs=()
  else
    mapfile -t incdirs < <(normalize_paths "$test_root" $incdirs_line)
  fi
  incdirs+=("$(dirname "$sv")")

  defines=()
  if [[ -n "$defines_line" ]]; then
    for d in $defines_line; do
      defines+=("$d")
    done
  fi
  log_tag="$base"
  rel_path="${sv#"$test_root/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"

  mlir="$tmpdir/${base}.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  bmc_log="$tmpdir/${base}.circt-bmc.log"

  if [[ "$use_uvm" == "1" && ! -d "$UVM_PATH" ]]; then
    printf "ERROR\t%s\t%s (UVM path not found: %s)\n" \
      "$base" "$sv" "$UVM_PATH" >> "$results_tmp"
    error=$((error + 1))
    continue
  fi

  cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" && "$use_uvm" == "0" ]]; then
    cmd+=("--no-uvm-auto-include")
  fi
  if [[ "$use_uvm" == "1" ]]; then
    cmd+=("--uvm-path=$UVM_PATH")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
    cmd+=("${extra_args[@]}")
  fi
  for inc in "${incdirs[@]}"; do
    cmd+=("-I" "$inc")
  done
  for def in "${defines[@]}"; do
    cmd+=("-D" "$def")
  done
  if [[ -n "$top_module" ]]; then
    cmd+=("--top=$top_module")
  fi
  cmd+=("${files[@]}")

  if ! "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
    result="ERROR"
    if [[ "$should_fail" == "1" ]]; then
      result="XFAIL"
      xfail=$((xfail + 1))
    else
      error=$((error + 1))
    fi
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    continue
  fi

  if [[ "$run_bmc" == "0" ]]; then
    if [[ "$should_fail" == "1" ]]; then
      result="XPASS"
      xpass=$((xpass + 1))
    else
      result="PASS"
      pass=$((pass + 1))
    fi
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
    fi
    continue
  fi

  bmc_args=("-b" "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" \
    "--module" "$top_module")
  if [[ "$BMC_SMOKE_ONLY" != "1" && "$BMC_RUN_SMTLIB" == "1" ]]; then
    bmc_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
  elif [[ "$BMC_SMOKE_ONLY" != "1" ]]; then
    bmc_args+=("--shared-libs=$Z3_LIB")
  fi
  if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
    bmc_args+=("--rising-clocks-only")
  fi
  if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
    bmc_args+=("--allow-multi-clock")
  fi
  if [[ -n "$CIRCT_BMC_ARGS" ]]; then
    read -r -a extra_bmc_args <<<"$CIRCT_BMC_ARGS"
    bmc_args+=("${extra_bmc_args[@]}")
  fi
  out=""
  if out="$("$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2> "$bmc_log")"; then
    bmc_status=0
  else
    bmc_status=$?
  fi
  # NOTE: The "no property provided to check" warning is typically spurious.
  # It appears before LTLToCore and LowerClockedAssertLike passes run, but
  # after these passes, verif.clocked_assert (!ltl.property) becomes
  # verif.assert (i1), which is properly checked. This skip logic is disabled
  # by default (NO_PROPERTY_AS_SKIP=0) to avoid false SKIP results.
  if [[ "$NO_PROPERTY_AS_SKIP" == "1" ]] && \
      grep -q "no property provided to check in module" "$bmc_log"; then
    result="SKIP"
    skip=$((skip + 1))
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
      cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
        2>/dev/null || true
    fi
    continue
  fi

  if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
    if [[ "$bmc_status" -eq 0 ]]; then
      result="PASS"
    else
      result="ERROR"
    fi
  else
    if grep -q "BMC_RESULT=UNSAT" <<<"$out"; then
      result="PASS"
    elif grep -q "BMC_RESULT=SAT" <<<"$out"; then
      result="FAIL"
    else
      result="ERROR"
    fi
  fi

  if [[ "$BMC_SMOKE_ONLY" == "1" && "$should_fail" == "1" ]]; then
    result="XFAIL"
    xfail=$((xfail + 1))
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
      cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
        2>/dev/null || true
    fi
    continue
  fi

  if [[ "$should_fail" == "1" ]]; then
    if [[ "$result" == "PASS" ]]; then
      result="XPASS"
      xpass=$((xpass + 1))
    else
      result="XFAIL"
      xfail=$((xfail + 1))
    fi
  else
    case "$result" in
      PASS) pass=$((pass + 1)) ;;
      FAIL) fail=$((fail + 1)) ;;
      *) error=$((error + 1)) ;;
    esac
  fi

  printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
  if [[ -n "$KEEP_LOGS_DIR" ]]; then
    mkdir -p "$KEEP_LOGS_DIR"
    cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
    cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
      2>/dev/null || true
    cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
      2>/dev/null || true
  fi
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0)

sort "$results_tmp" > "$OUT"

echo "sv-tests SVA summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip"
echo "results: $OUT"
