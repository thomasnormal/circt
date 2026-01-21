#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
TAG_REGEX="${TAG_REGEX:-(^| )16\\.|(^| )9\\.4\\.4}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/sv-tests-bmc-results.txt}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
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
  if ! [[ "$tags" =~ $TAG_REGEX ]]; then
    skip=$((skip + 1))
    continue
  fi

  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  total=$((total + 1))

  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi

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

  mlir="$tmpdir/${base}.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  bmc_log="$tmpdir/${base}.circt-bmc.log"

  cmd=("$CIRCT_VERILOG" --ir-hw --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
    cmd+=("--no-uvm-auto-include")
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

  out="$("$CIRCT_BMC" -b "$BOUND" --ignore-asserts-until="$IGNORE_ASSERTS_UNTIL" \
    --module "$top_module" --shared-libs="$Z3_LIB" "$mlir" 2> "$bmc_log" || true)"

  if grep -q "Bound reached with no violations!" <<<"$out"; then
    result="PASS"
  elif grep -q "Assertion can be violated!" <<<"$out"; then
    result="FAIL"
  else
    result="ERROR"
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
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0)

sort "$results_tmp" > "$OUT"

echo "sv-tests SVA summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip"
echo "results: $OUT"
