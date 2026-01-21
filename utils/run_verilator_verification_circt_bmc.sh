#!/usr/bin/env bash
set -euo pipefail

VERIF_DIR="${1:-/home/thomas-ahle/verilator-verification}"
shift || true
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/verilator-verification-bmc-results.txt}"
XFAILS="${XFAILS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"

if [[ ! -d "$VERIF_DIR/tests" ]]; then
  echo "verilator-verification directory not found: $VERIF_DIR" >&2
  exit 1
fi

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
  suites=("$VERIF_DIR/tests/asserts" "$VERIF_DIR/tests/sequences" \
    "$VERIF_DIR/tests/event-control-expression")
else
  for i in "${!suites[@]}"; do
    if [[ "${suites[$i]}" != /* ]]; then
      suites[$i]="$VERIF_DIR/${suites[$i]}"
    fi
  done
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

is_xfail() {
  local name="$1"
  if [[ -z "$XFAILS" ]]; then
    return 1
  fi
  if [[ ! -f "$XFAILS" ]]; then
    return 1
  fi
  grep -Fxq "$name" "$XFAILS"
}

for suite in "${suites[@]}"; do
  if [[ ! -d "$suite" ]]; then
    echo "suite not found: $suite" >&2
    continue
  fi
  while IFS= read -r -d '' sv; do
    base="$(basename "$sv" .sv)"
    if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
      skip=$((skip + 1))
      continue
    fi

    total=$((total + 1))

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
    cmd+=("-I" "$(dirname "$sv")")
    cmd+=("--top=$TOP")
    cmd+=("$sv")

    if ! "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
      result="ERROR"
      if is_xfail "$base"; then
        result="XFAIL"
        xfail=$((xfail + 1))
      else
        error=$((error + 1))
      fi
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      continue
    fi

    out="$("$CIRCT_BMC" -b "$BOUND" --ignore-asserts-until="$IGNORE_ASSERTS_UNTIL" \
      --module "$TOP" --shared-libs="$Z3_LIB" "$mlir" 2> "$bmc_log" || true)"

    if grep -q "Bound reached with no violations!" <<<"$out"; then
      result="PASS"
    elif grep -q "Assertion can be violated!" <<<"$out"; then
      result="FAIL"
    else
      result="ERROR"
    fi

    if is_xfail "$base"; then
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
  done < <(find "$suite" -type f -name "*.sv" -print0)
done

sort "$results_tmp" > "$OUT"

echo "verilator-verification summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip"
echo "results: $OUT"
