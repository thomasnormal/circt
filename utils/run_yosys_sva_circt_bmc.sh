#!/usr/bin/env bash
set -euo pipefail

YOSYS_SVA_DIR="${1:-/home/thomas-ahle/yosys/tests/sva}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
SKIP_VHDL="${SKIP_VHDL:-1}"
SKIP_FAIL_WITHOUT_MACRO="${SKIP_FAIL_WITHOUT_MACRO:-1}"

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

run_case() {
  local sv="$1"
  local mode="$2"
  if [[ "$mode" == "fail" && "$SKIP_FAIL_WITHOUT_MACRO" == "1" ]]; then
    if ! rg -q '^[[:space:]]*`(ifn?def|if)[[:space:]]+FAIL\b' "$sv"; then
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
  local mlir="$tmpdir/${base}_${mode}.mlir"

  local verilog_args=()
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
    verilog_args+=("--no-uvm-auto-include")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
    verilog_args+=("${extra_args[@]}")
  fi
  if ! "$CIRCT_VERILOG" --ir-hw "${verilog_args[@]}" "${extra_def[@]}" "$sv" > "$mlir"; then
    echo "FAIL($mode): $base"
    failures=$((failures + 1))
    return
  fi
  local out
  out="$("$CIRCT_BMC" -b "$BOUND" --ignore-asserts-until="$IGNORE_ASSERTS_UNTIL" \
      --module "$TOP" --shared-libs="$Z3_LIB" "$mlir" || true)"

  if [[ "$mode" == "pass" ]]; then
    if ! grep -q "Bound reached with no violations!" <<<"$out"; then
      echo "FAIL(pass): $base"
      failures=$((failures + 1))
    else
      echo "PASS(pass): $base"
    fi
  else
    if ! grep -q "Assertion can be violated!" <<<"$out"; then
      echo "FAIL(fail): $base"
      failures=$((failures + 1))
    else
      echo "PASS(fail): $base"
    fi
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

echo "yosys SVA summary: $total tests, failures=$failures, skipped=$skipped"
exit "$failures"
