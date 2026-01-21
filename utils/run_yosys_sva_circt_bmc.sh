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

run_case() {
  local sv="$1"
  local mode="$2"
  local extra_def=()
  if [[ "$mode" == "fail" ]]; then
    extra_def=(-DFAIL)
  fi
  local base
  base="$(basename "$sv" .sv)"
  local mlir="$tmpdir/${base}_${mode}.mlir"

  "$CIRCT_VERILOG" --ir-hw "${extra_def[@]}" "$sv" > "$mlir"
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
  total=$((total + 1))
  run_case "$sv" pass
  run_case "$sv" fail
done

echo "yosys SVA summary: $total tests, failures=$failures"
exit "$failures"
