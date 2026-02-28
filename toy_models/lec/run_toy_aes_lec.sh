#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="${1:-$REPO_ROOT/toy_models/out/toy_aes_lec}"

CIRCT_VERILOG_BIN="${CIRCT_VERILOG:-$REPO_ROOT/build_test/bin/circt-verilog}"
CIRCT_OPT_BIN="${CIRCT_OPT:-$REPO_ROOT/build_test/bin/circt-opt}"
CIRCT_LEC_BIN="${CIRCT_LEC:-$REPO_ROOT/build_test/bin/circt-lec}"
Z3_PATH="${Z3_BIN:-$(command -v z3 || true)}"

if [[ ! -x "$CIRCT_VERILOG_BIN" || ! -x "$CIRCT_OPT_BIN" || ! -x "$CIRCT_LEC_BIN" ]]; then
  echo "missing CIRCT tools; set CIRCT_VERILOG/CIRCT_OPT/CIRCT_LEC" >&2
  exit 1
fi

if [[ -z "$Z3_PATH" ]]; then
  echo "missing z3; set Z3_BIN" >&2
  exit 1
fi

mkdir -p "$WORKDIR"
MOORE_MLIR="$WORKDIR/toy_aes.moore.mlir"
CORE_MLIR="$WORKDIR/toy_aes.core.mlir"
LEC_LOG="$WORKDIR/toy_aes.lec.log"

"$CIRCT_VERILOG_BIN" \
  --ir-moore \
  --single-unit \
  --no-uvm-auto-include \
  --top=toy_aes_lec_ref \
  --top=toy_aes_lec_impl \
  -o "$MOORE_MLIR" \
  "$SCRIPT_DIR/toy_aes_equiv.sv"

# Match runner behavior: strip trailing vpi.* attributes that may fail parsing.
sed -E 's/ attributes \{vpi\..*$//' "$MOORE_MLIR" >"$MOORE_MLIR.tmp"
mv "$MOORE_MLIR.tmp" "$MOORE_MLIR"

"$CIRCT_OPT_BIN" \
  "$MOORE_MLIR" \
  --moore-lower-concatref \
  --convert-moore-to-core \
  --mlir-disable-threading \
  -o "$CORE_MLIR"

"$CIRCT_LEC_BIN" \
  "$CORE_MLIR" \
  -c1=toy_aes_lec_ref \
  -c2=toy_aes_lec_impl \
  --run-smtlib \
  "--z3-path=$Z3_PATH" \
  --verify-each=false \
  --mlir-timing \
  --mlir-timing-display=tree \
  >"$LEC_LOG" 2>&1

cat "$LEC_LOG"
