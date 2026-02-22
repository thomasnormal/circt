#!/usr/bin/env bash
# Deterministic interpret-vs-compile parity wrapper for circt-sim AVIP matrix.
#
# Usage:
#   utils/run_avip_circt_sim_mode_parity.sh [out_dir]
#
# Key env vars:
#   FAIL_ON_MISMATCH=0|1                    (default: 1)
#   ALLOWLIST_FILE=/path/to/allowlist.txt   (optional)
#   INTERPRET_WRITE_JIT_REPORT=0|1          (default: 0)
#   COMPILE_WRITE_JIT_REPORT=0|1            (default: 1)
#
# All env vars accepted by utils/run_avip_circt_sim.sh are forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/avip-circt-sim-mode-parity-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

RUN_MATRIX="${RUN_MATRIX:-$SCRIPT_DIR/run_avip_circt_sim.sh}"
CHECK_PARITY="${CHECK_PARITY:-$SCRIPT_DIR/check_avip_circt_sim_mode_parity.py}"
FAIL_ON_MISMATCH="${FAIL_ON_MISMATCH:-1}"
ALLOWLIST_FILE="${ALLOWLIST_FILE:-}"
INTERPRET_WRITE_JIT_REPORT="${INTERPRET_WRITE_JIT_REPORT:-0}"
COMPILE_WRITE_JIT_REPORT="${COMPILE_WRITE_JIT_REPORT:-1}"

if [[ ! -x "$RUN_MATRIX" ]]; then
  echo "error: matrix runner not found or not executable: $RUN_MATRIX" >&2
  exit 1
fi
if [[ ! -f "$CHECK_PARITY" ]]; then
  echo "error: parity checker script not found: $CHECK_PARITY" >&2
  exit 1
fi

interpret_out="$OUT_DIR/interpret"
compile_out="$OUT_DIR/compile"
parity_tsv="$OUT_DIR/parity.tsv"

echo "[avip-circt-sim-parity] out_dir=$OUT_DIR"
echo "[avip-circt-sim-parity] running interpret matrix"
CIRCT_SIM_MODE=interpret CIRCT_SIM_WRITE_JIT_REPORT="$INTERPRET_WRITE_JIT_REPORT" \
  "$RUN_MATRIX" "$interpret_out"

echo "[avip-circt-sim-parity] running compile matrix"
CIRCT_SIM_MODE=compile CIRCT_SIM_WRITE_JIT_REPORT="$COMPILE_WRITE_JIT_REPORT" \
  "$RUN_MATRIX" "$compile_out"

check_cmd=(
  python3 "$CHECK_PARITY"
  --interpret-matrix "$interpret_out/matrix.tsv"
  --compile-matrix "$compile_out/matrix.tsv"
  --out-parity-tsv "$parity_tsv"
)
if [[ -n "$ALLOWLIST_FILE" ]]; then
  check_cmd+=(--allowlist-file "$ALLOWLIST_FILE")
fi
if [[ "$FAIL_ON_MISMATCH" != "0" ]]; then
  check_cmd+=(--fail-on-mismatch)
fi

"${check_cmd[@]}"

echo "[avip-circt-sim-parity] interpret_matrix=$interpret_out/matrix.tsv"
echo "[avip-circt-sim-parity] compile_matrix=$compile_out/matrix.tsv"
echo "[avip-circt-sim-parity] parity_tsv=$parity_tsv"
