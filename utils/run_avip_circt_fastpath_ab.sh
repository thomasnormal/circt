#!/usr/bin/env bash
# Deterministic interpreted-mode fast-path A/B runner with optional waveform
# differential checks, including optional Xcelium waveform comparison.
#
# Usage:
#   utils/run_avip_circt_fastpath_ab.sh [out_dir]
#
# Key env vars:
#   RUN_MATRIX=/path/to/run_avip_circt_sim.sh
#   CHECK_PARITY=/path/to/check_avip_circt_sim_mode_parity.py
#   CHECK_WAVE_PARITY=/path/to/check_avip_waveform_matrix_parity.py
#   COMPARE_VCD_TOOL=/path/to/compare_vcd_waveforms.py
#   RUN_XCELIUM_REFERENCE=/path/to/run_avip_xcelium_reference.sh
#
#   FAIL_ON_MISMATCH=0|1              (default: 1)   matrix A/B gate
#   COMPARE_WAVEFORMS=0|1             (default: 1)
#   FAIL_ON_WAVE_MISMATCH=0|1         (default: 1)
#   FAIL_ON_MISSING_VCD=0|1           (default: 1)
#   WAVE_COMPARE_ARGS="..."           (default: --require-same-signal-set)
#   WAVE_DUMP_VCD=0|1                 (default: 1)
#   WAVE_TRACE_ALL=0|1                (default: 1)
#
#   COMPARE_XCELIUM_WAVEFORMS=0|1     (default: 0)
#   XCELIUM_MATRIX=/path/to/matrix.tsv (optional; skips xcelium rerun)
#   XCELIUM_WAVE_COMPARE_ARGS="..."   (default: empty)
#   XCELIUM_FAIL_ON_WAVE_MISMATCH=0|1 (default: FAIL_ON_WAVE_MISMATCH)
#   XCELIUM_FAIL_ON_MISSING_VCD=0|1   (default: 0)
#
# All env vars accepted by utils/run_avip_circt_sim.sh and
# utils/run_avip_xcelium_reference.sh are forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/avip-circt-fastpath-ab-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

RUN_MATRIX="${RUN_MATRIX:-$SCRIPT_DIR/run_avip_circt_sim.sh}"
CHECK_PARITY="${CHECK_PARITY:-$SCRIPT_DIR/check_avip_circt_sim_mode_parity.py}"
CHECK_WAVE_PARITY="${CHECK_WAVE_PARITY:-$SCRIPT_DIR/check_avip_waveform_matrix_parity.py}"
COMPARE_VCD_TOOL="${COMPARE_VCD_TOOL:-$SCRIPT_DIR/compare_vcd_waveforms.py}"
RUN_XCELIUM_REFERENCE="${RUN_XCELIUM_REFERENCE:-$SCRIPT_DIR/run_avip_xcelium_reference.sh}"

FAIL_ON_MISMATCH="${FAIL_ON_MISMATCH:-1}"
COMPARE_WAVEFORMS="${COMPARE_WAVEFORMS:-1}"
FAIL_ON_WAVE_MISMATCH="${FAIL_ON_WAVE_MISMATCH:-1}"
FAIL_ON_MISSING_VCD="${FAIL_ON_MISSING_VCD:-1}"
WAVE_COMPARE_ARGS="${WAVE_COMPARE_ARGS:---require-same-signal-set}"
WAVE_DUMP_VCD="${WAVE_DUMP_VCD:-1}"
WAVE_TRACE_ALL="${WAVE_TRACE_ALL:-1}"

COMPARE_XCELIUM_WAVEFORMS="${COMPARE_XCELIUM_WAVEFORMS:-0}"
XCELIUM_MATRIX="${XCELIUM_MATRIX:-}"
XCELIUM_WAVE_COMPARE_ARGS="${XCELIUM_WAVE_COMPARE_ARGS:-}"
XCELIUM_FAIL_ON_WAVE_MISMATCH="${XCELIUM_FAIL_ON_WAVE_MISMATCH:-$FAIL_ON_WAVE_MISMATCH}"
XCELIUM_FAIL_ON_MISSING_VCD="${XCELIUM_FAIL_ON_MISSING_VCD:-0}"

if [[ ! -x "$RUN_MATRIX" ]]; then
  echo "error: matrix runner not found or not executable: $RUN_MATRIX" >&2
  exit 1
fi
if [[ ! -f "$CHECK_PARITY" ]]; then
  echo "error: parity checker script not found: $CHECK_PARITY" >&2
  exit 1
fi
if [[ ! -f "$CHECK_WAVE_PARITY" ]]; then
  echo "error: waveform parity checker script not found: $CHECK_WAVE_PARITY" >&2
  exit 1
fi
if [[ ! -f "$COMPARE_VCD_TOOL" ]]; then
  echo "error: waveform compare tool not found: $COMPARE_VCD_TOOL" >&2
  exit 1
fi
if [[ "$COMPARE_XCELIUM_WAVEFORMS" != "0" && ! -x "$RUN_XCELIUM_REFERENCE" ]]; then
  echo "error: xcelium runner not found or not executable: $RUN_XCELIUM_REFERENCE" >&2
  exit 1
fi

lane_off_out="$OUT_DIR/fastpath_off"
lane_on_out="$OUT_DIR/fastpath_on"
parity_tsv="$OUT_DIR/fastpath_parity.tsv"
wave_parity_tsv="$OUT_DIR/fastpath_wave_parity.tsv"

echo "[avip-fastpath-ab] out_dir=$OUT_DIR"
echo "[avip-fastpath-ab] running interpreted baseline lane (direct fast paths OFF)"
CIRCT_SIM_MODE=interpret \
CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=0 \
CIRCT_SIM_DUMP_VCD="$WAVE_DUMP_VCD" \
CIRCT_SIM_TRACE_ALL="$WAVE_TRACE_ALL" \
  "$RUN_MATRIX" "$lane_off_out"

echo "[avip-fastpath-ab] running interpreted experimental lane (direct fast paths ON)"
CIRCT_SIM_MODE=interpret \
CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=1 \
CIRCT_SIM_DUMP_VCD="$WAVE_DUMP_VCD" \
CIRCT_SIM_TRACE_ALL="$WAVE_TRACE_ALL" \
  "$RUN_MATRIX" "$lane_on_out"

parity_cmd=(
  python3 "$CHECK_PARITY"
  --interpret-matrix "$lane_off_out/matrix.tsv"
  --compile-matrix "$lane_on_out/matrix.tsv"
  --out-parity-tsv "$parity_tsv"
)
if [[ "$FAIL_ON_MISMATCH" != "0" ]]; then
  parity_cmd+=(--fail-on-mismatch)
fi
"${parity_cmd[@]}"

if [[ "$COMPARE_WAVEFORMS" != "0" ]]; then
  wave_cmd=(
    python3 "$CHECK_WAVE_PARITY"
    --lhs-matrix "$lane_off_out/matrix.tsv"
    --rhs-matrix "$lane_on_out/matrix.tsv"
    --lhs-label fastpath_off
    --rhs-label fastpath_on
    --compare-tool "$COMPARE_VCD_TOOL"
    --out-tsv "$wave_parity_tsv"
  )
  if [[ -n "$WAVE_COMPARE_ARGS" ]]; then
    wave_cmd+=(--compare-arg="$WAVE_COMPARE_ARGS")
  fi
  if [[ "$FAIL_ON_WAVE_MISMATCH" != "0" ]]; then
    wave_cmd+=(--fail-on-mismatch)
  fi
  if [[ "$FAIL_ON_MISSING_VCD" != "0" ]]; then
    wave_cmd+=(--fail-on-missing-vcd)
  fi
  "${wave_cmd[@]}"
fi

if [[ "$COMPARE_XCELIUM_WAVEFORMS" != "0" ]]; then
  if [[ -z "$XCELIUM_MATRIX" ]]; then
    xcelium_out="$OUT_DIR/xcelium"
    echo "[avip-fastpath-ab] running xcelium reference lane for waveform comparison"
    XCELIUM_COLLECT_VCD=1 "$RUN_XCELIUM_REFERENCE" "$xcelium_out"
    XCELIUM_MATRIX="$xcelium_out/matrix.tsv"
  fi

  if [[ ! -f "$XCELIUM_MATRIX" ]]; then
    echo "error: xcelium matrix not found: $XCELIUM_MATRIX" >&2
    exit 1
  fi

  x_wave_off_tsv="$OUT_DIR/fastpath_off_vs_xcelium_wave.tsv"
  x_wave_on_tsv="$OUT_DIR/fastpath_on_vs_xcelium_wave.tsv"

  x_wave_off_cmd=(
    python3 "$CHECK_WAVE_PARITY"
    --lhs-matrix "$lane_off_out/matrix.tsv"
    --rhs-matrix "$XCELIUM_MATRIX"
    --lhs-label fastpath_off
    --rhs-label xcelium
    --compare-tool "$COMPARE_VCD_TOOL"
    --out-tsv "$x_wave_off_tsv"
  )
  if [[ -n "$XCELIUM_WAVE_COMPARE_ARGS" ]]; then
    x_wave_off_cmd+=(--compare-arg="$XCELIUM_WAVE_COMPARE_ARGS")
  fi
  if [[ "$XCELIUM_FAIL_ON_WAVE_MISMATCH" != "0" ]]; then
    x_wave_off_cmd+=(--fail-on-mismatch)
  fi
  if [[ "$XCELIUM_FAIL_ON_MISSING_VCD" != "0" ]]; then
    x_wave_off_cmd+=(--fail-on-missing-vcd)
  fi
  "${x_wave_off_cmd[@]}"

  x_wave_on_cmd=(
    python3 "$CHECK_WAVE_PARITY"
    --lhs-matrix "$lane_on_out/matrix.tsv"
    --rhs-matrix "$XCELIUM_MATRIX"
    --lhs-label fastpath_on
    --rhs-label xcelium
    --compare-tool "$COMPARE_VCD_TOOL"
    --out-tsv "$x_wave_on_tsv"
  )
  if [[ -n "$XCELIUM_WAVE_COMPARE_ARGS" ]]; then
    x_wave_on_cmd+=(--compare-arg="$XCELIUM_WAVE_COMPARE_ARGS")
  fi
  if [[ "$XCELIUM_FAIL_ON_WAVE_MISMATCH" != "0" ]]; then
    x_wave_on_cmd+=(--fail-on-mismatch)
  fi
  if [[ "$XCELIUM_FAIL_ON_MISSING_VCD" != "0" ]]; then
    x_wave_on_cmd+=(--fail-on-missing-vcd)
  fi
  "${x_wave_on_cmd[@]}"

  echo "[avip-fastpath-ab] xcelium_matrix=$XCELIUM_MATRIX"
  echo "[avip-fastpath-ab] fastpath_off_vs_xcelium_wave_tsv=$x_wave_off_tsv"
  echo "[avip-fastpath-ab] fastpath_on_vs_xcelium_wave_tsv=$x_wave_on_tsv"
fi

echo "[avip-fastpath-ab] fastpath_off_matrix=$lane_off_out/matrix.tsv"
echo "[avip-fastpath-ab] fastpath_on_matrix=$lane_on_out/matrix.tsv"
echo "[avip-fastpath-ab] parity_tsv=$parity_tsv"
if [[ "$COMPARE_WAVEFORMS" != "0" ]]; then
  echo "[avip-fastpath-ab] wave_parity_tsv=$wave_parity_tsv"
fi
