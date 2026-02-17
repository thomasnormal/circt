#!/usr/bin/env bash
# Deterministic compile-mode AVIP runner + JIT deopt policy gate.
#
# Usage:
#   utils/run_avip_circt_sim_jit_policy_gate.sh [out_dir]
#
# Key env vars:
#   RUN_MATRIX=/path/to/run_avip_circt_sim.sh
#   SUMMARIZE_JIT_REPORTS=/path/to/summarize_circt_sim_jit_reports.py
#   JIT_REPORT_GLOB=*.jit-report.json
#   COMPILE_WRITE_JIT_REPORT=0|1                       (default: 1)
#   JIT_POLICY_ALLOWLIST_FILE=/path/to/allowlist.txt   (optional)
#   JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT=0|1   (default: 1)
#   JIT_POLICY_FAIL_ON_REASON=reason0,reason1          (optional CSV)
#   JIT_POLICY_FAIL_ON_REASON_DETAIL=reason:detail,... (optional CSV)
#
# All env vars accepted by utils/run_avip_circt_sim.sh are forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/avip-circt-sim-jit-policy-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

RUN_MATRIX="${RUN_MATRIX:-$SCRIPT_DIR/run_avip_circt_sim.sh}"
SUMMARIZE_JIT_REPORTS="${SUMMARIZE_JIT_REPORTS:-$SCRIPT_DIR/summarize_circt_sim_jit_reports.py}"
JIT_REPORT_GLOB="${JIT_REPORT_GLOB:-*.jit-report.json}"
COMPILE_WRITE_JIT_REPORT="${COMPILE_WRITE_JIT_REPORT:-1}"
JIT_POLICY_ALLOWLIST_FILE="${JIT_POLICY_ALLOWLIST_FILE:-}"
JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT="${JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT:-1}"
JIT_POLICY_FAIL_ON_REASON="${JIT_POLICY_FAIL_ON_REASON:-}"
JIT_POLICY_FAIL_ON_REASON_DETAIL="${JIT_POLICY_FAIL_ON_REASON_DETAIL:-}"

if [[ ! -x "$RUN_MATRIX" ]]; then
  echo "error: matrix runner not found or not executable: $RUN_MATRIX" >&2
  exit 1
fi
if [[ ! -f "$SUMMARIZE_JIT_REPORTS" ]]; then
  echo "error: JIT summarizer script not found: $SUMMARIZE_JIT_REPORTS" >&2
  exit 1
fi
if [[ -n "$JIT_POLICY_ALLOWLIST_FILE" && ! -f "$JIT_POLICY_ALLOWLIST_FILE" ]]; then
  echo "error: JIT policy allowlist file not found: $JIT_POLICY_ALLOWLIST_FILE" >&2
  exit 1
fi

compile_out="$OUT_DIR/compile"
summary_log="$OUT_DIR/jit_deopt_summary.log"
reason_tsv="$OUT_DIR/jit_deopt_reasons.tsv"
detail_tsv="$OUT_DIR/jit_deopt_reason_details.tsv"
process_tsv="$OUT_DIR/jit_deopt_processes.tsv"

echo "[avip-circt-sim-jit-policy] out_dir=$OUT_DIR"
echo "[avip-circt-sim-jit-policy] running compile matrix"
CIRCT_SIM_MODE=compile CIRCT_SIM_WRITE_JIT_REPORT="$COMPILE_WRITE_JIT_REPORT" \
  "$RUN_MATRIX" "$compile_out"

mapfile -t jit_reports < <(find "$compile_out" -type f -name "$JIT_REPORT_GLOB" | sort)
if [[ ${#jit_reports[@]} -eq 0 ]]; then
  echo "error: no JIT report files found under $compile_out (glob=$JIT_REPORT_GLOB)" >&2
  exit 1
fi

summary_cmd=(
  python3 "$SUMMARIZE_JIT_REPORTS"
  "${jit_reports[@]}"
  --out-reason-tsv "$reason_tsv"
  --out-detail-tsv "$detail_tsv"
  --out-process-tsv "$process_tsv"
)
if [[ -n "$JIT_POLICY_ALLOWLIST_FILE" ]]; then
  summary_cmd+=(--allowlist-file "$JIT_POLICY_ALLOWLIST_FILE")
fi
if [[ "$JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT" != "0" ]]; then
  summary_cmd+=(--fail-on-any-non-allowlisted-deopt)
fi
if [[ -n "$JIT_POLICY_FAIL_ON_REASON" ]]; then
  IFS=',' read -ra blocked_reasons <<< "$JIT_POLICY_FAIL_ON_REASON"
  for reason in "${blocked_reasons[@]}"; do
    reason="${reason//[[:space:]]/}"
    [[ -z "$reason" ]] && continue
    summary_cmd+=(--fail-on-reason "$reason")
  done
fi
if [[ -n "$JIT_POLICY_FAIL_ON_REASON_DETAIL" ]]; then
  IFS=',' read -ra blocked_reason_details <<< "$JIT_POLICY_FAIL_ON_REASON_DETAIL"
  for reason_detail in "${blocked_reason_details[@]}"; do
    reason_detail="${reason_detail#${reason_detail%%[![:space:]]*}}"
    reason_detail="${reason_detail%${reason_detail##*[![:space:]]}}"
    [[ -z "$reason_detail" ]] && continue
    summary_cmd+=(--fail-on-reason-detail "$reason_detail")
  done
fi

"${summary_cmd[@]}" 2>&1 | tee "$summary_log"

echo "[avip-circt-sim-jit-policy] compile_matrix=$compile_out/matrix.tsv"
echo "[avip-circt-sim-jit-policy] summary_log=$summary_log"
echo "[avip-circt-sim-jit-policy] reason_tsv=$reason_tsv"
echo "[avip-circt-sim-jit-policy] detail_tsv=$detail_tsv"
echo "[avip-circt-sim-jit-policy] process_tsv=$process_tsv"
