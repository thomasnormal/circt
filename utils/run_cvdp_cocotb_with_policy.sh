#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNNER="${CVDP_POLICY_RUNNER:-python3 $SCRIPT_DIR/run_cvdp_cocotb_runner.py}"
MIN_COCOTB_PASS="${CVDP_MIN_COCOTB_PASS:-20}"
MAX_RUNTIME_FAILS="${CVDP_MAX_RUNTIME_FAILS:-40}"
STRICT_RC="${CVDP_POLICY_STRICT_RC:-0}"
RUNNER_TIMEOUT_SEC="${CVDP_POLICY_TIMEOUT_SEC:-0}"

if ! [[ "$MIN_COCOTB_PASS" =~ ^[0-9]+$ ]]; then
  echo "[cvdp-cocotb-policy] invalid CVDP_MIN_COCOTB_PASS: $MIN_COCOTB_PASS" >&2
  exit 2
fi
if ! [[ "$MAX_RUNTIME_FAILS" =~ ^[0-9]+$ ]]; then
  echo "[cvdp-cocotb-policy] invalid CVDP_MAX_RUNTIME_FAILS: $MAX_RUNTIME_FAILS" >&2
  exit 2
fi
if [[ "$STRICT_RC" != "0" && "$STRICT_RC" != "1" ]]; then
  echo "[cvdp-cocotb-policy] invalid CVDP_POLICY_STRICT_RC: $STRICT_RC" >&2
  exit 2
fi
if ! [[ "$RUNNER_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "[cvdp-cocotb-policy] invalid CVDP_POLICY_TIMEOUT_SEC: $RUNNER_TIMEOUT_SEC" >&2
  exit 2
fi

OUT_DIR=""
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do
  if [[ "${args[i]}" == "-o" || "${args[i]}" == "--output" ]]; then
    if ((i + 1 < ${#args[@]})); then
      OUT_DIR="${args[i + 1]}"
    fi
  fi
done
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/unified-cvdp-commercial"
  args+=(-o "$OUT_DIR")
fi
mkdir -p "$OUT_DIR"
results_json="$OUT_DIR/circt_cocotb_results.json"
rm -f "$results_json"

set +e
cmd="$RUNNER"
for arg in "${args[@]}"; do
  cmd+=" $(printf '%q' "$arg")"
done
if [[ "$RUNNER_TIMEOUT_SEC" -gt 0 ]]; then
  timeout --signal=TERM --kill-after=5 "${RUNNER_TIMEOUT_SEC}s" bash -lc "$cmd" 2>&1
else
  bash -lc "$cmd" 2>&1
fi
runner_rc=$?
set -e

if [[ "$runner_rc" -eq 124 ]]; then
  echo "[cvdp-cocotb-policy] FAIL: runner timed out after ${RUNNER_TIMEOUT_SEC}s" >&2
  exit 1
fi

if [[ ! -f "$results_json" ]]; then
  echo "[cvdp-cocotb-policy] missing results json: $results_json (runner_rc=$runner_rc)" >&2
  exit 1
fi

read -r compile_pass cocotb_pass cocotb_fail sim_fail sim_timeout < <(
  python3 - "$results_json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    obj = json.load(f)
r = obj.get("results", {})
def n(k):
    return len(r.get(k, []))
print(n("compile_pass"), n("cocotb_pass"), n("cocotb_fail"), n("sim_fail"), n("sim_timeout"))
PY
)

functional_mismatches=$cocotb_fail
runtime_fails=$((sim_fail + sim_timeout))

echo "[cvdp-cocotb-policy] summary: compile_pass=$compile_pass cocotb_pass=$cocotb_pass cocotb_fail=$cocotb_fail functional_mismatches=$functional_mismatches sim_fail=$sim_fail sim_timeout=$sim_timeout runtime_fails=$runtime_fails runner_rc=$runner_rc min_pass=$MIN_COCOTB_PASS max_runtime_fails=$MAX_RUNTIME_FAILS"

if [[ "$compile_pass" -eq 0 ]]; then
  echo "[cvdp-cocotb-policy] FAIL: no compile-pass entries" >&2
  exit 1
fi
if [[ "$cocotb_pass" -lt "$MIN_COCOTB_PASS" ]]; then
  echo "[cvdp-cocotb-policy] FAIL: cocotb_pass=$cocotb_pass < min=$MIN_COCOTB_PASS" >&2
  exit 1
fi
if [[ "$runtime_fails" -gt "$MAX_RUNTIME_FAILS" ]]; then
  echo "[cvdp-cocotb-policy] FAIL: runtime_fails=$runtime_fails > max=$MAX_RUNTIME_FAILS" >&2
  exit 1
fi
if [[ "$STRICT_RC" == "1" && "$runner_rc" -ne 0 ]]; then
  echo "[cvdp-cocotb-policy] FAIL: runner_rc=$runner_rc in strict mode" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy] PASS"
