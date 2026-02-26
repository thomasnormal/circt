#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
WRAPPER="$REPO_ROOT/utils/run_cvdp_cocotb_with_policy.sh"

if [[ ! -x "$WRAPPER" ]]; then
  echo "[cvdp-cocotb-policy-check] missing executable wrapper: $WRAPPER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mock_runner="$tmpdir/mock_runner.py"
cat >"$mock_runner" <<'PY'
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

out_dir = None
for i, arg in enumerate(sys.argv):
    if arg in ("-o", "--output") and i + 1 < len(sys.argv):
        out_dir = Path(sys.argv[i + 1])
        break
if out_dir is None:
    out_dir = Path(os.environ.get("CVDP_POLICY_OUT_DIR", "/tmp/cvdp-policy-out"))
out_dir.mkdir(parents=True, exist_ok=True)

mode = os.environ.get("CVDP_POLICY_TEST_MODE", "good")
if mode == "good":
    results = {
        "results": {
            "compile_pass": ["a", "b", "c"],
            "compile_fail": [],
            "no_sv": [],
            "sim_pass": [],
            "sim_fail": [],
            "sim_timeout": [],
            "cocotb_pass": ["p1", "p2", "p3", "p4"],
            "cocotb_fail": ["f1"],
        }
    }
elif mode == "lowpass":
    results = {
        "results": {
            "compile_pass": ["a"],
            "compile_fail": [],
            "no_sv": [],
            "sim_pass": [],
            "sim_fail": [],
            "sim_timeout": [],
            "cocotb_pass": ["p1"],
            "cocotb_fail": [],
        }
    }
elif mode == "toomanyfails":
    results = {
        "results": {
            "compile_pass": ["a", "b"],
            "compile_fail": [],
            "no_sv": [],
            "sim_pass": [],
            "sim_fail": ["s1", "s2", "s3"],
            "sim_timeout": ["t1"],
            "cocotb_pass": ["p1", "p2", "p3", "p4"],
            "cocotb_fail": ["f1", "f2", "f3"],
        }
    }
elif mode == "cocotb_only_fails":
    results = {
        "results": {
            "compile_pass": ["a", "b", "c", "d"],
            "compile_fail": [],
            "no_sv": [],
            "sim_pass": [],
            "sim_fail": [],
            "sim_timeout": [],
            "cocotb_pass": ["p1", "p2", "p3", "p4"],
            "cocotb_fail": ["f1", "f2", "f3", "f4", "f5", "f6"],
        }
    }
elif mode == "hang":
    import time
    time.sleep(2.5)
    results = {
        "results": {
            "compile_pass": ["a", "b"],
            "compile_fail": [],
            "no_sv": [],
            "sim_pass": [],
            "sim_fail": [],
            "sim_timeout": [],
            "cocotb_pass": ["p1", "p2", "p3", "p4"],
            "cocotb_fail": [],
        }
    }
else:
    raise RuntimeError(f"unknown mode: {mode}")

(out_dir / "circt_cocotb_results.json").write_text(json.dumps(results))
sys.exit(1)
PY
chmod +x "$mock_runner"

echo "[cvdp-cocotb-policy-check] case: rc-nonzero-but-meets-thresholds"
CVDP_POLICY_RUNNER="$mock_runner" \
CVDP_POLICY_TEST_MODE=good \
CVDP_MIN_COCOTB_PASS=3 \
CVDP_MAX_RUNTIME_FAILS=2 \
  "$WRAPPER" -f /dev/null -o "$tmpdir/good" >"$tmpdir/good.out" 2>&1
if ! grep -q '\[cvdp-cocotb-policy\] PASS' "$tmpdir/good.out"; then
  echo "[cvdp-cocotb-policy-check] good case missing PASS marker" >&2
  cat "$tmpdir/good.out" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy-check] case: low-pass-must-fail"
set +e
CVDP_POLICY_RUNNER="$mock_runner" \
CVDP_POLICY_TEST_MODE=lowpass \
CVDP_MIN_COCOTB_PASS=3 \
CVDP_MAX_RUNTIME_FAILS=2 \
  "$WRAPPER" -f /dev/null -o "$tmpdir/lowpass" >"$tmpdir/lowpass.out" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[cvdp-cocotb-policy-check] low-pass case unexpectedly passed" >&2
  cat "$tmpdir/lowpass.out" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy-check] case: too-many-runtime-fails-must-fail"
set +e
CVDP_POLICY_RUNNER="$mock_runner" \
CVDP_POLICY_TEST_MODE=toomanyfails \
CVDP_MIN_COCOTB_PASS=3 \
CVDP_MAX_RUNTIME_FAILS=2 \
  "$WRAPPER" -f /dev/null -o "$tmpdir/toomany" >"$tmpdir/toomany.out" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[cvdp-cocotb-policy-check] too-many-fails case unexpectedly passed" >&2
  cat "$tmpdir/toomany.out" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy-check] case: cocotb-fails-are-functional-mismatch"
CVDP_POLICY_RUNNER="$mock_runner" \
CVDP_POLICY_TEST_MODE=cocotb_only_fails \
CVDP_MIN_COCOTB_PASS=3 \
CVDP_MAX_RUNTIME_FAILS=0 \
  "$WRAPPER" -f /dev/null -o "$tmpdir/cocotb-only-fails" >"$tmpdir/cocotb-only-fails.out" 2>&1
if ! grep -q '\[cvdp-cocotb-policy\] PASS' "$tmpdir/cocotb-only-fails.out"; then
  echo "[cvdp-cocotb-policy-check] cocotb-only-fails case missing PASS marker" >&2
  cat "$tmpdir/cocotb-only-fails.out" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy-check] case: timeout-must-fail-fast"
set +e
CVDP_POLICY_RUNNER="$mock_runner" \
CVDP_POLICY_TEST_MODE=hang \
CVDP_POLICY_TIMEOUT_SEC=1 \
CVDP_MIN_COCOTB_PASS=3 \
CVDP_MAX_RUNTIME_FAILS=2 \
  "$WRAPPER" -f /dev/null -o "$tmpdir/hang" >"$tmpdir/hang.out" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[cvdp-cocotb-policy-check] timeout case unexpectedly passed" >&2
  cat "$tmpdir/hang.out" >&2
  exit 1
fi
if ! grep -q 'runner timed out' "$tmpdir/hang.out"; then
  echo "[cvdp-cocotb-policy-check] timeout case missing timeout marker" >&2
  cat "$tmpdir/hang.out" >&2
  exit 1
fi

echo "[cvdp-cocotb-policy-check] PASS"
