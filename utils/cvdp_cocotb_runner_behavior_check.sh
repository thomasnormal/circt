#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_PY="$SCRIPT_DIR/run_cvdp_cocotb_runner.py"

if [[ ! -f "$RUNNER_PY" ]]; then
  echo "[cvdp-cocotb-runner-check] missing runner wrapper: $RUNNER_PY" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[cvdp-cocotb-runner-check] case: preserve valid cocotb_tools fallback"
cat >"$tmpdir/harness_library.py" <<'PY'
from cocotb.triggers import RisingEdge
try:
    from cocotb_tools.runner import get_runner
except ImportError:
    from cocotb.runner import get_runner
PY

python3 - "$RUNNER_PY" "$tmpdir/harness_library.py" <<'PY'
import importlib.util
import py_compile
import sys
from pathlib import Path

runner_path = Path(sys.argv[1])
target = Path(sys.argv[2])

spec = importlib.util.spec_from_file_location("cvdp_runner_wrapper", runner_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

before = target.read_text()
after = mod.patch_cocotb_compat_source(before)
target.write_text(after)
py_compile.compile(str(target), doraise=True)

if after.count("from cocotb_tools.runner import get_runner") != 1:
    raise SystemExit("expected exactly one cocotb_tools import")
if "except ImportError:\n    from cocotb.runner import get_runner" not in after:
    raise SystemExit("expected cocotb fallback import block")
PY

echo "[cvdp-cocotb-runner-check] case: timeout classification is strict"
python3 - "$RUNNER_PY" <<'PY'
import importlib.util
import sys
from pathlib import Path

runner_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cvdp_runner_wrapper", runner_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

status = mod.classify_result(ok=False, passes=0, fails=1, output="Single-parameter TIMEOUT checks failed")
if status != "COCOTB_FAIL":
    raise SystemExit(f"expected COCOTB_FAIL, got {status}")

status = mod.classify_result(ok=False, passes=0, fails=0, output="TIMEOUT\npartial log")
if status != "SIM_TIMEOUT":
    raise SystemExit(f"expected SIM_TIMEOUT, got {status}")

status = mod.classify_result(
    ok=False,
    passes=0,
    fails=0,
    output="TIMEOUT\n[circt-sim] Stage: init (prev: 1ms, total: 2ms)\n",
)
if status != "COCOTB_FAIL":
    raise SystemExit(f"expected COCOTB_FAIL for timed-out initialized sim, got {status}")

status = mod.classify_result(
    ok=False,
    passes=0,
    fails=0,
    output="error: resource guard triggered: wall time exceeded",
)
if status != "COCOTB_FAIL":
    raise SystemExit(f"expected COCOTB_FAIL for resource guard, got {status}")
PY

echo "[cvdp-cocotb-runner-check] PASS"
