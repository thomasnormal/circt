#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/base_runner.py" <<'PY'
import json
from pathlib import Path

WORK_DIR = Path("/tmp/cvdp-stub-run")
COMPILE_TIMEOUT = 1
SIM_TIMEOUT = 1

def extract_files(datapoint, problem_dir):
    # No SV present -> wrapper must synthesize stub module.
    (problem_dir / "src").mkdir(parents=True, exist_ok=True)
    py = problem_dir / "src" / "test_dummy.py"
    py.write_text("import cocotb\n@cocotb.test()\nasync def test_smoke(dut):\n    assert dut is not None\n")
    env = {"TOPLEVEL": "stub_top", "MODULE": "test_dummy"}
    return [], [py], env

def compile_sv(sv_files, output_mlir):
    # Stub module must be present and named correctly.
    assert len(sv_files) == 1
    sv_text = Path(sv_files[0]).read_text()
    assert "module stub_top" in sv_text
    output_mlir.write_text("hw.module @stub_top() {\n  hw.output\n}\n")
    return True, ""

def run_cocotb_sim(mlir_file, top_module, test_module, test_dir, sv_files=None, max_time_fs=0):
    # Ensure wrapper forwarded the stub and bounded max_time_fs.
    assert top_module == "stub_top"
    assert test_module == "test_dummy"
    assert sv_files and len(sv_files) == 1
    assert max_time_fs and max_time_fs < 10_000_000_000
    return True, 1, 0, "cocotb                             Running tests\nTESTS=1 PASS=1 FAIL=0\n"
PY

cat >"$tmpdir/ds.jsonl" <<'JSONL'
{"id":"cvdp_copilot_stubcase_0001","categories":[],"harness":{"files":{"src/.env":"TOPLEVEL=stub_top\nMODULE=test_dummy\n"}},"input":{"context":{}},"output":{"context":{}}}
JSONL

out="$tmpdir/out"
mkdir -p "$out"

set +e
output="$(
  CVDP_COCOTB_BASE_RUNNER="$tmpdir/base_runner.py" \
  CVDP_STUB_MAX_TIME_FS=2000000000 \
  python3 "$REPO_ROOT/utils/run_cvdp_cocotb_runner.py" -f "$tmpdir/ds.jsonl" -o "$out" 2>&1
)"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[cvdp-cocotb-stub-sv-check] wrapper returned rc=$rc" >&2
  echo "$output" >&2
  exit 1
fi

if ! grep -Fq '[STUB_SV] cvdp_copilot_stubcase_0001' <<<"$output"; then
  echo "[cvdp-cocotb-stub-sv-check] missing STUB_SV marker" >&2
  echo "$output" >&2
  exit 1
fi
if ! grep -Fq '[COCOTB_PASS' <<<"$output"; then
  echo "[cvdp-cocotb-stub-sv-check] missing cocotb PASS marker" >&2
  echo "$output" >&2
  exit 1
fi

results_json="$out/circt_cocotb_results.json"
if [[ ! -f "$results_json" ]]; then
  echo "[cvdp-cocotb-stub-sv-check] missing results json: $results_json" >&2
  exit 1
fi

python3 - "$results_json" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1]))
r=obj["results"]
assert "stub_sv" in r, "missing stub_sv list"
assert r["stub_sv"] == ["cvdp_copilot_stubcase_0001"], r["stub_sv"]
assert r["compile_pass"] == ["cvdp_copilot_stubcase_0001"], r["compile_pass"]
assert r["cocotb_pass"] == ["cvdp_copilot_stubcase_0001"], r["cocotb_pass"]
print("OK")
PY

echo "[cvdp-cocotb-stub-sv-check] PASS"
