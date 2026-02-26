#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
WRAPPER="$REPO_ROOT/utils/run_cvdp_golden_smoke_runner.py"

if [[ ! -x "$WRAPPER" ]]; then
  echo "[cvdp-golden-runner-check] missing executable wrapper: $WRAPPER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/base_runner.py" <<'PY'
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

CIRCT_VERILOG = Path("__VERILOG__")
CIRCT_SIM = Path("__SIM__")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()
    out = Path(args.output or ".")
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner_state.json").write_text(
        json.dumps({"circt_verilog": str(CIRCT_VERILOG), "circt_sim": str(CIRCT_SIM)})
    )

if __name__ == "__main__":
    main()
PY

cat >"$tmpdir/circt-verilog" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"$tmpdir/circt-sim" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$tmpdir/circt-verilog" "$tmpdir/circt-sim"

sed -i "s|__VERILOG__|$tmpdir/circt-verilog|g" "$tmpdir/base_runner.py"
sed -i "s|__SIM__|$tmpdir/circt-sim|g" "$tmpdir/base_runner.py"

out_dir="$tmpdir/out"
mkdir -p "$out_dir"

CVDP_GOLDEN_BASE_RUNNER="$tmpdir/base_runner.py" \
python3 "$WRAPPER" -o "$out_dir"

python3 - "$out_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
state = json.loads((out / "runner_state.json").read_text())
sim = Path(state["circt_sim"])
verilog = Path(state["circt_verilog"])

if ".tool-snapshot" not in str(sim) or sim.name != "circt-sim":
    raise SystemExit("circt-sim path was not snapshot-based")
if ".tool-snapshot" not in str(verilog) or verilog.name != "circt-verilog":
    raise SystemExit("circt-verilog path was not snapshot-based")
if not sim.exists() or not verilog.exists():
    raise SystemExit("snapshot tools were not created")
if not os.access(sim, os.X_OK) or not os.access(verilog, os.X_OK):
    raise SystemExit("snapshot tools are not executable")
PY

echo "[cvdp-golden-runner-check] PASS"

echo "[cvdp-golden-runner-check] case: missing tools fail"
cat >"$tmpdir/base_runner_missing.py" <<'PY'
#!/usr/bin/env python3
import argparse
from pathlib import Path

CIRCT_VERILOG = Path("__MISSING_VERILOG__")
CIRCT_SIM = Path("__MISSING_SIM__")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="")
    parser.parse_args()

if __name__ == "__main__":
    main()
PY

missing_verilog="$tmpdir/build_test/bin/circt-verilog"
missing_sim="$tmpdir/build_test/bin/circt-sim"
sed -i "s|__MISSING_VERILOG__|$missing_verilog|g" "$tmpdir/base_runner_missing.py"
sed -i "s|__MISSING_SIM__|$missing_sim|g" "$tmpdir/base_runner_missing.py"

set +e
CVDP_GOLDEN_BASE_RUNNER="$tmpdir/base_runner_missing.py" \
python3 "$WRAPPER" -o "$tmpdir/out-missing" >/dev/null 2>"$tmpdir/missing.err"
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo "[cvdp-golden-runner-check] expected failure for missing tools" >&2
  exit 1
fi
if ! rg -q 'tool not found' "$tmpdir/missing.err"; then
  echo "[cvdp-golden-runner-check] expected tool-not-found diagnostic" >&2
  cat "$tmpdir/missing.err" >&2
  exit 1
fi

echo "[cvdp-golden-runner-check] PASS"
