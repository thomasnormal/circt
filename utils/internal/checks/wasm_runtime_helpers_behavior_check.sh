#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

SCRIPT="${1:-utils/wasm_resource_guard_default_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-runtime-helpers-behavior] missing executable: $SCRIPT" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

base_build="$tmpdir/base-build"
with_verilog_build="$tmpdir/with-verilog-build"
fake_bin="$tmpdir/fake-bin"
mkdir -p "$base_build/bin" "$with_verilog_build/bin" "$fake_bin"

# Helper preflights only require .js tool files.
touch "$base_build/bin/circt-bmc.js" "$base_build/bin/circt-sim.js"
cp "$base_build/bin/circt-bmc.js" "$with_verilog_build/bin/circt-bmc.js"
cp "$base_build/bin/circt-sim.js" "$with_verilog_build/bin/circt-sim.js"
touch "$with_verilog_build/bin/circt-verilog.js"

cat >"$fake_bin/node" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
tool="${1:-}"
case "$tool" in
  *circt-bmc.js)
    echo "(check-sat)"
    ;;
  *circt-sim.js)
    echo "Simulation completed"
    ;;
  *circt-verilog.js)
    echo "hw.module @M()"
    ;;
  *)
    ;;
esac
exit 0
EOF
chmod +x "$fake_bin/node"

missing_sv="$tmpdir/missing.sv"
BMC_INPUT="${BMC_INPUT:-test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir}"
SIM_INPUT="${SIM_INPUT:-test/Tools/circt-sim/llhd-combinational.mlir}"
case1_out="$tmpdir/case1.out"
case1_err="$tmpdir/case1.err"
case2_out="$tmpdir/case2.out"
case2_err="$tmpdir/case2.err"

if [[ ! -f "$BMC_INPUT" || ! -f "$SIM_INPUT" ]]; then
  echo "[wasm-runtime-helpers-behavior] missing baseline test input(s): BMC_INPUT=$BMC_INPUT SIM_INPUT=$SIM_INPUT" >&2
  exit 1
fi

# Case 1: no circt-verilog.js in BUILD_DIR => missing SV input must not fail.
set +e
PATH="$fake_bin:$PATH" \
  BUILD_DIR="$base_build" \
  NODE_BIN=node \
  BMC_TEST_INPUT="$BMC_INPUT" \
  SIM_TEST_INPUT="$SIM_INPUT" \
  SV_TEST_INPUT="$missing_sv" \
  "$SCRIPT" >"$case1_out" 2>"$case1_err"
case1_rc=$?
set -e
if [[ "$case1_rc" -ne 0 ]]; then
  echo "[wasm-runtime-helpers-behavior] case1 failed: missing SV input should be ignored when circt-verilog.js is absent" >&2
  cat "$case1_err" >&2
  exit 1
fi

# Case 2: circt-verilog.js exists => missing SV input must fail explicitly.
set +e
PATH="$fake_bin:$PATH" \
  BUILD_DIR="$with_verilog_build" \
  NODE_BIN=node \
  BMC_TEST_INPUT="$BMC_INPUT" \
  SIM_TEST_INPUT="$SIM_INPUT" \
  SV_TEST_INPUT="$missing_sv" \
  "$SCRIPT" >"$case2_out" 2>"$case2_err"
case2_rc=$?
set -e
if [[ "$case2_rc" -eq 0 ]]; then
  echo "[wasm-runtime-helpers-behavior] case2 unexpectedly succeeded with missing SV input and circt-verilog.js present" >&2
  exit 1
fi
if ! grep -Fq -- "[wasm-rg-default] missing test input: $missing_sv" "$case2_err"; then
  echo "[wasm-runtime-helpers-behavior] case2 missing explicit missing-SV-input diagnostic" >&2
  cat "$case2_err" >&2
  exit 1
fi

echo "[wasm-runtime-helpers-behavior] PASS"
