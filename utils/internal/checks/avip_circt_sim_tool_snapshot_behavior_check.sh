#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_avip_circt_sim.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[avip-circt-sim-snapshot-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mbit="$tmpdir/mbit"
mkdir -p "$mbit/apb_avip/sim"
cat >"$mbit/apb_avip/sim/apb_compile.f" <<'EOL'
dummy.sv
EOL

mock_run_avip="$tmpdir/mock-run-avip.sh"
cat >"$mock_run_avip" <<'EOL'
#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${OUT:-}" ]]; then
  echo "OUT not set" >&2
  exit 2
fi
cat >"$OUT" <<'MLIR'
module {
}
MLIR
echo "0"
echo "$OUT"
EOL
chmod +x "$mock_run_avip"

mock_circt_verilog="$tmpdir/mock-circt-verilog.sh"
cat >"$mock_circt_verilog" <<'EOL'
#!/usr/bin/env bash
set -euo pipefail
echo "unexpected direct circt-verilog invocation" >&2
exit 99
EOL
chmod +x "$mock_circt_verilog"

mock_circt_sim="$tmpdir/mock-circt-sim.sh"
cat >"$mock_circt_sim" <<'EOL'
#!/usr/bin/env bash
set -euo pipefail
: "${MOCK_SIM_INVOKE_LOG:?}"
echo "$0 $*" >> "$MOCK_SIM_INVOKE_LOG"
cat <<'SIMLOG'
Simulation terminated at time 123 fs
UVM_FATAL : 0
UVM_ERROR : 0
Coverage = 0 %
Coverage = 0 %
SIMLOG
exit 0
EOL
chmod +x "$mock_circt_sim"

out_dir="$tmpdir/out"
invoke_log="$tmpdir/sim-invocations.log"

echo "[avip-circt-sim-snapshot-check] case: sim-tool-invoked-from-snapshot-copy"
MOCK_SIM_INVOKE_LOG="$invoke_log" \
MBIT_DIR="$mbit" \
AVIPS=apb \
SEEDS=1 \
RUN_AVIP="$mock_run_avip" \
CIRCT_VERILOG="$mock_circt_verilog" \
CIRCT_SIM="$mock_circt_sim" \
CIRCT_ALLOW_NONCANONICAL_TOOLS=1 \
COMPILE_TIMEOUT=10 \
SIM_TIMEOUT=10 \
SIM_TIMEOUT_GRACE=2 \
MAX_WALL_MS=12000 \
  "$RUNNER" "$out_dir" >/dev/null

if [[ ! -s "$invoke_log" ]]; then
  echo "[avip-circt-sim-snapshot-check] missing sim invocation log" >&2
  exit 1
fi

if ! grep -q "${out_dir}/.tool-snapshot/circt-sim" "$invoke_log"; then
  echo "[avip-circt-sim-snapshot-check] expected circt-sim snapshot invocation not found" >&2
  cat "$invoke_log" >&2
  exit 1
fi

bad_circt_sim="$tmpdir/bad-circt-sim.sh"
cat >"$bad_circt_sim" <<'EOL'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "broken sim" >&2
  exit 134
fi
echo "bad sim should not be executed" >&2
exit 99
EOL
chmod +x "$bad_circt_sim"

echo "[avip-circt-sim-snapshot-check] case: unhealthy-primary-fails"
set +e
MBIT_DIR="$mbit" \
AVIPS=apb \
SEEDS=1 \
RUN_AVIP="$mock_run_avip" \
CIRCT_VERILOG="$mock_circt_verilog" \
CIRCT_SIM="$bad_circt_sim" \
CIRCT_ALLOW_NONCANONICAL_TOOLS=1 \
COMPILE_TIMEOUT=10 \
SIM_TIMEOUT=10 \
SIM_TIMEOUT_GRACE=2 \
MAX_WALL_MS=12000 \
  "$RUNNER" "$tmpdir/out-fail" >/dev/null 2>"$tmpdir/fail.err"
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[avip-circt-sim-snapshot-check] expected probe failure, but run succeeded" >&2
  cat "$tmpdir/fail.err" >&2 || true
  exit 1
fi
if ! grep -q 'circt-sim probe failed' "$tmpdir/fail.err"; then
  echo "[avip-circt-sim-snapshot-check] expected probe-failed error" >&2
  cat "$tmpdir/fail.err" >&2 || true
  exit 1
fi

echo "[avip-circt-sim-snapshot-check] PASS"
