#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_avip_circt_sim.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[avip-circt-sim-snapshot-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mbit="$tmpdir/mbit"
mkdir -p "$mbit/apb_avip/sim"
cat >"$mbit/apb_avip/sim/apb_compile.f" <<'EOF'
dummy.sv
EOF

mock_run_avip="$tmpdir/mock-run-avip.sh"
cat >"$mock_run_avip" <<'EOF'
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
EOF
chmod +x "$mock_run_avip"

mock_circt_verilog="$tmpdir/mock-circt-verilog.sh"
cat >"$mock_circt_verilog" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "unexpected direct circt-verilog invocation" >&2
exit 99
EOF
chmod +x "$mock_circt_verilog"

mock_circt_sim="$tmpdir/mock-circt-sim.sh"
cat >"$mock_circt_sim" <<'EOF'
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
EOF
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
cat >"$bad_circt_sim" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "broken sim" >&2
  exit 134
fi
echo "bad sim should not be executed" >&2
exit 99
EOF
chmod +x "$bad_circt_sim"

good_fallback_sim="$tmpdir/good-fallback-circt-sim.sh"
cat >"$good_fallback_sim" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${MOCK_SIM_INVOKE_LOG:?}"
echo "GOOD_FALLBACK $0 $*" >> "$MOCK_SIM_INVOKE_LOG"
cat <<'SIMLOG'
Simulation terminated at time 321 fs
UVM_FATAL : 0
UVM_ERROR : 0
Coverage = 0 %
Coverage = 0 %
SIMLOG
exit 0
EOF
chmod +x "$good_fallback_sim"

invoke_log_fallback="$tmpdir/sim-invocations-fallback.log"

echo "[avip-circt-sim-snapshot-check] case: unhealthy-primary-sim-falls-back"
MOCK_SIM_INVOKE_LOG="$invoke_log_fallback" \
MBIT_DIR="$mbit" \
AVIPS=apb \
SEEDS=1 \
RUN_AVIP="$mock_run_avip" \
CIRCT_VERILOG="$mock_circt_verilog" \
CIRCT_SIM="$bad_circt_sim" \
CIRCT_SIM_FALLBACK="$good_fallback_sim" \
COMPILE_TIMEOUT=10 \
SIM_TIMEOUT=10 \
SIM_TIMEOUT_GRACE=2 \
MAX_WALL_MS=12000 \
  "$RUNNER" "$tmpdir/out-fallback" >/dev/null

if ! grep -q 'GOOD_FALLBACK ' "$invoke_log_fallback"; then
  echo "[avip-circt-sim-snapshot-check] expected fallback sim invocation not found" >&2
  cat "$invoke_log_fallback" >&2 || true
  exit 1
fi

echo "[avip-circt-sim-snapshot-check] case: unhealthy-primary-without-fallback-fails"
set +e
MBIT_DIR="$mbit" \
AVIPS=apb \
SEEDS=1 \
RUN_AVIP="$mock_run_avip" \
CIRCT_VERILOG="$mock_circt_verilog" \
CIRCT_SIM="$bad_circt_sim" \
COMPILE_TIMEOUT=10 \
SIM_TIMEOUT=10 \
SIM_TIMEOUT_GRACE=2 \
MAX_WALL_MS=12000 \
  "$RUNNER" "$tmpdir/out-nofallback" >/dev/null 2>"$tmpdir/nofallback.err"
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[avip-circt-sim-snapshot-check] expected failure without fallback, but run succeeded" >&2
  cat "$tmpdir/nofallback.err" >&2 || true
  exit 1
fi
if ! grep -q 'circt-sim probe failed' "$tmpdir/nofallback.err"; then
  echo "[avip-circt-sim-snapshot-check] expected probe-failed error without fallback" >&2
  cat "$tmpdir/nofallback.err" >&2 || true
  exit 1
fi

echo "[avip-circt-sim-snapshot-check] PASS"
