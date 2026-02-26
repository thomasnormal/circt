#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"

runner="$repo_root/utils/run_avip_circt_sim.sh"
if [[ ! -x "$runner" ]]; then
  echo "[avip-circt-sim-parse-check] missing runner: $runner" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Minimal MBIT fixture for one AVIP.
mbit="$tmpdir/mbit"
mkdir -p "$mbit/apb_avip/sim"
cat >"$mbit/apb_avip/sim/apb_compile.f" <<'EOF'
dummy.sv
EOF

mock_run_avip="$tmpdir/mock-run-avip.sh"
cat >"$mock_run_avip" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?}"
cat >"$OUT" <<'MLIR'
module {
}
MLIR
EOF
chmod +x "$mock_run_avip"

mock_circt_sim="$tmpdir/mock-circt-sim.sh"
cat >"$mock_circt_sim" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  exit 0
fi
cat <<'LOG'
[circt-sim] Simulation completed at time 123 fs

=================================================
          Coverage Report
=================================================

Covergroup: cg1
  Overall coverage: 12.50%

Covergroup: cg2
  Overall coverage: 88.00%
LOG
exit 0
EOF
chmod +x "$mock_circt_sim"

out_dir="$tmpdir/out"
MBIT_DIR="$mbit" \
AVIPS=apb \
SEEDS=1 \
SIM_TIMEOUT=10 \
SIM_TIMEOUT_GRACE=5 \
MAX_WALL_MS=15000 \
RUN_AVIP="$mock_run_avip" \
CIRCT_VERILOG=/bin/true \
CIRCT_SIM="$mock_circt_sim" \
CIRCT_ALLOW_NONCANONICAL_TOOLS=1 \
  "$runner" "$out_dir" >/dev/null

matrix="$out_dir/matrix.tsv"
if [[ ! -f "$matrix" ]]; then
  echo "[avip-circt-sim-parse-check] missing matrix: $matrix" >&2
  exit 1
fi

# Expect sim_time_fs and two coverage values to be parsed.
row="$(tail -n +2 "$matrix" | head -n 1)"
if [[ -z "$row" ]]; then
  echo "[avip-circt-sim-parse-check] missing matrix row" >&2
  cat "$matrix" >&2
  exit 1
fi

sim_time_fs="$(cut -f8 <<<"$row")"
cov1="$(cut -f11 <<<"$row")"
cov2="$(cut -f12 <<<"$row")"

if [[ "$sim_time_fs" != "123" ]]; then
  echo "[avip-circt-sim-parse-check] expected sim_time_fs=123, got '$sim_time_fs'" >&2
  cat "$matrix" >&2
  exit 1
fi
if [[ "$cov1" != "12.50" || "$cov2" != "88.00" ]]; then
  echo "[avip-circt-sim-parse-check] expected cov_1=12.50 cov_2=88.00, got '$cov1' '$cov2'" >&2
  cat "$matrix" >&2
  exit 1
fi

echo "[avip-circt-sim-parse-check] PASS"
