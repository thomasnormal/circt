#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
runner="$repo_root/utils/run_avip_circt_sim.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$tmp_dir/mbit/apb_avip" "$tmp_dir/out"

fake_compile="$tmp_dir/fake-compile.sh"
cat >"$fake_compile" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cat > "$OUT" <<'IR'
module {
}
IR
EOF
chmod +x "$fake_compile"

fake_sim="$tmp_dir/fake-circt-sim.sh"
sim_count_file="$tmp_dir/sim.count"
cat >"$fake_sim" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "--help" ]]; then
  exit 0
fi
count=0
if [[ -f "$sim_count_file" ]]; then
  count="\$(cat "$sim_count_file")"
fi
count=\$((count + 1))
echo "\$count" > "$sim_count_file"
if [[ "\$count" -eq 1 ]]; then
  echo "UVM_FATAL @ 0: FCTTYP Factory did not return a component of type 'AhbSlaveSequencer'."
  exit 1
fi
echo "[circt-sim] Simulation completed at time 10 fs"
echo "[circt-sim] Simulation completed"
exit 0
EOF
chmod +x "$fake_sim"

out_dir="$tmp_dir/out/run"
MBIT_DIR="$tmp_dir/mbit" \
AVIPS=apb \
SEEDS=1 \
SIM_RETRIES=1 \
RUN_AVIP="$fake_compile" \
CIRCT_VERILOG=/bin/true \
CIRCT_SIM="$fake_sim" \
CIRCT_ALLOW_NONCANONICAL_TOOLS=1 \
"$runner" "$out_dir" >/dev/null 2>&1

matrix="$out_dir/matrix.tsv"
if [[ ! -f "$matrix" ]]; then
  echo "missing matrix output: $matrix" >&2
  exit 1
fi

sim_status="$(awk -F'\t' 'NR==2 {print $5}' "$matrix")"
sim_exit="$(awk -F'\t' 'NR==2 {print $6}' "$matrix")"
sim_log="$(awk -F'\t' 'NR==2 {print $15}' "$matrix")"
if [[ "$sim_status" != "OK" || "$sim_exit" != "0" ]]; then
  echo "expected retry to recover FCTTYP flake (status=$sim_status exit=$sim_exit)" >&2
  exit 1
fi

if [[ ! -f "$sim_log" ]]; then
  echo "missing sim log path from matrix: $sim_log" >&2
  exit 1
fi

if ! rg -q "Simulation completed" "$sim_log"; then
  echo "expected successful final attempt log" >&2
  exit 1
fi

attempt1="$sim_log.attempt1.log"
if [[ ! -f "$attempt1" ]]; then
  echo "expected first failed attempt to be preserved: $attempt1" >&2
  exit 1
fi

if ! rg -q "FCTTYP" "$attempt1"; then
  echo "expected first attempt log to capture FCTTYP failure" >&2
  exit 1
fi

if [[ "$(cat "$sim_count_file")" != "2" ]]; then
  echo "expected exactly one retry (2 sim invocations)" >&2
  exit 1
fi

echo "PASS: run_avip_circt_sim retries FCTTYP flakes and records attempts"
