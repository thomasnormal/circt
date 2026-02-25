#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
runner="$repo_root/utils/run_avip_circt_verilog.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

avip="$tmp_dir/axi4Lite_avip"
mkdir -p \
  "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterWriteVIP/sim" \
  "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterWriteVIP/src/masterWriteEnv" \
  "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterReadVIP/sim" \
  "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterReadVIP/src/masterReadEnv" \
  "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveWriteVIP/sim" \
  "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveWriteVIP/src/slaveWriteEnv" \
  "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveReadVIP/sim" \
  "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveReadVIP/src/slaveReadEnv" \
  "$tmp_dir/uvm"

cat > "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterWriteVIP/sim/Axi4LiteWriteMaster.f" <<'EOF_F'
${AXI4LITE_MASTERWRITE}/src/masterWriteEnv/KeepViaFilelistOnly.sv
EOF_F
cat > "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterReadVIP/sim/Axi4LiteReadMaster.f" <<'EOF_F'
${AXI4LITE_MASTERREAD}/src/masterReadEnv/KeepViaFilelistOnly.sv
EOF_F
cat > "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveWriteVIP/sim/Axi4LiteWriteSlave.f" <<'EOF_F'
${AXI4LITE_SLAVEWRITE}/src/slaveWriteEnv/KeepViaFilelistOnly.sv
EOF_F
cat > "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveReadVIP/sim/Axi4LiteReadSlave.f" <<'EOF_F'
${AXI4LITE_SLAVEREAD}/src/slaveReadEnv/KeepViaFilelistOnly.sv
EOF_F

cat > "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterWriteVIP/src/masterWriteEnv/KeepViaFilelistOnly.sv" <<'EOF_SV'
module keep_master_write;
endmodule
EOF_SV
cat > "$avip/src/axi4LiteMasterVIP/src/axi4LiteMasterReadVIP/src/masterReadEnv/KeepViaFilelistOnly.sv" <<'EOF_SV'
module keep_master_read;
endmodule
EOF_SV
cat > "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveWriteVIP/src/slaveWriteEnv/KeepViaFilelistOnly.sv" <<'EOF_SV'
module keep_slave_write;
endmodule
EOF_SV
cat > "$avip/src/axi4LiteSlaveVIP/src/axi4LiteSlaveReadVIP/src/slaveReadEnv/KeepViaFilelistOnly.sv" <<'EOF_SV'
module keep_slave_read;
endmodule
EOF_SV

cat > "$tmp_dir/uvm/uvm_pkg.sv" <<'EOF_UVM'
package uvm_pkg;
endpackage
EOF_UVM

fake_tool="$tmp_dir/fake-circt-verilog"
cat > "$fake_tool" <<'EOF_TOOL'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@"
EOF_TOOL
chmod +x "$fake_tool"

out_mlir="$tmp_dir/out.mlir"
run_log="$tmp_dir/run.log"
if ! CIRCT_VERILOG="$fake_tool" \
  UVM_DIR="$tmp_dir/uvm" \
  OUT="$out_mlir" \
  "$runner" "$avip" >"$run_log" 2>&1; then
  echo "runner failed unexpectedly" >&2
  cat "$run_log" >&2
  exit 1
fi

if ! grep -q 'KeepViaFilelistOnly.sv' "$out_mlir"; then
  echo "expected AXI4Lite nested filelist entries in compiler args" >&2
  cat "$run_log" >&2
  exit 1
fi

if grep -q 'synthesized AVIP filelist' "$run_log"; then
  echo "did not expect synthesized fallback for AXI4Lite nested filelists" >&2
  cat "$run_log" >&2
  exit 1
fi

echo "PASS: AXI4Lite nested filelist fallback is used when root sim filelist is missing"
