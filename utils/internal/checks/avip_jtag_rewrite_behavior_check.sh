#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_avip_circt_verilog.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[avip-jtag-rewrite-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

avip="$tmpdir/jtag_avip"
mkdir -p "$avip/sim" "$avip/src/hdlTop/jtagControllerDeviceAgentBfm" \
         "$avip/src/hdlTop/jtagTargetDeviceAgentBfm" "$tmpdir/uvm"

cat >"$avip/sim/JtagCompile.f" <<'EOF'
../src/hdlTop/jtagControllerDeviceAgentBfm/JtagControllerDeviceAgentBfm.sv
../src/hdlTop/jtagTargetDeviceAgentBfm/JtagTargetDeviceDriverBfm.sv
../src/hdlTop/jtagTargetDeviceAgentBfm/JtagTargetDeviceAgentBfm.sv
EOF

cat >"$avip/src/hdlTop/jtagControllerDeviceAgentBfm/JtagControllerDeviceAgentBfm.sv" <<'EOF'
module JtagControllerDeviceAgentBfm;
  bind jtagControllerDeviceMonitorBfm JtagControllerDeviceAssertions TestVectrorTestingAssertions(.clk(clk),.Tdi(Tdi), .reset(reset),.Tms(Tms));
endmodule
EOF

cat >"$avip/src/hdlTop/jtagTargetDeviceAgentBfm/JtagTargetDeviceDriverBfm.sv" <<'EOF'
typedef enum logic [4:0] {A=5'b0} JtagInstructionOpcodeEnum;
logic [7:0] registerBank [JtagInstructionOpcodeEnum];
logic [4:0] instructionRegister;
module JtagTargetDeviceDriverBfm;
  initial begin
    registerBank[instructionRegister] = registerBank[instructionRegister];
  end
endmodule
EOF

cat >"$avip/src/hdlTop/jtagTargetDeviceAgentBfm/JtagTargetDeviceAgentBfm.sv" <<'EOF'
module JtagTargetDeviceAgentBfm;
  bind JtagTargetDeviceMonitorBfm JtagTargetDeviceAssertions TestVectrorTestingAssertions(.clk(jtagIf.clk),.Tdo(jtagIf.Tdo),.Tms(jtagIf.Tms),.reset(jtagIf.reset));
endmodule
EOF

cat >"$tmpdir/uvm/uvm_pkg.sv" <<'EOF'
package uvm_pkg;
endpackage
EOF

mock_circt_verilog="$tmpdir/mock-circt-verilog.sh"
cat >"$mock_circt_verilog" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

have_driver=0
have_ctrl=0
have_agent=0

for arg in "$@"; do
  case "$arg" in
    *.sv)
      base="$(basename "$arg")"
      if [[ "$base" == "JtagTargetDeviceDriverBfm.sv" ]]; then
        have_driver=1
        if grep -q 'registerBank\[instructionRegister\]' "$arg"; then
          echo "driver rewrite missing enum cast" >&2
          exit 2
        fi
        if ! grep -q "registerBank\\[JtagInstructionOpcodeEnum'(instructionRegister)\\]" "$arg"; then
          echo "driver rewrite missing explicit casted index" >&2
          exit 2
        fi
      elif [[ "$base" == "JtagControllerDeviceAgentBfm.sv" ]]; then
        have_ctrl=1
        if grep -q '^  bind ' "$arg"; then
          echo "controller bind line not removed" >&2
          exit 2
        fi
      elif [[ "$base" == "JtagTargetDeviceAgentBfm.sv" ]]; then
        have_agent=1
        if grep -q '^  bind ' "$arg"; then
          echo "target bind line not removed" >&2
          exit 2
        fi
      fi
      ;;
  esac
done

if [[ "$have_driver" -ne 1 || "$have_ctrl" -ne 1 || "$have_agent" -ne 1 ]]; then
  echo "expected jtag files not seen by mock circt-verilog" >&2
  exit 2
fi

cat <<'MLIR'
module {
}
MLIR
EOF
chmod +x "$mock_circt_verilog"

echo "[avip-jtag-rewrite-check] case: jtag-source-rewrites-applied-before-compile"
out="$tmpdir/out.mlir"
set +e
CIRCT_VERILOG="$mock_circt_verilog" \
UVM_DIR="$tmpdir/uvm" \
OUT="$out" \
DISABLE_UVM_AUTO_INCLUDE=1 \
  "$RUNNER" "$avip" "$avip/sim/JtagCompile.f" >/tmp/avip-jtag-rewrite.out 2>&1
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  cat /tmp/avip-jtag-rewrite.out >&2
  echo "[avip-jtag-rewrite-check] jtag rewrite check failed" >&2
  exit 1
fi

echo "[avip-jtag-rewrite-check] PASS"
