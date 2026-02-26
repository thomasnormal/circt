#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_avip_circt_verilog.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[avip-circt-verilog-sanitize-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/avip/sim" "$tmpdir/avip/src" "$tmpdir/uvm"
cat >"$tmpdir/avip/sim/compile.f" <<'EOF'
../src/dummy.sv
EOF
cat >"$tmpdir/avip/src/dummy.sv" <<'EOF'
module dummy;
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
cat <<'IR'
diagnostic banner before module
module {
  func.func @f() {
    return
  }
}
TAIL_GARBAGE_MARKER
  return
}
IR
EOF
chmod +x "$mock_circt_verilog"

out="$tmpdir/out.mlir"

echo "[avip-circt-verilog-sanitize-check] case: trailing-garbage-removed"
CIRCT_VERILOG="$mock_circt_verilog" \
UVM_DIR="$tmpdir/uvm" \
OUT="$out" \
DISABLE_UVM_AUTO_INCLUDE=1 \
  "$RUNNER" "$tmpdir/avip" "$tmpdir/avip/sim/compile.f" >/dev/null

if grep -q 'TAIL_GARBAGE_MARKER' "$out"; then
  echo "[avip-circt-verilog-sanitize-check] trailing garbage marker still present" >&2
  cat "$out" >&2
  exit 1
fi
if ! grep -q '^module' "$out"; then
  echo "[avip-circt-verilog-sanitize-check] missing module header after sanitize" >&2
  cat "$out" >&2
  exit 1
fi
if [[ "$(tail -n 1 "$out")" != "}" ]]; then
  echo "[avip-circt-verilog-sanitize-check] expected sanitized output to end with top-level }" >&2
  cat "$out" >&2
  exit 1
fi

echo "[avip-circt-verilog-sanitize-check] PASS"
