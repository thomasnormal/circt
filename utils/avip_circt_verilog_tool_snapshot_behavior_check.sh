#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
runner="$repo_root/utils/run_avip_circt_verilog.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

fake_tool="$tmp_dir/fake-circt-verilog"
invoked_path_file="$tmp_dir/invoked.path"
cat >"$fake_tool" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "\$0" > "$invoked_path_file"
echo "module {"
echo "}"
EOF
chmod 0644 "$fake_tool"

mkdir -p "$tmp_dir/avip/sim" "$tmp_dir/avip/src" "$tmp_dir/uvm" "$tmp_dir/out"
cat >"$tmp_dir/avip/src/top.sv" <<'EOF'
module top;
endmodule
EOF
cat >"$tmp_dir/avip/sim/compile.f" <<'EOF'
../src/top.sv
EOF
cat >"$tmp_dir/uvm/uvm_pkg.sv" <<'EOF'
package uvm_pkg;
endpackage
EOF

out_mlir="$tmp_dir/out/out.mlir"
CIRCT_VERILOG="$fake_tool" \
UVM_DIR="$tmp_dir/uvm" \
OUT="$out_mlir" \
"$runner" "$tmp_dir/avip" "$tmp_dir/avip/sim/compile.f" >/dev/null 2>&1

if [[ ! -f "$out_mlir" ]]; then
  echo "expected OUT file to be produced" >&2
  exit 1
fi

if ! grep -q '^module' "$out_mlir"; then
  echo "expected fake tool output in OUT file" >&2
  exit 1
fi

if [[ ! -f "$invoked_path_file" ]]; then
  echo "expected fake tool to write invocation path" >&2
  exit 1
fi

invoked_path="$(cat "$invoked_path_file")"
if [[ "$invoked_path" == "$fake_tool" ]]; then
  echo "expected runner to invoke a stable snapshot, not the mutable source path" >&2
  exit 1
fi

if [[ ! -x "$invoked_path" ]]; then
  echo "expected invoked snapshot tool to be executable: $invoked_path" >&2
  exit 1
fi

if [[ -x "$fake_tool" ]]; then
  echo "expected source tool permissions to remain unchanged" >&2
  exit 1
fi

echo "PASS: run_avip_circt_verilog snapshots tool and runs executable snapshot"
