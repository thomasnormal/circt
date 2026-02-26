#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
VERILOG_JS="${VERILOG_JS:-$BUILD_DIR/bin/circt-verilog.js}"
UVM_CORE_PATH="${UVM_CORE_PATH:-lib/Runtime/uvm-core}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-uvm-pkg-memfs] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$VERILOG_JS" ]]; then
  echo "[wasm-uvm-pkg-memfs] missing wasm tool: $VERILOG_JS" >&2
  exit 1
fi

if [[ ! -f "$UVM_CORE_PATH/src/uvm_pkg.sv" ]]; then
  echo "[wasm-uvm-pkg-memfs] missing UVM package source: $UVM_CORE_PATH/src/uvm_pkg.sv" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT
log="$tmpdir/uvm-pkg-reentry.log"

set +e
VERILOG_JS="$VERILOG_JS" UVM_CORE_PATH="$UVM_CORE_PATH" "$NODE_BIN" - <<'EOF' >"$log" 2>&1
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const util = require("node:util");
const vm = require("node:vm");

function fail(message) {
  throw new Error(message);
}

function waitUntilReady(ctx, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof ctx.callMain === "function" &&
          typeof ctx.Module._main === "function" &&
          ctx.Module.calledRun) {
        resolve();
        return;
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error("timeout waiting for wasm runtime"));
        return;
      }
      setTimeout(tick, 20);
    };
    tick();
  });
}

function writeHostSources(workDir) {
  const tbTop = [
    "import uvm_pkg::*;",
    "`include \"uvm_macros.svh\"",
    "`include \"my_test.sv\"",
    "",
    "module tb_top;",
    "  initial run_test(\"my_test\");",
    "endmodule",
    ""
  ].join("\n");

  const myTest = [
    "import uvm_pkg::*;",
    "`include \"uvm_macros.svh\"",
    "",
    "class my_test extends uvm_test;",
    "  `uvm_component_utils(my_test)",
    "  function new(string name, uvm_component parent);",
    "    super.new(name, parent);",
    "  endfunction",
    "  task run_phase(uvm_phase phase);",
    "  endtask",
    "endclass",
    ""
  ].join("\n");

  const tbTopPath = path.join(workDir, "tb_top.sv");
  const myTestPath = path.join(workDir, "my_test.sv");
  fs.writeFileSync(tbTopPath, tbTop, "utf8");
  fs.writeFileSync(myTestPath, myTest, "utf8");
  return {tbTopPath, myTestPath};
}

function runCompile(ctx, label, args, outPath) {
  let rc;
  try {
    rc = ctx.callMain(args);
  } catch (err) {
    fail(`${label} threw: ${util.inspect(err, {depth: 6})}`);
  }
  if (rc !== 0)
    fail(`${label} returned non-zero exit code: ${rc}`);

  let irText;
  try {
    irText = fs.readFileSync(outPath, "utf8");
  } catch (err) {
    fail(`${label} did not produce output file '${outPath}': ${util.inspect(err, {depth: 6})}`);
  }
  if (!irText.includes("llhd.process"))
    fail(`${label} output missing expected llhd.process content`);

  console.log(`${label} rc=${rc} out_bytes=${Buffer.byteLength(irText)}`);
}

async function main() {
  const toolJs = path.resolve(process.env.VERILOG_JS);
  const toolDir = path.dirname(toolJs);
  const source = fs.readFileSync(toolJs, "utf8");
  const uvmRoot = path.resolve(process.env.UVM_CORE_PATH);

  const ctx = {
    Module: {
      noInitialRun: true,
      locateFile: file => path.join(toolDir, file),
    },
    require,
    process,
    console,
    Buffer,
    URL,
    TextDecoder,
    TextEncoder,
    setTimeout,
    clearTimeout,
    performance,
    __dirname: toolDir,
    __filename: toolJs,
    module: {exports: {}},
    exports: {},
  };
  ctx.globalThis = ctx;
  ctx.global = ctx;

  vm.runInNewContext(source, ctx, {filename: toolJs});
  await waitUntilReady(ctx);

  const hostWork = fs.mkdtempSync(path.join(os.tmpdir(), "circt-uvm-reentry-"));
  const {tbTopPath} = writeHostSources(hostWork);
  const includeDir = path.join(uvmRoot, "src");

  const commonArgs = [
    "--resource-guard=false",
    "--ir-llhd",
    "--timescale", "1ns/1ns",
    "--uvm-path", uvmRoot,
    "-I", includeDir,
    "--top", "tb_top",
  ];

  const run1Out = path.join(hostWork, "run1.llhd.mlir");
  const run2Out = path.join(hostWork, "run2.llhd.mlir");

  runCompile(ctx, "RUN1", [...commonArgs, "-o", run1Out, tbTopPath], run1Out);
  runCompile(ctx, "RUN2", [...commonArgs, "-o", run2Out, tbTopPath], run2Out);
}

main().catch(err => {
  console.error(util.inspect(err, {depth: 8, breakLength: 120}));
  if (err && err.stack)
    console.error(err.stack);
  process.exit(1);
});
EOF
node_rc=$?
set -e
if [[ "$node_rc" -ne 0 ]]; then
  echo "[wasm-uvm-pkg-memfs] node harness failed with rc=$node_rc" >&2
  cat "$log" >&2
  exit 1
fi

clean_log="$tmpdir/uvm-pkg-reentry.clean.log"
# Strip ANSI color escapes emitted by clang diagnostics before grep checks.
sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g' "$log" >"$clean_log"

if ! grep -q '^RUN1 rc=0 out_bytes=' "$clean_log"; then
  echo "[wasm-uvm-pkg-memfs] run1 did not complete successfully" >&2
  cat "$log" >&2
  exit 1
fi

if ! grep -q '^RUN2 rc=0 out_bytes=' "$clean_log"; then
  echo "[wasm-uvm-pkg-memfs] run2 did not complete successfully" >&2
  cat "$log" >&2
  exit 1
fi

if grep -q 'Malformed attribute storage object' "$clean_log"; then
  echo "[wasm-uvm-pkg-memfs] detected malformed attribute assert during UVM compile" >&2
  cat "$log" >&2
  exit 1
fi

if grep -q 'Aborted(' "$clean_log"; then
  echo "[wasm-uvm-pkg-memfs] detected wasm abort during UVM compile" >&2
  cat "$log" >&2
  exit 1
fi

echo "[wasm-uvm-pkg-memfs] PASS"
