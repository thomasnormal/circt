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
trap 'rm -rf "$tmpdir"' EXIT
log="$tmpdir/uvm-pkg-memfs-reentry.log"

VERILOG_JS="$VERILOG_JS" UVM_CORE_PATH="$UVM_CORE_PATH" "$NODE_BIN" - <<'EOF' >"$log" 2>&1
const fs = require("node:fs");
const path = require("node:path");
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

function mkdirpInWasm(ctx, wasmPath) {
  const parts = wasmPath.split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur += `/${part}`;
    try {
      ctx.FS.mkdir(cur);
    } catch (_) {
      // already exists
    }
  }
}

function copyTreeToWasm(ctx, hostRoot, wasmRoot) {
  mkdirpInWasm(ctx, wasmRoot);
  for (const ent of fs.readdirSync(hostRoot, {withFileTypes: true})) {
    const hostPath = path.join(hostRoot, ent.name);
    const wasmPath = `${wasmRoot}/${ent.name}`;
    if (ent.isDirectory()) {
      copyTreeToWasm(ctx, hostPath, wasmPath);
      continue;
    }
    if (!ent.isFile())
      continue;
    const data = fs.readFileSync(hostPath);
    ctx.FS_createDataFile(wasmRoot, ent.name, data, true, true, true);
  }
}

function writeInlineSources(ctx) {
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

  mkdirpInWasm(ctx, "/workspace/src");
  mkdirpInWasm(ctx, "/workspace/out");
  ctx.FS_createDataFile("/workspace/src", "tb_top.sv", Buffer.from(tbTop), true, true, true);
  ctx.FS_createDataFile("/workspace/src", "my_test.sv", Buffer.from(myTest), true, true, true);
}

function runCompile(ctx, label, outPath) {
  const args = [
    "--resource-guard=false",
    "--ir-llhd",
    "--timescale", "1ns/1ns",
    "--uvm-path", "/circt/uvm-core",
    "-I", "/circt/uvm-core/src",
    "--top", "tb_top",
    "-o", outPath,
    "/workspace/src/tb_top.sv",
  ];

  let rc;
  try {
    rc = ctx.callMain(args);
  } catch (err) {
    fail(`${label} threw: ${err}`);
  }
  if (rc !== 0)
    fail(`${label} returned non-zero exit code: ${rc}`);

  let irText;
  try {
    irText = ctx.FS.readFile(outPath, {encoding: "utf8"});
  } catch (err) {
    fail(`${label} did not produce output file '${outPath}': ${err}`);
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

  copyTreeToWasm(ctx, uvmRoot, "/circt/uvm-core");
  writeInlineSources(ctx);

  runCompile(ctx, "RUN1", "/workspace/out/run1.llhd.mlir");
  runCompile(ctx, "RUN2", "/workspace/out/run2.llhd.mlir");
}

main().catch(err => {
  console.error(err.stack || String(err));
  process.exit(1);
});
EOF

if ! grep -q '^RUN1 rc=0 out_bytes=' "$log"; then
  echo "[wasm-uvm-pkg-memfs] run1 did not complete successfully" >&2
  cat "$log" >&2
  exit 1
fi

if ! grep -q '^RUN2 rc=0 out_bytes=' "$log"; then
  echo "[wasm-uvm-pkg-memfs] run2 did not complete successfully" >&2
  cat "$log" >&2
  exit 1
fi

if grep -q 'Malformed attribute storage object' "$log"; then
  echo "[wasm-uvm-pkg-memfs] detected malformed attribute assert during UVM compile" >&2
  cat "$log" >&2
  exit 1
fi

if grep -q 'Aborted(' "$log"; then
  echo "[wasm-uvm-pkg-memfs] detected wasm abort during UVM compile" >&2
  cat "$log" >&2
  exit 1
fi

echo "[wasm-uvm-pkg-memfs] PASS"
