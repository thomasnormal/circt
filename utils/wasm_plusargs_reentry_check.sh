#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-sim/wasm-plusargs-reentry.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-plusargs] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$SIM_JS" ]]; then
  echo "[wasm-plusargs] missing tool: $SIM_JS" >&2
  exit 1
fi
if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-plusargs] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
log="$tmpdir/plusargs-reentry.log"

SIM_JS="$SIM_JS" INPUT_MLIR="$INPUT_MLIR" "$NODE_BIN" - <<'EOF' >"$log" 2>&1
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const toolJs = path.resolve(process.env.SIM_JS);
const toolDir = path.dirname(toolJs);
const source = fs.readFileSync(toolJs, "utf8");
const input = path.resolve(process.env.INPUT_MLIR);

const ctx = {
  Module: {noInitialRun: true, locateFile: file => path.join(toolDir, file)},
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

function waitUntilReady(timeoutMs = 20000) {
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

(async () => {
  await waitUntilReady();
  const rc1 = ctx.callMain(["--resource-guard=false", "+VERBOSE", "+DEBUG=3", input]);
  console.log("--- RUN2 ---");
  const rc2 = ctx.callMain(["--resource-guard=false", input]);
  console.log("rc1", rc1, "rc2", rc2);
  if (rc1 !== 0 || rc2 !== 0)
    process.exitCode = 1;
})().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
EOF

if ! awk 'BEGIN{s=1} /--- RUN2 ---/{s=2;next} s==1{print}' "$log" | grep -q "verbose_found"; then
  echo "[wasm-plusargs] run1 missing verbose_found" >&2
  cat "$log" >&2
  exit 1
fi
if ! awk 'BEGIN{s=1} /--- RUN2 ---/{s=2;next} s==1{print}' "$log" | grep -q "debug_found"; then
  echo "[wasm-plusargs] run1 missing debug_found" >&2
  cat "$log" >&2
  exit 1
fi
if ! awk 'BEGIN{s=1} /--- RUN2 ---/{s=2;next} s==2{print}' "$log" | grep -q "verbose_not_found"; then
  echo "[wasm-plusargs] run2 leaked VERBOSE plusarg" >&2
  cat "$log" >&2
  exit 1
fi
if ! awk 'BEGIN{s=1} /--- RUN2 ---/{s=2;next} s==2{print}' "$log" | grep -q "debug_not_found"; then
  echo "[wasm-plusargs] run2 leaked DEBUG plusarg" >&2
  cat "$log" >&2
  exit 1
fi

echo "[wasm-plusargs] PASS"
