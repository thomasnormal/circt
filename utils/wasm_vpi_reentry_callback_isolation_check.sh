#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-sim/llhd-combinational.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-vpi-reentry] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$SIM_JS" ]]; then
  echo "[wasm-vpi-reentry] missing tool: $SIM_JS" >&2
  exit 1
fi
if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-vpi-reentry] missing input: $INPUT_MLIR" >&2
  exit 1
fi

SIM_JS="$SIM_JS" INPUT_MLIR="$INPUT_MLIR" "$NODE_BIN" - <<'EOF_NODE'
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const VPI = {
  cbStartOfSimulation: 11,
};

function fail(message) {
  throw new Error(message);
}

function waitUntilReady(ctx, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof ctx.callMain === "function" &&
          typeof ctx.Module.callMain !== "function")
        ctx.Module.callMain = ctx.callMain;
      if (typeof ctx.Module.callMain === "function" &&
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

function allocCb(ctx, reason, cbRtnPtr, timePtr = 0) {
  const ptr = ctx._malloc(28);
  ctx.HEAP32[(ptr + 0) >> 2] = reason;
  ctx.HEAPU32[(ptr + 4) >> 2] = cbRtnPtr >>> 0;
  ctx.HEAPU32[(ptr + 8) >> 2] = 0;
  ctx.HEAPU32[(ptr + 12) >> 2] = timePtr >>> 0;
  ctx.HEAPU32[(ptr + 16) >> 2] = 0;
  ctx.HEAP32[(ptr + 20) >> 2] = 0;
  ctx.HEAPU32[(ptr + 24) >> 2] = 0;
  return ptr;
}

async function settleAsyncifyRewind(iterations = 200) {
  for (let i = 0; i < iterations; ++i)
    await new Promise(resolve => setTimeout(resolve, 0));
}

async function main() {
  const toolJs = path.resolve(process.env.SIM_JS);
  const toolDir = path.dirname(toolJs);
  const source = fs.readFileSync(toolJs, "utf8");
  const input = path.resolve(process.env.INPUT_MLIR);

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

  const startCountByRun = [0, 0];
  let currentRun = -1;

  ctx.circtSimVpiYieldHook = (_cbFuncPtr, cbDataPtr) => {
    const reason = ctx.HEAP32[(cbDataPtr + 0) >> 2];
    if (reason === VPI.cbStartOfSimulation && currentRun >= 0)
      ++startCountByRun[currentRun];
  };

  // Enable VPI runtime without requiring vlog_startup_routines.
  ctx._vpi_startup_register(0);

  // Register a single cbStart callback before run1 only.
  const cbStart = allocCb(ctx, VPI.cbStartOfSimulation, 0, 0);
  const handle = ctx._vpi_register_cb(cbStart);
  if (!handle)
    fail("failed to pre-register cbStartOfSimulation callback");

  currentRun = 0;
  let rc = ctx.Module.callMain(["--resource-guard=false", input]);
  if (rc && typeof rc.then === "function")
    rc = await rc;
  await settleAsyncifyRewind();
  if (rc !== 0)
    fail(`run1 failed: rc=${rc}`);

  currentRun = 1;
  rc = ctx.Module.callMain(["--resource-guard=false", input]);
  if (rc && typeof rc.then === "function")
    rc = await rc;
  await settleAsyncifyRewind();
  if (rc !== 0)
    fail(`run2 failed: rc=${rc}`);

  if (startCountByRun[0] !== 1) {
    fail(`expected run1 to fire exactly one cbStart callback, saw ${startCountByRun[0]}`);
  }
  if (startCountByRun[1] !== 0) {
    fail(
      "callback leakage across callMain re-entry: " +
      `run2 saw ${startCountByRun[1]} stale cbStart callback(s)`
    );
  }

  console.log(
      `WASM_VPI_REENTRY_ISOLATION PASS run1=${startCountByRun[0]} run2=${startCountByRun[1]}`);
}

main().catch(err => {
  console.error(err.stack || String(err));
  process.exit(1);
});
EOF_NODE
