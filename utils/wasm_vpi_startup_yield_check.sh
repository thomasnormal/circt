#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
SIM_JS="${SIM_JS:-$BUILD_DIR/bin/circt-sim.js}"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-sim/llhd-combinational.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-vpi-startup-yield] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$SIM_JS" ]]; then
  echo "[wasm-vpi-startup-yield] missing wasm tool: $SIM_JS" >&2
  exit 1
fi

if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-vpi-startup-yield] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
log="$tmpdir/wasm-vpi-startup-yield.log"

SIM_JS="$SIM_JS" INPUT_MLIR="$INPUT_MLIR" "$NODE_BIN" - <<'EOF' >"$log" 2>&1
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const VPI = {
  cbAfterDelay: 9,
  cbStartOfSimulation: 11,
  vpiSimTime: 2,
};

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

function allocSimTimePs(ctx, ps) {
  const ptr = ctx._malloc(24);
  ctx.HEAP32[(ptr + 0) >> 2] = VPI.vpiSimTime;
  ctx.HEAPU32[(ptr + 4) >> 2] = 0;
  ctx.HEAPU32[(ptr + 8) >> 2] = ps >>> 0;
  return ptr;
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

  let startCount = 0;
  let afterDelayCount = 0;
  let afterDelayRegAttempted = false;
  let afterDelayHandle = 0;

  ctx.circtSimVpiYieldHook = async (_cbFuncPtr, cbDataPtr) => {
    const reason = ctx.HEAP32[(cbDataPtr + 0) >> 2];
    if (reason === VPI.cbStartOfSimulation) {
      ++startCount;
      await new Promise(resolve => setTimeout(resolve, 0));
      afterDelayRegAttempted = true;
      const t1ps = allocSimTimePs(ctx, 1);
      const cbAfterDelay = allocCb(ctx, VPI.cbAfterDelay, 0, t1ps);
      afterDelayHandle = ctx._vpi_register_cb(cbAfterDelay);
      return;
    }
    if (reason === VPI.cbAfterDelay)
      ++afterDelayCount;
  };

  // Enable VPI runtime without needing a native vlog_startup_routines path.
  ctx._vpi_startup_register(0);

  // Register cbStartOfSimulation before callMain with cb_rtn=0 (hook-only mode).
  const cbStart = allocCb(ctx, VPI.cbStartOfSimulation, 0, 0);
  const startHandle = ctx._vpi_register_cb(cbStart);
  if (!startHandle)
    fail("vpi_register_cb returned null for cbStartOfSimulation with cb_rtn=0");

  let rc = ctx.callMain(["--resource-guard=false", input]);
  if (rc && typeof rc.then === "function")
    rc = await rc;
  if (rc !== 0)
    fail(`circt-sim returned non-zero exit code: ${rc}`);

  // In Asyncify mode, callMain() may return before async rewinding fully
  // completes. Wait briefly for the post-await registration path to run.
  for (let i = 0; i < 200; ++i) {
    if (afterDelayRegAttempted && afterDelayHandle && afterDelayCount > 0)
      break;
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  if (startCount === 0)
    fail("yield hook did not observe cbStartOfSimulation");
  if (!afterDelayRegAttempted)
    fail("yield hook did not execute async cbStart registration path");
  if (!afterDelayHandle)
    fail("failed to register cbAfterDelay from async yield hook");
  if (afterDelayCount === 0)
    fail("cbAfterDelay callback did not fire");

  console.log(
      `WASM_VPI_STARTUP_YIELD PASS start=${startCount} afterDelay=${afterDelayCount}`);
}

main().catch(err => {
  console.error(err.stack || String(err));
  process.exit(1);
});
EOF

if ! grep -q '^WASM_VPI_STARTUP_YIELD PASS ' "$log"; then
  echo "[wasm-vpi-startup-yield] missing PASS marker" >&2
  cat "$log" >&2
  exit 1
fi

if grep -q 'Aborted(' "$log"; then
  echo "[wasm-vpi-startup-yield] detected wasm abort" >&2
  cat "$log" >&2
  exit 1
fi

echo "[wasm-vpi-startup-yield] PASS"
