#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
CHECK_VERILOG="${CHECK_VERILOG:-auto}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-version-reentry] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ "$CHECK_VERILOG" != "auto" && "$CHECK_VERILOG" != "0" && "$CHECK_VERILOG" != "1" ]]; then
  echo "[wasm-version-reentry] invalid CHECK_VERILOG value: $CHECK_VERILOG (expected auto, 0, or 1)" >&2
  exit 1
fi

bmc_js="$BUILD_DIR/bin/circt-bmc.js"
sim_js="$BUILD_DIR/bin/circt-sim.js"
verilog_js="$BUILD_DIR/bin/circt-verilog.js"

for required in "$bmc_js" "$sim_js"; do
  if [[ ! -f "$required" ]]; then
    echo "[wasm-version-reentry] missing tool: $required" >&2
    exit 1
  fi
done

if [[ "$CHECK_VERILOG" == "auto" ]]; then
  if [[ -f "$verilog_js" ]]; then
    CHECK_VERILOG=1
  else
    CHECK_VERILOG=0
  fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

tools=("$bmc_js" "$sim_js")
if [[ "$CHECK_VERILOG" == "1" ]]; then
  if [[ ! -f "$verilog_js" ]]; then
    echo "[wasm-version-reentry] CHECK_VERILOG=1 but circt-verilog.js is missing: $verilog_js" >&2
    exit 1
  fi
  tools+=("$verilog_js")
fi

for tool_js in "${tools[@]}"; do
  echo "[wasm-version-reentry] same-instance callMain: --version -> --version ($(basename "$tool_js"))"
  set +e
  "$NODE_BIN" - "$tool_js" >"$tmpdir/$(basename "$tool_js").out" 2>"$tmpdir/$(basename "$tool_js").err" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function die(msg) {
  throw new Error(msg);
}

function isExitStatusLike(err) {
  if (!err)
    return false;
  if (typeof err.status === "number" && Number.isFinite(err.status))
    return true;
  const msg = String(err.message || err);
  return /exit\(([-+]?\d+)\)/i.test(msg);
}

function extractExitStatus(err) {
  if (typeof err.status === "number" && Number.isFinite(err.status))
    return err.status;
  const msg = String(err.message || err);
  const match = msg.match(/exit\(([-+]?\d+)\)/i);
  if (match)
    return Number(match[1]);
  return 1;
}

function bufferLikeLength(arg) {
  if (typeof arg === "string")
    return Buffer.byteLength(arg);
  if (arg && typeof arg.length === "number")
    return arg.length;
  if (arg instanceof ArrayBuffer)
    return arg.byteLength;
  return 0;
}

function decodeUtf8Chunk(arg, offset = 0, length = null) {
  if (typeof arg === "string")
    return arg;
  if (Buffer.isBuffer(arg)) {
    const start = Math.max(0, offset | 0);
    const end = length == null ? arg.length : Math.min(arg.length, start + (length | 0));
    return arg.toString("utf8", start, Math.max(start, end));
  }
  if (ArrayBuffer.isView(arg)) {
    const view = arg;
    const start = Math.max(0, offset | 0);
    const avail = Math.max(0, view.byteLength - start);
    const take = length == null ? avail : Math.max(0, Math.min(avail, length | 0));
    if (take === 0)
      return "";
    return Buffer.from(view.buffer, view.byteOffset + start, take).toString("utf8");
  }
  if (arg instanceof ArrayBuffer) {
    const start = Math.max(0, offset | 0);
    const avail = Math.max(0, arg.byteLength - start);
    const take = length == null ? avail : Math.max(0, Math.min(avail, length | 0));
    if (take === 0)
      return "";
    return Buffer.from(arg, start, take).toString("utf8");
  }
  return "";
}

function countLinePrefix(text, prefix) {
  let count = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith(prefix))
      ++count;
  }
  return count;
}

function callMainWithExitCapture(module, args) {
  try {
    return module.callMain(args);
  } catch (err) {
    if (isExitStatusLike(err))
      return extractExitStatus(err);
    throw err;
  }
}

async function waitUntilReady(ctx, timeoutMs = 60000) {
  const start = Date.now();
  while (true) {
    if (typeof ctx.Module.callMain === "function" &&
        typeof ctx.Module._main === "function" &&
        ctx.Module.calledRun)
      return;
    if (Date.now() - start > timeoutMs)
      die("timed out waiting for wasm runtime initialization");
    await new Promise(resolve => setTimeout(resolve, 20));
  }
}

async function captureVersionCall(ctx) {
  const chunks = [];
  const origStdoutWrite = process.stdout.write.bind(process.stdout);
  const origStderrWrite = process.stderr.write.bind(process.stderr);
  const origFsWriteSync = fs.writeSync.bind(fs);
  const origFsWrite = fs.write.bind(fs);

  process.stdout.write = (chunk, encoding, cb) => {
    chunks.push(decodeUtf8Chunk(chunk));
    if (typeof cb === "function")
      cb();
    return true;
  };
  process.stderr.write = (chunk, encoding, cb) => {
    chunks.push(decodeUtf8Chunk(chunk));
    if (typeof cb === "function")
      cb();
    return true;
  };
  fs.writeSync = (fd, bufferOrString, offsetOrPosition, lengthOrEncoding, position) => {
    if (fd === 1 || fd === 2) {
      const hasExplicitRange =
          typeof offsetOrPosition === "number" && typeof lengthOrEncoding === "number";
      const offset = hasExplicitRange ? offsetOrPosition : 0;
      const length = hasExplicitRange ? lengthOrEncoding : bufferLikeLength(bufferOrString);
      chunks.push(decodeUtf8Chunk(bufferOrString, offset, length));
      if (typeof bufferOrString === "string")
        return Buffer.byteLength(bufferOrString);
      if (hasExplicitRange)
        return lengthOrEncoding;
      return bufferLikeLength(bufferOrString);
    }
    return origFsWriteSync(fd, bufferOrString, offsetOrPosition, lengthOrEncoding, position);
  };
  fs.write = (fd, bufferOrString, offsetOrString, lengthOrPosition, positionOrCb, maybeCb) => {
    if (fd === 1 || fd === 2) {
      const hasExplicitRange =
          typeof offsetOrString === "number" && typeof lengthOrPosition === "number";
      const offset = hasExplicitRange ? offsetOrString : 0;
      const length = hasExplicitRange ? lengthOrPosition : bufferLikeLength(bufferOrString);
      chunks.push(decodeUtf8Chunk(bufferOrString, offset, length));
      const cb = typeof positionOrCb === "function"
          ? positionOrCb
          : (typeof maybeCb === "function" ? maybeCb : null);
      const reportedBytes = typeof bufferOrString === "string"
          ? Buffer.byteLength(bufferOrString)
          : (hasExplicitRange ? lengthOrPosition : bufferLikeLength(bufferOrString));
      if (cb)
        cb(null, reportedBytes, bufferOrString);
      return;
    }
    return origFsWrite(fd, bufferOrString, offsetOrString, lengthOrPosition, positionOrCb, maybeCb);
  };

  try {
    const rc = callMainWithExitCapture(ctx.Module, ["--version"]);
    if (rc !== 0)
      die(`--version returned non-zero exit code ${rc}`);
    await new Promise(resolve => setTimeout(resolve, 20));
    return chunks.join("");
  } finally {
    process.stdout.write = origStdoutWrite;
    process.stderr.write = origStderrWrite;
    fs.writeSync = origFsWriteSync;
    fs.write = origFsWrite;
  }
}

async function main() {
  const toolJs = process.argv[2];
  if (!toolJs)
    die("missing tool path");
  const toolAbs = path.resolve(toolJs);
  if (!fs.existsSync(toolAbs))
    die(`tool not found: ${toolAbs}`);
  const toolDir = path.dirname(toolAbs);

  const source = fs.readFileSync(toolAbs, "utf8");
  const captureConsole = {
    log: () => {},
    error: () => {},
    warn: () => {},
    info: () => {},
  };
  const ctx = {
    Module: {
      noInitialRun: true,
      locateFile: file => path.join(toolDir, file),
    },
    require,
    process,
    console: captureConsole,
    Buffer,
    URL,
    TextDecoder,
    TextEncoder,
    setTimeout,
    clearTimeout,
    performance,
    __dirname: toolDir,
    __filename: toolAbs,
    module: {exports: {}},
    exports: {},
  };
  ctx.globalThis = ctx;
  ctx.global = ctx;

  vm.runInNewContext(source, ctx, {filename: toolAbs});
  await waitUntilReady(ctx);

  const first = await captureVersionCall(ctx);
  const second = await captureVersionCall(ctx);

  const circtCountFirst = countLinePrefix(first, "CIRCT ");
  const circtCountSecond = countLinePrefix(second, "CIRCT ");
  if (circtCountFirst !== 1)
    die(`first --version emitted CIRCT banner ${circtCountFirst} times (expected 1)`);
  if (circtCountSecond !== 1)
    die(`second --version emitted CIRCT banner ${circtCountSecond} times (expected 1)`);

  const slangCountFirst = countLinePrefix(first, "slang version");
  if (slangCountFirst > 0) {
    const slangCountSecond = countLinePrefix(second, "slang version");
    if (slangCountSecond !== slangCountFirst) {
      die(`second --version emitted slang banner ${slangCountSecond} times (expected ${slangCountFirst})`);
    }
  }
}

main().catch(err => {
  console.error(`[wasm-version-reentry] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
NODE
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "[wasm-version-reentry] failed for $(basename "$tool_js")" >&2
    cat "$tmpdir/$(basename "$tool_js").err" >&2
    exit 1
  fi
done

echo "[wasm-version-reentry] PASS"
