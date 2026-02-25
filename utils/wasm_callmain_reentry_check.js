#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function die(message) {
  console.error(`[wasm-reentry] error: ${message}`);
  process.exit(1);
}

function usage() {
  console.error(
      "usage: wasm_callmain_reentry_check.js <tool.js> " +
      "--first <args...> --second <args...> " +
      "[--preload-file <host-path> <wasm-path>] " +
      "[--expect-wasm-file-substr <wasm-path> <text>] " +
      "[--expect-substr <text>] [--forbid-substr <text>]");
  process.exit(2);
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

function appendCapped(chunks, state, text, maxBytes) {
  if (!text || state.bytes >= maxBytes)
    return;
  const remaining = maxBytes - state.bytes;
  const textBytes = Buffer.byteLength(text);
  if (textBytes <= remaining) {
    chunks.push(text);
    state.bytes += textBytes;
    return;
  }
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    const bytes = Buffer.byteLength(text.slice(0, mid));
    if (bytes <= remaining)
      low = mid;
    else
      high = mid - 1;
  }
  if (low > 0) {
    chunks.push(text.slice(0, low));
    state.bytes += Buffer.byteLength(text.slice(0, low));
  }
}

function parseArgs(argv) {
  if (argv.length < 5)
    usage();

  const toolJs = path.resolve(argv[0]);
  const first = [];
  const second = [];
  const expects = [];
  const forbids = ["InitLLVM was already initialized!"];
  const preloads = [];
  const expectedWasmFileSubstrings = [];

  let mode = "";
  for (let i = 1; i < argv.length; ++i) {
    const tok = argv[i];
    if (tok === "--first") {
      mode = "first";
      continue;
    }
    if (tok === "--second") {
      mode = "second";
      continue;
    }
    if (tok === "--expect-substr") {
      if (i + 1 >= argv.length)
        usage();
      expects.push(argv[++i]);
      continue;
    }
    if (tok === "--preload-file") {
      if (i + 2 >= argv.length)
        usage();
      preloads.push({hostPath: argv[++i], wasmPath: argv[++i]});
      continue;
    }
    if (tok === "--forbid-substr") {
      if (i + 1 >= argv.length)
        usage();
      forbids.push(argv[++i]);
      continue;
    }
    if (tok === "--expect-wasm-file-substr") {
      if (i + 2 >= argv.length)
        usage();
      expectedWasmFileSubstrings.push({path: argv[++i], text: argv[++i]});
      continue;
    }
    if (mode === "first") {
      first.push(tok);
      continue;
    }
    if (mode === "second") {
      second.push(tok);
      continue;
    }
    usage();
  }

  if (first.length === 0 || second.length === 0)
    usage();
  return {
    toolJs,
    first,
    second,
    expects,
    forbids,
    preloads,
    expectedWasmFileSubstrings
  };
}

function waitUntilReady(ctx, timeoutMs = 20000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof ctx.Module.callMain === "function" &&
          typeof ctx.Module._main === "function" &&
          ctx.Module.calledRun) {
        resolve();
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        reject(new Error("timed out waiting for wasm runtime initialization"));
        return;
      }
      setTimeout(tick, 20);
    };
    tick();
  });
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

function callMainWithExitCapture(module, args) {
  try {
    return module.callMain(args);
  } catch (err) {
    if (isExitStatusLike(err))
      return extractExitStatus(err);
    throw err;
  }
}

async function main() {
  const {toolJs, first, second, expects, forbids, preloads,
         expectedWasmFileSubstrings} =
      parseArgs(process.argv.slice(2));
  if (!fs.existsSync(toolJs))
    die(`tool not found: ${toolJs}`);

  const toolDir = path.dirname(toolJs);
  const source = fs.readFileSync(toolJs, "utf8");

  const outLines = [];
  const errLines = [];
  const captureConsole = {
    log: (...args) => outLines.push(args.join(" ")),
    error: (...args) => errLines.push(args.join(" ")),
    warn: (...args) => errLines.push(args.join(" ")),
    info: (...args) => outLines.push(args.join(" ")),
  };

  const context = {
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
    __filename: toolJs,
    module: {exports: {}},
    exports: {},
  };
  context.globalThis = context;
  context.global = context;

  const stdoutChunks = [];
  const stderrChunks = [];
  const fdStdoutChunks = [];
  const fdStderrChunks = [];
  const MAX_FD_CAPTURE_BYTES = 2 * 1024 * 1024;
  const fdStdoutState = {bytes: 0};
  const fdStderrState = {bytes: 0};
  const origStdoutWrite = process.stdout.write.bind(process.stdout);
  const origStderrWrite = process.stderr.write.bind(process.stderr);
  const origFsWriteSync = fs.writeSync.bind(fs);
  const origFsWrite = fs.write.bind(fs);

  process.stdout.write = (chunk, encoding, cb) => {
    stdoutChunks.push(decodeUtf8Chunk(chunk));
    if (typeof cb === "function")
      cb();
    return true;
  };
  fs.writeSync = (fd, bufferOrString, offsetOrPosition, lengthOrEncoding, position) => {
    if (fd === 1 || fd === 2) {
      const reportedBytes = typeof bufferOrString === "string"
          ? Buffer.byteLength(bufferOrString)
          : (typeof offsetOrPosition === "number" && typeof lengthOrEncoding === "number"
                 ? lengthOrEncoding
                 : bufferLikeLength(bufferOrString));
      const offset =
          typeof offsetOrPosition === "number" && typeof lengthOrEncoding === "number"
          ? offsetOrPosition
          : 0;
      const length =
          typeof offsetOrPosition === "number" && typeof lengthOrEncoding === "number"
          ? lengthOrEncoding
          : bufferLikeLength(bufferOrString);
      const captureLength = Math.min(length, 8192);
      const text = decodeUtf8Chunk(bufferOrString, offset, captureLength);
      if (fd === 1) {
        appendCapped(fdStdoutChunks, fdStdoutState, text, MAX_FD_CAPTURE_BYTES);
      } else {
        appendCapped(fdStderrChunks, fdStderrState, text, MAX_FD_CAPTURE_BYTES);
      }
      // Swallow terminal writes from wasm runtime while still reporting byte count.
      return reportedBytes;
    }
    return origFsWriteSync(fd, bufferOrString, offsetOrPosition, lengthOrEncoding, position);
  };
  fs.write = (fd, bufferOrString, offsetOrString, lengthOrPosition, positionOrCb, maybeCb) => {
    if (fd === 1 || fd === 2) {
      const isStringWrite = typeof bufferOrString === "string";
      const explicitLength = !isStringWrite && typeof offsetOrString === "number" &&
          typeof lengthOrPosition === "number";
      const reportedBytes = isStringWrite
          ? Buffer.byteLength(bufferOrString)
          : (explicitLength ? lengthOrPosition : bufferLikeLength(bufferOrString));
      const offset = explicitLength ? offsetOrString : 0;
      const captureLength =
          Math.min(explicitLength ? lengthOrPosition : bufferLikeLength(bufferOrString), 8192);
      const text = decodeUtf8Chunk(bufferOrString, offset, captureLength);
      if (fd === 1) {
        appendCapped(fdStdoutChunks, fdStdoutState, text, MAX_FD_CAPTURE_BYTES);
      } else {
        appendCapped(fdStderrChunks, fdStderrState, text, MAX_FD_CAPTURE_BYTES);
      }
      const cb = typeof positionOrCb === "function" ? positionOrCb :
          (typeof maybeCb === "function" ? maybeCb : null);
      if (cb)
        cb(null, reportedBytes, bufferOrString);
      return;
    }
    return origFsWrite(fd, bufferOrString, offsetOrString, lengthOrPosition, positionOrCb, maybeCb);
  };
  process.stderr.write = (chunk, encoding, cb) => {
    stderrChunks.push(decodeUtf8Chunk(chunk));
    if (typeof cb === "function")
      cb();
    return true;
  };

  let rc1 = -1;
  let rc2 = -1;
  const preloadPathRemap = new Map();
  const remapArg = arg => preloadPathRemap.get(arg) ?? arg;
  try {
    vm.runInNewContext(source, context, {filename: toolJs});
    await waitUntilReady(context);

    if (preloads.length > 0) {
      if (!context.FS)
        die("wasm FS API is unavailable for --preload-file");
      const writePreload = (targetPath, hostData) => {
        context.FS.mkdirTree(path.posix.dirname(targetPath));
        context.FS.writeFile(targetPath, hostData);
      };
      for (const preload of preloads) {
        const hostAbs = path.resolve(preload.hostPath);
        const requestedTarget = preload.wasmPath;
        const hostData = fs.readFileSync(hostAbs);
        let actualTarget = requestedTarget;
        try {
          writePreload(actualTarget, hostData);
        } catch (err) {
          // With NODERAWFS enabled, absolute wasm-style paths (e.g.
          // "/inputs/test.mlir") may map to host root, which is often
          // unwritable in CI sandboxes. Fall back to CWD-relative placement.
          if (!path.posix.isAbsolute(requestedTarget))
            throw err;
          const cwd = context.FS.cwd();
          actualTarget =
              path.posix.join(cwd, requestedTarget.replace(/^\/+/, ""));
          writePreload(actualTarget, hostData);
        }
        preloadPathRemap.set(requestedTarget, actualTarget);
      }
    }

    const firstArgs = first.map(remapArg);
    const secondArgs = second.map(remapArg);
    rc1 = callMainWithExitCapture(context.Module, firstArgs);
    rc2 = callMainWithExitCapture(context.Module, secondArgs);
    // Flush any deferred Node-side stdio callbacks before restoring hooks.
    await new Promise(resolve => setTimeout(resolve, 20));
  } finally {
    process.stdout.write = origStdoutWrite;
    process.stderr.write = origStderrWrite;
    fs.writeSync = origFsWriteSync;
    fs.write = origFsWrite;
  }

  const combined =
      `${stdoutChunks.join("")}\n${stderrChunks.join("")}\n` +
      `${fdStdoutChunks.join("")}\n${fdStderrChunks.join("")}\n` +
      `${outLines.join("\n")}\n${errLines.join("\n")}`;
  if (rc1 !== 0)
    die(`first call failed with exit code ${rc1}`);
  if (rc2 !== 0)
    die(`second call failed with exit code ${rc2}`);

  for (const s of expects) {
    if (!combined.includes(s))
      die(`missing expected substring: ${JSON.stringify(s)}`);
  }
  for (const s of forbids) {
    if (combined.includes(s))
      die(`forbidden substring seen: ${JSON.stringify(s)}`);
  }
  const remappedExpectedWasmFileSubstrings =
      expectedWasmFileSubstrings.map(
          item => ({path: remapArg(item.path), text: item.text}));
  for (const expectedFile of remappedExpectedWasmFileSubstrings) {
    if (!context.FS)
      die("wasm FS API is unavailable for --expect-wasm-file-substr");
    let fileText = "";
    try {
      fileText = context.FS.readFile(expectedFile.path, {encoding: "utf8"});
    } catch (err) {
      die(`failed reading wasm file ${JSON.stringify(expectedFile.path)}: ${
          err instanceof Error ? err.message : String(err)}`);
    }
    if (!fileText.includes(expectedFile.text)) {
      die(`wasm file ${JSON.stringify(expectedFile.path)} does not contain ` +
          `${JSON.stringify(expectedFile.text)}`);
    }
  }

  console.log(`[wasm-reentry] ok: ${path.basename(toolJs)} rc1=${rc1} rc2=${rc2}`);
}

main().catch(err => {
  die(err instanceof Error ? err.message : String(err));
});
