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
      "[--expect-substr <text>] [--forbid-substr <text>]");
  process.exit(2);
}

function parseArgs(argv) {
  if (argv.length < 5)
    usage();

  const toolJs = path.resolve(argv[0]);
  const first = [];
  const second = [];
  const expects = [];
  const forbids = ["InitLLVM was already initialized!"];

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
    if (tok === "--forbid-substr") {
      if (i + 1 >= argv.length)
        usage();
      forbids.push(argv[++i]);
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
  return {toolJs, first, second, expects, forbids};
}

function waitUntilReady(ctx, timeoutMs = 20000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof ctx.callMain === "function" &&
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

async function main() {
  const {toolJs, first, second, expects, forbids} = parseArgs(process.argv.slice(2));
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
  const origStdoutWrite = process.stdout.write.bind(process.stdout);
  const origStderrWrite = process.stderr.write.bind(process.stderr);

  process.stdout.write = (chunk, encoding, cb) => {
    stdoutChunks.push(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
    if (typeof cb === "function")
      cb();
    return true;
  };
  process.stderr.write = (chunk, encoding, cb) => {
    stderrChunks.push(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
    if (typeof cb === "function")
      cb();
    return true;
  };

  let rc1 = -1;
  let rc2 = -1;
  try {
    vm.runInNewContext(source, context, {filename: toolJs});
    await waitUntilReady(context);

    rc1 = context.callMain(first);
    rc2 = context.callMain(second);
  } finally {
    process.stdout.write = origStdoutWrite;
    process.stderr.write = origStderrWrite;
  }

  const combined =
      `${stdoutChunks.join("")}\n${stderrChunks.join("")}\n` +
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

  console.log(`[wasm-reentry] ok: ${path.basename(toolJs)} rc1=${rc1} rc2=${rc2}`);
}

main().catch(err => {
  die(err instanceof Error ? err.message : String(err));
});
