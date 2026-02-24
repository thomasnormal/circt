#!/usr/bin/env node

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const host = process.env.REPRO_HOST || "127.0.0.1";
const port = Number(process.env.REPRO_PORT || "43175");
const baseUrl = `http://${host}:${port}`;

const verilogJsPath = process.env.VERILOG_JS
  ? path.resolve(process.env.VERILOG_JS)
  : path.join(repoRoot, "build-wasm", "bin", "circt-verilog.js");
const verilogWasmPath = process.env.VERILOG_WASM
  ? path.resolve(process.env.VERILOG_WASM)
  : path.join(repoRoot, "build-wasm", "bin", "circt-verilog.wasm");
const uvmSrcRoot = process.env.UVM_SRC_ROOT
  ? path.resolve(process.env.UVM_SRC_ROOT)
  : path.join(repoRoot, "lib", "Runtime", "uvm-core", "src");

const timeoutMs = Number(process.env.REPRO_TIMEOUT_MS || "180000");
const serverReadyTimeoutMs = Number(process.env.REPRO_SERVER_READY_MS || "20000");

function parseExpectation(argv) {
  let expect = "pass";
  for (const arg of argv.slice(2)) {
    if (arg === "--expect-pass") {
      expect = "pass";
      continue;
    }
    if (arg === "--expect-fail") {
      expect = "fail";
      continue;
    }
    if (arg === "-h" || arg === "--help") {
      console.log(
        [
          "Usage:",
          "  node utils/repro_wasm_uvm_browser_assert.mjs [--expect-pass|--expect-fail]",
          "",
          "Modes:",
          "  --expect-pass  Require clean compile (default).",
          "  --expect-fail  Require malformed-attribute abort signature."
        ].join("\n")
      );
      process.exit(0);
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  return expect;
}

function requireReadable(filePath, label) {
  try {
    fs.accessSync(filePath, fs.constants.R_OK);
  } catch {
    throw new Error(`missing ${label}: ${filePath}`);
  }
}

function collectUvmFiles(root) {
  const out = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(abs);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!/\.(sv|svh)$/i.test(entry.name)) continue;
      out.push(path.relative(root, abs).split(path.sep).join("/"));
    }
  };
  walk(root);
  out.sort();
  return out;
}

function contentTypeFor(requestPath) {
  if (requestPath.endsWith(".js")) return "application/javascript";
  if (requestPath.endsWith(".wasm")) return "application/wasm";
  if (requestPath.endsWith(".sv") || requestPath.endsWith(".svh")) return "text/plain; charset=utf-8";
  if (requestPath.endsWith(".json")) return "application/json; charset=utf-8";
  return "text/plain; charset=utf-8";
}

function startServer(uvmFiles) {
  const manifestBody = JSON.stringify({ rootPath: "/circt/uvm-core/src", files: uvmFiles });
  const harnessHtml = "<!doctype html><meta charset='utf-8'><title>circt wasm repro</title>";

  const server = http.createServer((req, res) => {
    const url = new URL(req.url || "/", baseUrl);
    const pathname = url.pathname;

    try {
      if (pathname === "/" || pathname === "/index.html") {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
        res.end(harnessHtml);
        return;
      }

      if (pathname === "/circt/circt-verilog.js") {
        res.writeHead(200, { "content-type": "application/javascript", "cache-control": "no-store" });
        fs.createReadStream(verilogJsPath).pipe(res);
        return;
      }

      if (pathname === "/circt/circt-verilog.wasm") {
        res.writeHead(200, { "content-type": "application/wasm", "cache-control": "no-store" });
        fs.createReadStream(verilogWasmPath).pipe(res);
        return;
      }

      if (pathname === "/uvm-manifest.json") {
        res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
        res.end(manifestBody);
        return;
      }

      if (pathname.startsWith("/uvm/src/")) {
        const rel = pathname.slice("/uvm/src/".length).replace(/^\/+/, "");
        if (!rel || rel.includes("..")) {
          res.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
          res.end("bad path");
          return;
        }
        const abs = path.join(uvmSrcRoot, rel);
        if (!abs.startsWith(uvmSrcRoot + path.sep) && abs !== path.join(uvmSrcRoot, rel)) {
          res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
          res.end("forbidden");
          return;
        }
        if (!fs.existsSync(abs)) {
          res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
          res.end("not found");
          return;
        }
        res.writeHead(200, { "content-type": contentTypeFor(abs), "cache-control": "no-store" });
        fs.createReadStream(abs).pipe(res);
        return;
      }

      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end("not found");
    } catch (error) {
      res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      res.end(String(error && error.stack ? error.stack : error));
    }
  });

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`server did not become ready within ${serverReadyTimeoutMs}ms`));
    }, serverReadyTimeoutMs);
    server.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    server.listen(port, host, () => {
      clearTimeout(timer);
      resolve(server);
    });
  });
}

async function getChromium() {
  try {
    const mod = await import("@playwright/test");
    if (mod && mod.chromium) return mod.chromium;
  } catch {}
  try {
    const mod = await import("playwright");
    if (mod && mod.chromium) return mod.chromium;
  } catch {}
  throw new Error(
    "Missing Playwright runtime. Install one of:\n" +
      "  npm i -D @playwright/test\n" +
      "  npm i -D playwright"
  );
}

async function runReproInBrowser(chromium) {
  const browser = await chromium.launch({
    headless: true,
    channel: "chromium",
    args: [
      "--use-angle=swiftshader",
      "--enable-webgl",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist"
    ]
  });

  try {
    const page = await browser.newPage({ baseURL: baseUrl });
    await page.goto("/");

    const runResult = await page.evaluate(
      async ({ timeoutMs }) => {
        const PATH_SHIM = {
          sep: "/",
          isAbsolute: (p) => String(p).startsWith("/"),
          normalize: (p) => String(p).replace(/\/+/g, "/").replace(/(.)\/$/, "$1") || "/",
          dirname: (p) => {
            const s = String(p);
            const i = s.lastIndexOf("/");
            return i <= 0 ? "/" : s.slice(0, i);
          },
          basename: (p, ext) => {
            const b = String(p).split("/").pop() || "";
            return ext && b.endsWith(ext) ? b.slice(0, -ext.length) : b;
          },
          extname: (p) => {
            const b = String(p).split("/").pop() || "";
            const i = b.lastIndexOf(".");
            return i <= 0 ? "" : b.slice(i);
          },
          join: (...parts) => parts.join("/").replace(/\/+/g, "/"),
          join2: (a, b) => `${a}/${b}`.replace(/\/+/g, "/"),
          resolve: (...parts) => parts.join("/").replace(/\/+/g, "/")
        };
        PATH_SHIM.posix = PATH_SHIM;

        function makeInMemFS(onStdoutChunk, onStderrChunk) {
          const store = {};
          const symlinks = {};
          const dirs = new Set([
            "/",
            "/dev",
            "/workspace",
            "/workspace/src",
            "/workspace/out",
            "/circt",
            "/circt/uvm-core",
            "/circt/uvm-core/src"
          ]);
          const fds = {};
          let nextFd = 3;
          const stdoutChunks = [];
          const stderrChunks = [];

          function ensureParentDir(filePath) {
            const parts = String(filePath).split("/").filter(Boolean);
            let cur = "";
            for (let i = 0; i < parts.length - 1; i += 1) {
              cur += `/${parts[i]}`;
              dirs.add(cur);
            }
          }

          function makeStat(filePath) {
            const p = String(filePath);
            if (symlinks[p]) {
              return {
                ino: 3,
                mode: 0o120777,
                size: symlinks[p].length,
                dev: 1,
                nlink: 1,
                uid: 0,
                gid: 0,
                rdev: 0,
                blksize: 4096,
                blocks: 1,
                atime: new Date(),
                mtime: new Date(),
                ctime: new Date(),
                isDirectory: () => false,
                isFile: () => false,
                isSymbolicLink: () => true
              };
            }
            if (dirs.has(p)) {
              return {
                ino: 1,
                mode: 0o40755,
                size: 0,
                dev: 1,
                nlink: 1,
                uid: 0,
                gid: 0,
                rdev: 0,
                blksize: 4096,
                blocks: 0,
                atime: new Date(),
                mtime: new Date(),
                ctime: new Date(),
                isDirectory: () => true,
                isFile: () => false,
                isSymbolicLink: () => false
              };
            }
            if (store[p]) {
              const size = store[p].length;
              return {
                ino: 2,
                mode: 0o100644,
                size,
                dev: 1,
                nlink: 1,
                uid: 0,
                gid: 0,
                rdev: 0,
                blksize: 4096,
                blocks: Math.ceil(size / 512),
                atime: new Date(),
                mtime: new Date(),
                ctime: new Date(),
                isDirectory: () => false,
                isFile: () => true,
                isSymbolicLink: () => false
              };
            }
            const error = new Error(`ENOENT: no such file or directory, stat '${p}'`);
            error.code = "ENOENT";
            throw error;
          }

          const nodeApi = {
            readFileSync: (filePath, opts) => {
              const p = String(filePath);
              if (!store[p]) {
                const error = new Error(`ENOENT: no such file or directory, open '${p}'`);
                error.code = "ENOENT";
                throw error;
              }
              const enc = typeof opts === "string" ? opts : opts && opts.encoding;
              return enc ? new TextDecoder().decode(store[p]) : store[p];
            },
            existsSync: (p) => dirs.has(String(p)) || !!store[String(p)] || !!symlinks[String(p)],
            statSync: makeStat,
            lstatSync: makeStat,
            realpathSync: (p) => symlinks[String(p)] || String(p),
            readlinkSync: (p) => {
              const link = symlinks[String(p)];
              if (link) return link;
              const error = new Error(`EINVAL: invalid argument, readlink '${String(p)}'`);
              error.code = "EINVAL";
              throw error;
            },
            symlinkSync: (target, p) => {
              const linkPath = String(p);
              ensureParentDir(linkPath);
              symlinks[linkPath] = String(target);
            },
            fstatSync: (fd) => {
              const f = fds[fd];
              if (!f) {
                const error = new Error("EBADF");
                error.code = "EBADF";
                throw error;
              }
              return makeStat(f.path);
            },
            readdirSync: (p) => {
              const dirPath = String(p);
              if (!dirs.has(dirPath)) {
                const error = new Error(`ENOENT: ${dirPath}`);
                error.code = "ENOENT";
                throw error;
              }
              const prefix = dirPath === "/" ? "/" : `${dirPath}/`;
              const entries = new Set();
              for (const dir of dirs) {
                if (dir !== dirPath && dir.startsWith(prefix)) {
                  const rel = dir.slice(prefix.length);
                  if (!rel.includes("/")) entries.add(rel);
                }
              }
              for (const filePath of Object.keys(store)) {
                if (filePath.startsWith(prefix)) {
                  const rel = filePath.slice(prefix.length);
                  if (!rel.includes("/")) entries.add(rel);
                }
              }
              return Array.from(entries);
            },
            mkdirSync: (p) => {
              dirs.add(String(p));
            },
            rmdirSync: (p) => {
              dirs.delete(String(p));
            },
            unlinkSync: (p) => {
              delete store[String(p)];
              delete symlinks[String(p)];
            },
            renameSync: (from, to) => {
              const src = String(from);
              const dst = String(to);
              if (store[src]) {
                store[dst] = store[src];
                delete store[src];
              }
              if (symlinks[src]) {
                symlinks[dst] = symlinks[src];
                delete symlinks[src];
              }
            },
            chmodSync: () => {},
            chownSync: () => {},
            utimesSync: () => {},
            fsyncSync: () => {},
            ftruncateSync: (fd, len) => {
              const f = fds[fd];
              const data = store[f.path] || new Uint8Array(0);
              store[f.path] =
                len <= data.length
                  ? data.subarray(0, len)
                  : (() => {
                      const next = new Uint8Array(len);
                      next.set(data);
                      return next;
                    })();
            },
            openSync: (filePath, flags) => {
              const p = String(filePath);
              let writable = false;
              if (typeof flags === "string") {
                writable = flags.includes("w") || flags.includes("a") || flags.includes("+");
              } else {
                const numericFlags = Number(flags) || 0;
                const accessMode = numericFlags & 3;
                writable =
                  accessMode !== 0 ||
                  !!(numericFlags & 64) ||
                  !!(numericFlags & 512) ||
                  !!(numericFlags & 1024);
              }
              if (writable) {
                ensureParentDir(p);
                store[p] = new Uint8Array(0);
              } else if (!store[p] && !dirs.has(p)) {
                const error = new Error(`ENOENT: ${p}`);
                error.code = "ENOENT";
                throw error;
              }
              const fd = nextFd++;
              fds[fd] = { path: p, pos: 0 };
              return fd;
            },
            closeSync: (fd) => {
              delete fds[fd];
            },
            readSync: (fd, buf, bufOffset, length, position) => {
              if (fd === 0) return 0;
              const f = fds[fd];
              const data = store[f.path] || new Uint8Array(0);
              const pos = position != null ? position : f.pos;
              const avail = Math.min(length, data.length - pos);
              if (avail <= 0) return 0;
              buf.set(data.subarray(pos, pos + avail), bufOffset);
              if (position == null) f.pos += avail;
              return avail;
            },
            writeSync: (fd, buf, bufOffset, length, position) => {
              let src;
              if (typeof buf === "string") src = new TextEncoder().encode(buf);
              else if (ArrayBuffer.isView(buf))
                src = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
              else if (buf instanceof ArrayBuffer) src = new Uint8Array(buf);
              else src = new Uint8Array(0);

              let start = Number.isFinite(bufOffset) ? Number(bufOffset) : 0;
              if (start < 0) start = 0;
              let writeLen = Number.isFinite(length) ? Number(length) : src.length - start;
              if (writeLen < 0) writeLen = 0;
              const end = Math.min(src.length, start + writeLen);
              const chunk = src.subarray(start, end);

              if (fd === 1) {
                const text = new TextDecoder().decode(chunk);
                stdoutChunks.push(text);
                if (typeof onStdoutChunk === "function") onStdoutChunk(text);
                return chunk.length;
              }
              if (fd === 2) {
                const text = new TextDecoder().decode(chunk);
                stderrChunks.push(text);
                if (typeof onStderrChunk === "function") onStderrChunk(text);
                return chunk.length;
              }

              const f = fds[fd];
              const pos = position != null ? position : f.pos;
              let data = store[f.path] || new Uint8Array(0);
              const needed = pos + chunk.length;
              if (needed > data.length) {
                const grown = new Uint8Array(needed);
                grown.set(data);
                data = grown;
              }
              data.set(chunk, pos);
              store[f.path] = data;
              if (position == null) f.pos += chunk.length;
              return chunk.length;
            }
          };

          return {
            nodeApi,
            ensureDir: (p) => {
              dirs.add(String(p));
            },
            writeTextFile: (p, text) => {
              const filePath = String(p);
              ensureParentDir(filePath);
              store[filePath] = new TextEncoder().encode(String(text ?? ""));
            },
            readTextFile: (p) => {
              const filePath = String(p);
              if (!store[filePath]) return null;
              return new TextDecoder().decode(store[filePath]);
            },
            getStdout: () => stdoutChunks.join(""),
            getStderr: () => stderrChunks.join("")
          };
        }

        function isExitException(error) {
          if (!error) return false;
          const name = String(error.name || "");
          const message = String(error.message || "");
          return (
            name === "ExitStatus" ||
            /exit\(/i.test(message) ||
            (typeof error.status === "number" && Number.isFinite(error.status))
          );
        }

        function extractExitCode(error) {
          if (!error) return 1;
          if (typeof error.status === "number" && Number.isFinite(error.status)) return error.status;
          const message = String(error.message || "");
          const match = message.match(/exit\(([-+]?\d+)\)/i);
          if (match) return Number(match[1]);
          return 1;
        }

        function waitForRuntime() {
          const start = Date.now();
          return new Promise((resolve, reject) => {
            const tick = () => {
              const module = globalThis.Module;
              if (module && typeof globalThis.callMain === "function" && typeof module.callMain !== "function") {
                module.callMain = globalThis.callMain;
              }
              if (module && !module.FS && globalThis.FS) {
                module.FS = globalThis.FS;
              }
              if (
                module &&
                typeof module.callMain === "function" &&
                typeof module._main === "function" &&
                module.FS &&
                typeof module.FS.writeFile === "function" &&
                typeof module.FS.readFile === "function"
              ) {
                resolve(module);
                return;
              }
              if (Date.now() - start >= timeoutMs) {
                reject(new Error(`runtime not ready after ${timeoutMs}ms`));
                return;
              }
              setTimeout(tick, 25);
            };
            tick();
          });
        }

        const streamed = [];
        const appendLine = (stream, line) => {
          streamed.push(`[${stream}] ${String(line ?? "")}`);
        };

        const jsUrl = "/circt/circt-verilog.js";
        const wasmUrl = "/circt/circt-verilog.wasm";
        const toolScriptResp = await fetch(jsUrl, { cache: "no-store" });
        if (!toolScriptResp.ok) throw new Error(`failed to fetch tool script: ${toolScriptResp.status}`);
        const toolScript = await toolScriptResp.text();
        const isNoderawfs =
          toolScript.includes("NODERAWFS is currently only supported") ||
          toolScript.includes("var nodePath=require(");
        if (!isNoderawfs) {
          throw new Error("expected NODERAWFS build, but detection tokens were not found");
        }

        const inMemFS = makeInMemFS(
          (text) => appendLine("stdout", text),
          (text) => appendLine("stderr", text)
        );

        globalThis.Module = {
          noInitialRun: true,
          print: (line) => appendLine("stdout", line),
          printErr: (line) => appendLine("stderr", line),
          locateFile: (p) => (String(p).endsWith(".wasm") ? wasmUrl : p),
          instantiateWasm: (imports, callback) => {
            WebAssembly.instantiateStreaming(fetch(wasmUrl), imports)
              .then((result) => callback(result.instance, result.module))
              .catch(async () => {
                const buf = await (await fetch(wasmUrl)).arrayBuffer();
                const result = await WebAssembly.instantiate(buf, imports);
                callback(result.instance, result.module);
              });
            return {};
          }
        };

        const proc = {};
        proc.versions = { node: "18.0.0" };
        proc.version = "v18.0.0";
        proc.platform = "linux";
        proc.argv = ["node", "/tool"];
        proc.type = "worker";
        proc.exitCode = 0;
        proc.exit = (code) => {
          throw { name: "ExitStatus", message: `exit(${code | 0})`, status: code | 0 };
        };
        proc.on = () => proc;
        proc.stdout = { write: (s) => appendLine("stdout", s), isTTY: false };
        proc.stderr = { write: (s) => appendLine("stderr", s), isTTY: false };
        proc.stdin = null;
        proc.env = {};
        proc.cwd = () => "/workspace";
        proc.binding = (name) => {
          if (name === "constants") {
            return {
              fs: {
                O_APPEND: 1024,
                O_CREAT: 64,
                O_EXCL: 128,
                O_NOCTTY: 256,
                O_RDONLY: 0,
                O_RDWR: 2,
                O_SYNC: 4096,
                O_TRUNC: 512,
                O_WRONLY: 1,
                O_NOFOLLOW: 131072
              }
            };
          }
          throw new Error(`process.binding(${String(name)}) is not available`);
        };

        globalThis.process = proc;
        globalThis.global = globalThis;
        globalThis.__dirname = "/";
        globalThis.__filename = "/tool.js";
        globalThis.require = (mod) => {
          if (mod === "path") return PATH_SHIM;
          if (mod === "fs") return inMemFS.nodeApi;
          if (mod === "crypto")
            return { randomBytes: (n) => crypto.getRandomValues(new Uint8Array(n)) };
          if (mod === "child_process")
            return { spawnSync: () => ({ status: 1, stdout: "", stderr: "" }) };
          throw new Error(`require('${mod}') is not available in browser`);
        };

        (0, eval)(toolScript);

        const module = await waitForRuntime();

        const manifestResp = await fetch("/uvm-manifest.json", { cache: "no-store" });
        if (!manifestResp.ok) throw new Error(`failed to fetch UVM manifest: ${manifestResp.status}`);
        const manifest = await manifestResp.json();
        const relPaths = Array.isArray(manifest.files) ? manifest.files : [];
        for (const rel of relPaths) {
          const srcResp = await fetch(`/uvm/src/${rel}`, { cache: "no-store" });
          if (!srcResp.ok) throw new Error(`failed to fetch UVM source: ${rel}`);
          const srcText = await srcResp.text();
          inMemFS.writeTextFile(`/circt/uvm-core/src/${rel}`, srcText);
        }

        inMemFS.ensureDir("/workspace");
        inMemFS.ensureDir("/workspace/src");
        inMemFS.ensureDir("/workspace/out");
        inMemFS.writeTextFile(
          "/workspace/src/my_test.sv",
          [
            "import uvm_pkg::*;",
            '`include "uvm_macros.svh"',
            "",
            "class my_test extends uvm_test;",
            "  `uvm_component_utils(my_test)",
            "  function new(string name, uvm_component parent);",
            "    super.new(name, parent);",
            "  endfunction",
            "  task run_phase(uvm_phase phase);",
            "    phase.raise_objection(this);",
            '    `uvm_info("TEST", "Hello from UVM!", UVM_LOW)',
            "    phase.drop_objection(this);",
            "  endtask",
            "endclass",
            ""
          ].join("\n")
        );
        inMemFS.writeTextFile(
          "/workspace/src/tb_top.sv",
          [
            "import uvm_pkg::*;",
            '`include "uvm_macros.svh"',
            '`include "my_test.sv"',
            "",
            "module tb_top;",
            '  initial run_test("my_test");',
            "endmodule",
            ""
          ].join("\n")
        );

        const args = [
          "--resource-guard=false",
          "--ir-llhd",
          "--timescale",
          "1ns/1ns",
          "--uvm-path",
          "/circt/uvm-core",
          "-I",
          "/circt/uvm-core/src",
          "--top",
          "tb_top",
          "-o",
          "/workspace/out/design.llhd.mlir",
          "/workspace/src/tb_top.sv"
        ];

        let exitCode = 0;
        let callMainError = null;
        try {
          const ret = module.callMain(args);
          if (typeof ret === "number" && Number.isFinite(ret)) exitCode = ret;
        } catch (error) {
          if (isExitException(error)) exitCode = extractExitCode(error);
          else callMainError = String(error && error.stack ? error.stack : error);
        }

        const outMlir = inMemFS.readTextFile("/workspace/out/design.llhd.mlir");
        return {
          exitCode,
          callMainError,
          outMlirBytes: typeof outMlir === "string" ? outMlir.length : 0,
          stdout: inMemFS.getStdout(),
          stderr: inMemFS.getStderr(),
          streamed
        };
      },
      { timeoutMs }
    );

    return runResult;
  } finally {
    await browser.close();
  }
}

function summarize(result) {
  const combined = [
    result.callMainError || "",
    result.stdout || "",
    result.stderr || "",
    ...(Array.isArray(result.streamed) ? result.streamed : [])
  ].join("\n");

  const hasMalformed = /Malformed attribute storage object/.test(combined);
  const hasAbort = /Aborted\(/.test(combined);
  const reachedSim = /\$ circt-sim\b/.test(combined);

  return { combined, hasMalformed, hasAbort, reachedSim };
}

async function main() {
  const expect = parseExpectation(process.argv);
  requireReadable(verilogJsPath, "circt-verilog.js");
  requireReadable(verilogWasmPath, "circt-verilog.wasm");
  requireReadable(path.join(uvmSrcRoot, "uvm_pkg.sv"), "uvm_pkg.sv");

  const uvmFiles = collectUvmFiles(uvmSrcRoot);
  const server = await startServer(uvmFiles);
  let chromium;
  try {
    chromium = await getChromium();
    const result = await runReproInBrowser(chromium);
    const s = summarize(result);

    console.log("=== Browser UVM Repro Summary ===");
    console.log(`exitCode=${result.exitCode}`);
    console.log(`callMainErrorPresent=${result.callMainError ? "true" : "false"}`);
    console.log(`outMlirBytes=${result.outMlirBytes}`);
    console.log(`hasMalformed=${s.hasMalformed}`);
    console.log(`hasAbort=${s.hasAbort}`);
    console.log(`reachedSim=${s.reachedSim}`);
    console.log("=== Log Snippet (first 250 lines) ===");
    console.log(s.combined.split(/\r?\n/).slice(0, 250).join("\n"));

    const reproduced =
      (result.exitCode !== 0 || !!result.callMainError) &&
      result.outMlirBytes === 0 &&
      s.hasMalformed &&
      s.hasAbort &&
      !s.reachedSim;
    const cleanPass =
      result.exitCode === 0 &&
      !result.callMainError &&
      result.outMlirBytes > 0 &&
      !s.hasMalformed &&
      !s.hasAbort;

    if (expect === "fail") {
      if (!reproduced) {
        console.error("ERROR: expected malformed-attribute abort signature was not reproduced");
        process.exitCode = 1;
        return;
      }
      console.log("REPRODUCED: malformed attribute storage abort in browser-like wasm execution");
      return;
    }

    if (!cleanPass) {
      console.error("ERROR: expected clean pass, but compile did not satisfy pass criteria");
      process.exitCode = 1;
      return;
    }
    console.log("PASS: browser-like wasm UVM compile completed without malformed-attribute abort");
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((error) => {
  console.error(String(error && error.stack ? error.stack : error));
  process.exit(1);
});
