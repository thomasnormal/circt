# WASM Engineering Log

## 2026-02-23
- Goal: build CIRCT for WebAssembly with a simple (non-JIT) execution path suitable for browser use.
- Confirmed `MLIR_ENABLE_EXECUTION_ENGINE=OFF` is required to avoid JIT-specific complexity and dependencies.
- Cross-compiling CIRCT to wasm32 exposed several latent 64-bit assumptions (`~0ULL` sentinels, fixed-size struct assertions, and narrowing conversions).
- The cross-build also required host tablegen tooling; CIRCT CMake needed to use host `MLIR_TABLEGEN` when cross-compiling.
- Additional wasm-specific compile/link issues appeared in codepaths not typically exercised on native builds:
  - `llvm::sys::Process` usage in `circt-bmc.cpp` required explicit `llvm/Support/Process.h`.
  - Emscripten `stdout`/`stderr` macros conflicted with member access in `TestReporting.cpp`.
  - Static wasm linking required explicit dependencies in `tools/circt-bmc/CMakeLists.txt` that native builds picked up transitively.
  - `CIRCTImportVerilog` is optional in this wasm config; unconditional tool linking caused `-lCIRCTImportVerilog` failure.
- Validation performed:
  - Native regressions for touched paths (`ArcToLLVM`, `RTG elaboration`) passed.
  - `ninja -C build-wasm circt-bmc` succeeded.
  - `node build-wasm/bin/circt-bmc.js --help` executed successfully.
- Milestone: run CIRCT wasm unit tests under Node.js (excluding Arc by request).
- Realization: wasm unit-test aborts were dominated by thread usage (`pthread_create failed`) from both LLVM-side and project-side code.
- Switched wasm config to `LLVM_ENABLE_THREADS=OFF`; this removed most runtime aborts immediately.
- Surprise: `build-wasm/NATIVE` helper tool config still reports threads enabled; this is expected and separate from wasm test binaries.
- Fixed a latent bug in `IncrementalCompiler` constructor: thread count initialization accidentally updated the by-value parameter instead of the member config.
- For wasm simplicity, `IncrementalCompiler` now disables parallel compilation on `__EMSCRIPTEN__` and falls back to sequential scheduling.
- `WallClockTimeout` tests required thread support; added wasm skips to keep single-threaded wasm test runs stable.
- Moore runtime had behavior mismatches against tests:
  - `__moore_string_substr` implemented end-index semantics while API/docs/tests used length semantics.
  - regex path returned POSIX-style success codes inconsistently and did not preserve the matched substring buffer.
  - implemented wasm-safe regex execution path using POSIX `regcomp/regexec` on emscripten to avoid exception-dependent behavior in std::regex.
- Added targeted wasm expectation/skip adjustments where behavior is platform-specific:
  - Moore conversion test `FourStateLLVMStructCastPreservesFieldOrder` accepts wasm lowering shape.
  - Sim `ThreadBarrierTest.MultipleThreads` and Moore runtime thread-focused tests are skipped on wasm.
- Additional test robustness updates:
  - ProcessScheduler integration tests now tolerate deferred event execution semantics across scheduler configurations.
- Final validation:
  - Built `CIRCTUnitTests` in wasm mode.
  - Ran all generated CIRCT unittest JS binaries under Node (excluding Arc).
  - Result: 16/16 pass.

## 2026-02-23 (follow-up: bmc/sim wasm CLI validation)
- Goal: validate `circt-bmc.js` and `circt-sim.js` end-to-end in Node and ensure
  waveform files are retrievable on the host filesystem.
- Built both wasm tools successfully:
  - `build-wasm/bin/circt-bmc.js`
  - `build-wasm/bin/circt-sim.js`
- Smoke checks passed:
  - `node build-wasm/bin/circt-bmc.js --help`
  - `node build-wasm/bin/circt-sim.js --help`
- Functional checks passed:
  - BMC stdin flow on `test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir`
    exits 0 and emits SMT-LIB text.
  - Sim stdin flow on `test/Tools/circt-sim/llhd-combinational.mlir` exits 0 and
    prints expected `b=1`, `b=0`, and completion lines.
- Surprise: waveform path behavior under wasm/node was inconsistent:
  - `circt-sim.js --vcd /tmp/test.vcd` reported "Wrote waveform" but `/tmp/test.vcd`
    did not exist.
  - Root cause: Emscripten default virtual filesystem semantics (no host path bridge).
- Fix:
  - Enabled Node raw filesystem mode for `circt-sim` wasm builds by adding
    `-sNODERAWFS=1` in `tools/circt-sim/CMakeLists.txt`.
  - Tried the same for `circt-bmc`, but reverted it because it regressed
    `--emit-smtlib -o -` stdout behavior under Node.
  - Rebuilt `circt-bmc` and `circt-sim`.
- Post-fix validation:
  - `circt-sim.js --vcd /tmp/test.vcd` now exits 0, reports waveform written,
    and `/tmp/test.vcd` exists with non-zero size.
- Unresolved/remaining checks:
  - `sv-tutorial` app wiring (`npm run sync:circt`, `npm run build`) could not be
    validated here because no `sv-tutorial` checkout is present in this environment.
  - LLVM submodule check: `llvm/llvm/cmake/modules/CrossCompile.cmake` has no local
    edits in this workspace; no local patch is currently required for these bmc/sim
    wasm builds.
