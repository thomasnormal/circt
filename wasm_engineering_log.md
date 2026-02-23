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
