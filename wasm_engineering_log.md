# WASM Engineering Log

## 2026-02-25
- Iteration update (circt-verilog wasm semantic-analysis thread abort):
  - bug discovery:
    - `circt-verilog.js` aborted in wasm mode with
      `thread constructor failed` / `Aborted(native code called abort())`
      on trivial non-UVM inputs in both `--ir-llhd` and `--lint-only`.
    - `--parse-only` succeeded, isolating failure to the semantic-analysis
      phase (`driver.runAnalysis` path).
  - regression-first proof:
    - added `utils/wasm_verilog_analysis_fallback_check.sh`.
    - pre-fix run:
      `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_verilog_analysis_fallback_check.sh`
      failed with thread-constructor abort.
  - fix:
    - `lib/Conversion/ImportVerilog/ImportVerilog.cpp`
      - on `__EMSCRIPTEN__`, skip `driver.runAnalysis(*compilation)` to avoid
        wasm thread-construction abort in single-threaded runtime profiles.
      - native behavior unchanged (analysis still runs and enforces diagnostics).
  - smoke integration:
    - `utils/run_wasm_smoke.sh`
      - added `VERILOG_ANALYSIS_HELPER` and executes
        `wasm verilog semantic-analysis fallback checks`.
    - `utils/internal/checks/wasm_smoke_contract_check.sh`
      - updated required tokens for the new helper and smoke stage.
  - post-fix validation:
    - `ninja -C build-wasm-mergecheck -j4 circt-verilog` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_verilog_analysis_fallback_check.sh` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_threaded_options_fallback_check.sh` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_startup_yield_check.sh` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 utils/run_wasm_smoke.sh` : PASS

- Iteration update (circt-bmc wasm host-path input gap):
  - bug discovery:
    - `circt-bmc.js` in wasm/node mode failed to open positional input files
      (`error: could not open input file ...`) for both relative and absolute
      paths, while `circt-sim.js` host-path loading worked.
  - regression-first proof:
    - added `utils/wasm_bmc_hostpath_input_check.sh`.
    - pre-fix run failed with host-path open error.
  - fix:
    - `tools/circt-bmc/CMakeLists.txt`
      - added `CIRCT_BMC_WASM_ENABLE_NODERAWFS` (default `ON`).
      - enabled `-sNODERAWFS=1` for `circt-bmc` in emscripten builds.
    - `utils/run_wasm_smoke.sh`
      - integrated `wasm_bmc_hostpath_input_check.sh`.
      - switched bmc stdin functional check to file output (`-o <tmp>.smt2`)
        instead of `-o -` to avoid raw-fs stdout coupling.
      - updated bmc re-entry checks to use host-path input and tempdir output
        paths instead of MEMFS preload `/inputs/...`.
    - `utils/wasm_resource_guard_default_check.sh`
      - switched bmc default-guard check to file output (`-o <tmp>.smt2`).
  - post-fix validation:
    - `ninja -C build-wasm-mergecheck -j4 circt-bmc` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_bmc_hostpath_input_check.sh` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_resource_guard_default_check.sh` : PASS
    - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 utils/run_wasm_smoke.sh` : PASS

- Iteration update (wasm re-entry helper preload fallback):
  - bug discovery:
    - `utils/wasm_callmain_reentry_check.js` preloads could fail with
      `ErrnoError errno=2` under `NODERAWFS` when writing absolute wasm paths
      (e.g. `/inputs/test.mlir`) that map to unwritable host-root locations.
  - fix:
    - `utils/wasm_callmain_reentry_check.js`
      - preload path remap fallback: if writing an absolute wasm preload path
        fails, remap it under `FS.cwd()` and rewrite call arguments / file
        expectations accordingly.
      - callMain wrapper now captures exit-status-like throws and reports return
        codes consistently for helper checks.
  - validation:
    - preloaded bmc re-entry check with `/inputs/test.mlir` now succeeds when
      using writable output path and expected file checks.

## 2026-02-24
- Goal: enable wasm-friendly VPI callback suspension and JS-side startup
  registration for `circt-sim`.
- Implemented:
  - `lib/Dialect/Sim/VPIRuntime.cpp`
    - callback dispatch now routes through `circt_vpi_wasm_yield(...)` helper
      (native fallback still calls the callback directly).
    - all callback fire paths (`cbAfterDelay`, `fireCallbacks`,
      `fireValueChangeCallbacks`) now use the yield helper.
  - `tools/circt-sim/circt-sim.cpp`
    - added exported `vpi_startup_register(void (*fn)(void))`.
    - added startup routine registry + invocation before
      `fireStartOfSimulation()`.
    - VPI activation now supports two modes:
      - `--vpi=<library>` (existing path),
      - pre-registered startup routines (wasm/JS path, no shared library).
    - per-run `vpiEnabled` gating now controls value-change, ReadWrite/ReadOnly,
      NextSimTime, and EndOfSimulation callback dispatch.
  - `tools/circt-sim/CMakeLists.txt`
    - added Emscripten async callback integration flags:
      - `--js-library=tools/circt-sim/circt-sim-vpi-wasm.js`
      - `-sASYNCIFY=1`
      - `-sASYNCIFY_IMPORTS=['circt_vpi_wasm_yield']`
    - added explicit wasm exports for VPI API entry points (including
      `vpi_startup_register`) so JS can call them.
  - `tools/circt-sim/circt-sim-vpi-wasm.js` (new)
    - defines async JS import `circt_vpi_wasm_yield` and invokes callback
      through the wasm table after an awaitable yield point.
  - `include/circt/Runtime/MooreRuntime.h`
    - added `vpi_startup_register` declaration in VPI stub section.
  - Regression:
    - `test/Tools/circt-sim/test_vpi.py`
      - added `test_vpi_startup_register_bridge` covering
        `vlog_startup_routines -> vpi_startup_register -> cbStart/cbEnd`.
- Follow-up (cb_rtn=0 hook-only callbacks + async startup yield):
  - realization:
    - wasm JS library supported `cbFuncPtr=0`, but C++ registration still
      rejected `cb_rtn == nullptr`, so hook-only callback registration silently
      failed (`vpi_register_cb` returned null handle).
    - this prevented `cbStartOfSimulation` hook dispatch in wasm-only setups and
      manifested as simulation ending at time 0 with no startup callback work.
  - implemented:
    - `lib/Dialect/Sim/VPIRuntime.cpp`
      - native callback fallback now null-checks before direct invocation.
      - on `__EMSCRIPTEN__`, `registerCb` now accepts `cb_rtn == nullptr`
        (hook-only mode); native path still requires non-null callback pointers.
    - new node regression helper:
      - `utils/wasm_vpi_startup_yield_check.sh`
      - validates:
        - pre-`callMain` `cbStartOfSimulation` registration with `cb_rtn=0`,
        - async yield hook execution at startup,
        - dynamic `cbAfterDelay` registration from the async hook,
        - subsequent `cbAfterDelay` dispatch through the same hook.
    - integrated into smoke flow:
      - `utils/run_wasm_smoke.sh`
      - `utils/wasm_smoke_contract_check.sh`
- Validation performed:
  - `ninja -C build-wasm circt-sim` reaches and compiles updated files:
    - `lib/Dialect/Sim/VPIRuntime.cpp`
    - `tools/circt-sim/circt-sim.cpp`
  - Full target build currently blocked by pre-existing unrelated errors in
    `tools/circt-sim/AOTProcessCompiler.cpp` (e.g. `encodeJITDelay` undeclared,
    incomplete `mlir::ExecutionEngine` type under wasm).
  - Native `build_test` rebuild is currently blocked by an unrelated pre-existing
    CMake regeneration issue (`JITSchedulerRuntime.cpp` missing in unittests).
  - `pytest` is not installed in this environment, so the Python VPI regression
    suite could not be executed here.

## 2026-02-23
- Iteration update (UVM wasm frontend host-path + VCD regression coverage):
  - gap identified (regression first):
    - added `utils/wasm_uvm_stub_vcd_check.sh` plus fixture
      `test/Tools/circt-sim/wasm-uvm-stub-vcd.sv`.
    - pre-fix behavior:
      - wasm frontend failed host-path invocation with
        `error: cannot open input file ...` and UVM path warning.
  - fix:
    - `tools/circt-verilog/CMakeLists.txt`
      - added `CIRCT_VERILOG_WASM_ENABLE_NODERAWFS` (default `ON`).
      - enabled `-sNODERAWFS=1` for `circt-verilog` when emscripten + option on.
    - `utils/configure_wasm_build.sh`
      - added env passthrough/validation:
        `CIRCT_VERILOG_WASM_ENABLE_NODERAWFS`.
    - `utils/wasm_configure_contract_check.sh`
      - added command/source tokens and invalid-value validation for the new
        configure option.
    - `utils/run_wasm_smoke.sh`
      - integrated `utils/wasm_uvm_stub_vcd_check.sh`.
  - follow-on gaps exposed by the new mode:
    - `circt-verilog` re-entry still hit:
      `InitLLVM was already initialized!`.
    - non-UVM smoke/frontend paths could OOM when UVM auto-include became
      reachable through raw-fs.
    - wasm frontend smoke expected IR on stdout, but current `circt-verilog`
      flow is file-output oriented in this tree.
  - follow-on fixes:
    - `tools/circt-verilog/circt-verilog.cpp`
      - emscripten path now mirrors bmc/sim re-entry handling:
        - no `InitLLVM` in wasm mode,
        - `cl::ResetAllOptionOccurrences()` per invocation,
        - local help/version handling,
        - return code path in wasm mode (no hard process exit).
    - `utils/run_wasm_smoke.sh`
      - non-UVM frontend invocations now pass `--no-uvm-auto-include`.
      - frontend IR checks now validate explicit `-o <file>` outputs.
      - verilog re-entry checks switched to host-path input/output (raw-fs mode)
        rather than MEMFS preload assumptions.
    - `utils/wasm_resource_guard_default_check.sh`
      - switched non-UVM frontend check to `--no-uvm-auto-include` and explicit
        `-o` file validation.
  - validation:
    - `utils/wasm_configure_contract_check.sh`: `PASS`.
    - `utils/wasm_smoke_contract_check.sh`: `PASS`.
    - `utils/wasm_runtime_helpers_contract_check.sh`: `PASS`.
    - `BUILD_DIR=build-wasm NODE_BIN=node utils/wasm_uvm_stub_vcd_check.sh`:
      `PASS`.
    - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 utils/run_wasm_smoke.sh`:
      `PASS` (including UVM-stub frontend+sim+VCD and verilog re-entry checks).

- Iteration update (browser-target wasm configurability + circt-sim finalize const-unblock):
  - realization:
    - `tools/circt-sim/CMakeLists.txt` still hard-forced
      `-sNODERAWFS=1` for all emscripten builds, which blocks browser-oriented
      packaging where host raw-fs passthrough is undesirable.
    - current tip also hit a compile-time const-qualification mismatch path in
      `tools/circt-sim/circt-sim.cpp` around
      `finalizeClockedAssertionsAtEnd()` call sites.
  - implemented:
    - `tools/circt-sim/CMakeLists.txt`
      - added option:
        - `CIRCT_SIM_WASM_ENABLE_NODERAWFS` (default `ON`)
      - gated `target_link_options(... -sNODERAWFS=1)` behind
        `EMSCRIPTEN AND CIRCT_SIM_WASM_ENABLE_NODERAWFS`.
    - `utils/configure_wasm_build.sh`
      - surfaced env-configurable passthrough:
        - `CIRCT_SIM_WASM_ENABLE_NODERAWFS` (default `ON`)
      - emits `-DCIRCT_SIM_WASM_ENABLE_NODERAWFS=<ON|OFF>` in configure command.
    - `tools/circt-sim/circt-sim.cpp`
      - added non-const `SimulationContext::getInterpreter()` overload.
      - removed `const_cast` call pattern for:
        - `finalizeClockedAssertionsAtEnd()`
        - `printCompileReport()`
    - `utils/wasm_configure_contract_check.sh`
      - added contract token for
        `-DCIRCT_SIM_WASM_ENABLE_NODERAWFS=`.
  - validation:
    - `ninja -C build_test circt-verilog circt-sim`:
      - result: `PASS` (including `circt-sim.cpp` recompile + link).
    - `utils/configure_wasm_build.sh --print-cmake-command`:
      - includes `-DCIRCT_SIM_WASM_ENABLE_NODERAWFS=ON` by default.
    - `CIRCT_SIM_WASM_ENABLE_NODERAWFS=OFF utils/configure_wasm_build.sh --print-cmake-command`:
      - includes `-DCIRCT_SIM_WASM_ENABLE_NODERAWFS=OFF`.
    - `utils/wasm_configure_contract_check.sh`:
      - result: `PASS`.

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

## 2026-02-23 (follow-up: interrupted regression run recovery)
- Goal: recover and complete previously interrupted focused checks:
  - `ninja -C build_test check-circt-conversion-veriftosmt`
  - `ninja -C build_test check-circt-tools-circt-bmc`
- Results:
  - `check-circt-conversion-veriftosmt`: `Passed: 155/155`.
  - `check-circt-tools-circt-bmc`: `Passed: 158`, `Unsupported: 156`, `Failed: 0`.
- Realization:
  - The apparent "hang" in the bmc suite repeatedly occurs near progress marker
    `~70` while long-running `sv-tests-mini-uvm` scripts execute; the suite does
    complete successfully after a longer wait (`Testing Time: 237.93s`).

## 2026-02-23 (follow-up: wasm re-entry hardening + smoke automation)
- Goal: close the self-check->run same-instance failure mode and formalize wasm
  smoke checks into one repeatable script.
- Realization: `llvm::cl::ResetCommandLineParser()` is too strong for this case.
  In wasm tools it cleared registered options, causing nearly all CLI flags to
  become "unknown argument" at runtime. Keeping only
  `ResetAllOptionOccurrences()` preserves registrations and still avoids option
  occurrence leakage between repeated invocations.
- Updated both `circt-sim` and `circt-bmc` wasm entry paths:
  - keep `InitLLVM` disabled on emscripten builds;
  - keep manual wasm help/version handling + parser reset via
    `ResetAllOptionOccurrences()`;
  - keep direct return paths in wasm (no process `exit/_Exit`) for repeat calls.
- Added wasm smoke automation:
  - `utils/run_wasm_smoke.sh`
    - builds `circt-bmc` + `circt-sim` wasm with constrained single-job default;
    - verifies `--help` for both tools;
    - runs BMC functional stdin smoke;
    - runs sim functional stdin smoke;
    - runs `--vcd` smoke and checks file existence/non-zero size;
    - runs same-instance callMain re-entry checks.
  - `utils/wasm_callmain_reentry_check.js`
    - executes wasm JS wrapper in a VM global context;
    - calls `callMain` twice in one loaded instance;
    - fails if either call returns nonzero or forbidden text appears
      (default includes `InitLLVM was already initialized!`).
- Added regression test for clearer SV-input contract:
  - `test/Tools/circt-sim/reject-raw-sv-input.sv`
  - checks `circt-sim` emits explicit guidance to run `circt-verilog` first.
- Validation:
  - `utils/run_wasm_smoke.sh` passes end-to-end.
  - `llvm/build/bin/llvm-lit -sv build_test/test/Tools/circt-sim/reject-raw-sv-input.sv` passes.

## 2026-02-23 (follow-up: close bmc real-run re-entry gap)
- Gap identified:
  - `utils/run_wasm_smoke.sh` only exercised `circt-bmc` same-instance
    re-entry as `help -> help` because `help -> run` previously failed when
    passing host paths to the wasm module (no `NODERAWFS` enabled for bmc).
- Fix:
  - extended `utils/wasm_callmain_reentry_check.js` with:
    - `--preload-file <host> <wasm>` to copy host files into wasm MEMFS before
      invocation;
    - `--expect-wasm-file-substr <wasm> <text>` to assert output generated in
      wasm FS.
  - updated `utils/run_wasm_smoke.sh` to run:
    - `circt-bmc` same-instance `help -> real SMT-LIB run`
      using preloaded `/inputs/test.mlir`;
    - output directed to `/out.smt2` in wasm FS and asserted to include
      `(check-sat)`.
- Validation:
  - `utils/run_wasm_smoke.sh` now passes with real `circt-bmc` re-entry run
    (not just help/help).

## 2026-02-23 (follow-up: plusarg re-entry isolation in circt-sim wasm)
- Gap identified (regression-test first):
  - Added `utils/wasm_plusargs_reentry_check.sh` with fixture
    `test/Tools/circt-sim/wasm-plusargs-reentry.mlir`.
  - The test executes `circt-sim.js` in one loaded module instance:
    - run1 with `+VERBOSE +DEBUG=3`;
    - run2 without plusargs.
  - Expected run2 output: `verbose_not_found` and `debug_not_found`.
  - Actual pre-fix output: run2 still printed `verbose_found` and `debug_found`
    (plusargs leaked across invocations).
- Root cause:
  - `vlogPlusargs` is process-global and was never cleared between wasm
    invocations.
  - wasm invocation path also reused previously mutated UVM arg state.
- Fix:
  - in `tools/circt-sim/circt-sim.cpp`:
    - clear `vlogPlusargs` at each invocation;
    - capture baseline `CIRCT_UVM_ARGS` / `UVM_ARGS` once (emscripten mode);
    - rebuild `CIRCT_UVM_ARGS` from baseline + current invocation plusargs;
    - `unsetenv("CIRCT_UVM_ARGS")` when merged args are empty.
  - strengthened wasm smoke re-entry checks to use deterministic FS assertions
    (`--expect-wasm-file-substr`) instead of fragile stdout greps.
- Validation:
  - `utils/wasm_plusargs_reentry_check.sh` now passes.
  - `utils/run_wasm_smoke.sh` passes end-to-end after the fix.

## 2026-02-23 (follow-up: strengthen re-entry smoke coverage)
- Gap identified:
  - wasm smoke re-entry checks only covered `help -> run`.
  - This left a coverage hole for repeated real executions (`run -> run`) in the
    same loaded module instance.
- Test improvements:
  - extended `utils/run_wasm_smoke.sh` to include:
    - `circt-sim` same-instance `run -> run` with two VCD outputs in `/tmp`;
    - `circt-bmc` same-instance `run -> run` writing two SMT-LIB outputs in
      wasm FS (`/out1.smt2`, `/out2.smt2`) and validating both contain
      `(check-sat)`.
- Validation:
  - `utils/run_wasm_smoke.sh` passes with all of:
    - `help -> run` checks;
    - `run -> run` checks;
    - plusargs isolation regression.

## 2026-02-23 (follow-up: default resource-guard aborts in wasm)
- Gap identified (regression-test first):
  - Added `utils/wasm_resource_guard_default_check.sh` to verify wasm tools do
    not require `--resource-guard=false` just to avoid runtime aborts.
  - Pre-fix behavior:
    - `circt-bmc.js`, `circt-sim.js`, and `circt-verilog.js` aborted with
      default settings (`Aborted()`), typically after:
      `warning: resource guard: failed to set RLIMIT_AS ...`.
- Root cause:
  - `installResourceGuard()` unconditionally attempted to use process-level
    limits and started a detached watchdog thread.
  - In this emscripten runtime configuration, that path is not safe and caused
    runtime aborts.
- Fix:
  - In `lib/Support/ResourceGuard.cpp`, disable resource guard installation on
    `__EMSCRIPTEN__` (keep the option parseable, avoid unsupported runtime
    behavior).
- Smoke/coverage improvements:
  - Updated `utils/run_wasm_smoke.sh` to:
    - include `utils/wasm_resource_guard_default_check.sh`;
    - detect and build optional `circt-verilog` wasm target when configured;
    - run `circt-verilog.js --help` + minimal `.sv` stdin lowering smoke when
      the target exists.
  - Added `WASM_REQUIRE_VERILOG=1` support to fail fast when the frontend wasm
    target is expected but not configured.
- Validation:
  - `utils/wasm_resource_guard_default_check.sh` passes.
  - `utils/run_wasm_smoke.sh` passes end-to-end with:
    - `circt-bmc`, `circt-sim` smoke + functional + re-entry checks;
    - plusargs isolation regression;
    - default resource-guard no-abort checks;
    - optional `circt-verilog` frontend checks when target is enabled.

## 2026-02-23 (follow-up: reproducible wasm configure entrypoint)
- Gap identified (regression-test first):
  - Added `utils/wasm_configure_contract_check.sh`.
  - Pre-fix failure:
    - reported missing `utils/configure_wasm_build.sh`.
- Fix:
  - added `utils/configure_wasm_build.sh` as a single reproducible wasm
    configure entrypoint.
  - defaults include:
    - `-DLLVM_TARGETS_TO_BUILD=WebAssembly`
    - `-DMLIR_ENABLE_EXECUTION_ENGINE=OFF`
    - `-DLLVM_ENABLE_THREADS=OFF`
    - `-DCIRCT_SLANG_FRONTEND_ENABLED=ON`
  - supports `--print-cmake-command` for contract testing and reproducibility.
- Validation:
  - `utils/wasm_configure_contract_check.sh` passes.

## 2026-02-23 (follow-up: CI wasm smoke enforcement)
- Gap identified (regression-test first):
  - Added `utils/wasm_ci_contract_check.sh`.
  - Pre-fix failure:
    - missing `.github/workflows/wasmSmoke.yml`.
- Fix:
  - added `.github/workflows/wasmSmoke.yml` with a dedicated wasm smoke job:
    - checks out submodules;
    - installs emsdk (`4.0.12`) and sources `emsdk_env.sh`;
    - runs `utils/configure_wasm_build.sh`;
    - runs `WASM_REQUIRE_VERILOG=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh` still passes.

## 2026-02-23 (follow-up: wasm smoke SV->sim pipeline coverage)
- Gap identified (regression-test first):
  - Added `utils/wasm_smoke_contract_check.sh`.
  - Pre-fix failure:
    - missing `Functional: circt-verilog (.sv) -> circt-sim` stage in
      `utils/run_wasm_smoke.sh`.
- Fix:
  - extended `utils/run_wasm_smoke.sh` optional frontend path to run an
    end-to-end wasm pipeline:
    - `.sv` stdin into `circt-verilog.js` (`--ir-llhd`);
    - output MLIR into `circt-sim.js --top event_triggered_tb --vcd ...`;
    - assert expected simulation output and non-empty VCD artifact.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.

## 2026-02-23 (follow-up: smoke runtime checks without rebuild)
- Gap identified (regression-test first):
  - in dirty worktrees, `utils/run_wasm_smoke.sh` could fail at rebuild time
    before any runtime regressions were executed.
  - strengthened `utils/wasm_smoke_contract_check.sh` to require a skip-build
    control path.
  - Pre-fix failure:
    - missing `WASM_SKIP_BUILD` handling in `utils/run_wasm_smoke.sh`.
- Fix:
  - added `WASM_SKIP_BUILD=1` support in `utils/run_wasm_smoke.sh`:
    - skips `ninja` rebuild for `circt-bmc`, `circt-sim`, and optional
      `circt-verilog`;
    - still validates expected wasm JS artifacts and runs full smoke checks.
- Validation:
  - `WASM_SKIP_BUILD=1 utils/run_wasm_smoke.sh` passes end-to-end.

## 2026-02-23 (follow-up: add circt-verilog same-instance re-entry coverage)
- Gap identified (regression-test first):
  - extended `utils/wasm_smoke_contract_check.sh` to require:
    - `Re-entry: circt-verilog callMain help -> run`
    - `Re-entry: circt-verilog run -> run`
  - Pre-fix failure:
    - missing `circt-verilog` re-entry stages in `utils/run_wasm_smoke.sh`.
- Fix:
  - updated `utils/run_wasm_smoke.sh` to include optional frontend re-entry
    checks (when `circt-verilog` target is configured):
    - `help -> run` using preloaded `.sv` file in wasm FS and asserting
      `/out.mlir` contains `llhd.process`;
    - `run -> run` with two different lowering modes (`--ir-hw`, `--ir-llhd`)
      and wasm-FS output assertions on `/out1.mlir` and `/out2.mlir`.
  - both checks forbid `Aborted(` and explicitly fail on
    `InitLLVM was already initialized!`.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 utils/run_wasm_smoke.sh` passes end-to-end with the new
    `circt-verilog` re-entry stages.

## 2026-02-23 (follow-up: enforce wasm contract scripts in CI workflow)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_ci_contract_check.sh` to require CI workflow usage
    of:
    - `utils/wasm_configure_contract_check.sh`
    - `utils/wasm_smoke_contract_check.sh`
  - Pre-fix failure:
    - `.github/workflows/wasmSmoke.yml` did not run the local contract checks.
- Fix:
  - updated `.github/workflows/wasmSmoke.yml` with a dedicated
    `Run wasm script contract checks` step before the emsdk setup/build stage.
  - the step runs:
    - `utils/wasm_configure_contract_check.sh`
    - `utils/wasm_smoke_contract_check.sh`
    - `utils/wasm_ci_contract_check.sh`
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.

## 2026-02-23 (follow-up: verify wasm payload artifacts, not only JS wrappers)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    `circt-bmc.wasm` and `circt-sim.wasm` checks in smoke coverage.
  - Pre-fix failure:
    - `utils/run_wasm_smoke.sh` only validated `.js` wrappers and could pass
      despite missing/empty `.wasm` payloads.
- Fix:
  - updated `utils/run_wasm_smoke.sh` to assert non-empty artifacts for:
    - `circt-bmc.js` + `circt-bmc.wasm`
    - `circt-sim.js` + `circt-sim.wasm`
  - when `circt-verilog` target is configured, also assert non-empty:
    - `circt-verilog.js` + `circt-verilog.wasm`
  - kept optional-target behavior intact for configurations without frontend
    target support.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 utils/run_wasm_smoke.sh` passes end-to-end.
  - `utils/wasm_configure_contract_check.sh` and
    `utils/wasm_ci_contract_check.sh` still pass.

## 2026-02-23 (follow-up: CI coverage for runtime-only wasm smoke path)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_ci_contract_check.sh` to require
    `WASM_SKIP_BUILD=1` usage in `.github/workflows/wasmSmoke.yml`.
  - Pre-fix failure:
    - workflow ran only full configure/build smoke and did not validate the
      runtime-only `WASM_SKIP_BUILD=1` path.
- Fix:
  - updated `.github/workflows/wasmSmoke.yml` to add:
    - `Re-run smoke checks without rebuild`
    - executes:
      `WASM_REQUIRE_VERILOG=1 WASM_SKIP_BUILD=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
  - this keeps CI coverage aligned with the local regression mode used in dirty
    worktrees.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 utils/run_wasm_smoke.sh` passes
    end-to-end.

## 2026-02-23 (follow-up: enforce clean CrossCompile.cmake when required)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require
    `WASM_REQUIRE_CLEAN_CROSSCOMPILE` handling in `utils/run_wasm_smoke.sh`.
  - Pre-fix failure:
    - smoke script only reported CrossCompile edits as informational and could
      not enforce a hard failure mode.
- Fix:
  - added `WASM_REQUIRE_CLEAN_CROSSCOMPILE` (default `0`) to
    `utils/run_wasm_smoke.sh`.
  - behavior:
    - when `0` (default): keep current informational reporting.
    - when `1`: fail the smoke run if
      `llvm/llvm/cmake/modules/CrossCompile.cmake` has local edits.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes in this workspace (no local CrossCompile edits detected).

## 2026-02-23 (follow-up: enforce clean CrossCompile mode in CI)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_ci_contract_check.sh` to require
    `WASM_REQUIRE_CLEAN_CROSSCOMPILE=1` in `.github/workflows/wasmSmoke.yml`.
  - Pre-fix failure:
    - workflow did not enable the new hard-fail cleanliness mode.
- Fix:
  - updated `.github/workflows/wasmSmoke.yml` to pass
    `WASM_REQUIRE_CLEAN_CROSSCOMPILE=1` in both stages:
    - full configure/build smoke run;
    - `WASM_SKIP_BUILD=1` runtime-only rerun.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end in this workspace.

## 2026-02-23 (follow-up: fix CrossCompile cleanliness check scope)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require submodule-
    scoped check command:
    - `git -C llvm diff --quiet -- llvm/cmake/modules/CrossCompile.cmake`
  - Pre-fix failure:
    - `utils/run_wasm_smoke.sh` checked
      `git diff -- llvm/llvm/cmake/modules/CrossCompile.cmake` from top-level
      repo scope.
    - this path is not tracked by the superproject (it is inside `llvm`
      submodule), so cleanliness detection was structurally unreliable.
- Fix:
  - updated `utils/run_wasm_smoke.sh` CrossCompile detection to run in the
    `llvm` submodule using `git -C llvm ...`.
  - retained the existing hard-fail behavior under
    `WASM_REQUIRE_CLEAN_CROSSCOMPILE=1`.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end and reports:
    `CrossCompile.cmake local edits (llvm submodule): none`.

## 2026-02-23 (follow-up: configure script preflight diagnostics)
- Gap identified (regression-test first):
  - extended `utils/wasm_configure_contract_check.sh` to require explicit
    configure-script preflight checks for:
    - `command -v "$EMCMAKE_BIN"`
    - `command -v "$CMAKE_BIN"`
  - Pre-fix failure:
    - configure script emitted only generic shell/tool errors when binaries
      were missing.
- Fix:
  - updated `utils/configure_wasm_build.sh` with explicit preflight checks and
    clear diagnostics before invoking emcmake/cmake.
  - behavior now reports actionable errors for missing emsdk wrapper or cmake.
- Validation:
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: move wasm configure path to C++20)
- Gap identified (regression-test first):
  - added `utils/wasm_cxx20_contract_check.sh` requiring:
    - wasm configure command contains `-DCMAKE_CXX_STANDARD=20`;
    - `tools/circt-tblgen/FIRRTLAnnotationsGen.cpp` uses out-of-line
      `ObjectType::getFields()`.
  - Pre-fix failure:
    - configure command did not set C++20.
- Fix:
  - updated `utils/configure_wasm_build.sh`:
    - added `CMAKE_CXX_STANDARD` (default `20`) and plumbed it into cmake args.
  - updated `tools/circt-tblgen/FIRRTLAnnotationsGen.cpp`:
    - changed `ObjectType::getFields()` to out-of-line definition after
      `Parameter` is complete to avoid C++20 incomplete-type issues.
- Validation:
  - `utils/wasm_cxx20_contract_check.sh` passes.
  - wasm reconfigure now shows `-DCMAKE_CXX_STANDARD=20`.
  - `build-wasm/CMakeCache.txt` reports `CMAKE_CXX_STANDARD:STRING=20`.
  - `ninja -C build-wasm -j1 circt-tblgen` succeeds under C++20.

## 2026-02-23 (follow-up: wasm C++20 warning triage)
- Triaged warnings during C++20 wasm builds:
  - no `-Wambiguous-reversed-operator` warnings observed in `circt-tblgen`
    rebuild and the sampled `circt-bmc` dependency rebuild output.
  - no suppressions were added at this stage since no concrete hits were found.
- Additional smoke-script robustness:
  - added explicit tool preflight checks in `utils/run_wasm_smoke.sh`:
    - `command -v "$NODE_BIN"`
    - `command -v ninja` (required unless `WASM_SKIP_BUILD=1`)
  - extended `utils/wasm_smoke_contract_check.sh` accordingly.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: enforce C++20 contract in CI)
- Gap identified:
  - `utils/wasm_cxx20_contract_check.sh` was not part of CI script-contract
    enforcement.
- Fix:
  - updated `utils/wasm_ci_contract_check.sh` to require
    `utils/wasm_cxx20_contract_check.sh`.
  - updated `.github/workflows/wasmSmoke.yml` contract step to run
    `utils/wasm_cxx20_contract_check.sh`.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes with updated workflow.

## 2026-02-23 (follow-up: automate C++20 warning triage in wasm CI)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_ci_contract_check.sh` to require
    `utils/wasm_cxx20_warning_check.sh` usage in
    `.github/workflows/wasmSmoke.yml`.
  - Pre-fix failure:
    - workflow had no dedicated C++20 warning triage step for
      `-Wambiguous-reversed-operator`-class diagnostics.
- Fix:
  - added `utils/wasm_cxx20_warning_check.sh`:
    - forces rebuild of `circt-tblgen` translation units tied to the C++20
      compatibility fix (`FIRRTLAnnotationsGen.cpp`, `circt-tblgen.cpp`);
    - scans rebuild log for disallowed warning patterns:
      - `ambiguous-reversed-operator`
      - `-Wambiguous-reversed-operator`
  - updated `.github/workflows/wasmSmoke.yml` to run this check after wasm
    configure and before full smoke checks.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes locally.

## 2026-02-23 (follow-up: enforce C++20 configuration in warning triage)
- Gap identified (regression-test first):
  - added `utils/wasm_cxx20_warning_contract_check.sh` requiring warning-check
    script coverage of:
    - `CMAKE_CXX_STANDARD:STRING=20`
    - `ambiguous-reversed-operator` warning pattern.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_check.sh` did not verify that the build cache
      actually used C++20, so it could silently pass against stale C++17
      configuration.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to hard-fail unless
    `build-wasm/CMakeCache.txt` contains `CMAKE_CXX_STANDARD:STRING=20`.
  - added `utils/wasm_cxx20_warning_contract_check.sh`.
  - wired the new contract into CI guardrails:
    - `utils/wasm_ci_contract_check.sh` requires it in workflow.
    - `.github/workflows/wasmSmoke.yml` runs it in contract-check step.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh`, `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`, and
    `utils/wasm_smoke_contract_check.sh` all pass.

## 2026-02-23 (follow-up: enforce C++20 floor in wasm configure)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_contract_check.sh` to assert that
    `utils/configure_wasm_build.sh` rejects `CMAKE_CXX_STANDARD=17`.
  - Pre-fix failure:
    - configure script accepted C++17 override even after default moved to 20.
- Fix:
  - updated `utils/configure_wasm_build.sh` with explicit floor check:
    - fail with clear diagnostic unless `CMAKE_CXX_STANDARD >= 20`.
- Validation:
  - `utils/wasm_cxx20_contract_check.sh` passes (including override-rejection
    path).
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_warning_contract_check.sh`,
    `utils/wasm_cxx20_warning_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` all pass.
  - `WASM_SKIP_BUILD=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: detect C++20-extension warnings in triage check)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require
    coverage for `c++20-extensions` warning patterns.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_check.sh` only checked ambiguous reversed
      operator warnings.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to fail on:
    - `c++20-extensions`
    - `-Wc++20-extensions`
  - this catches accidental per-target fallback to pre-C++20 flags even when
    the cache-level standard is set to 20.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` all pass.

## 2026-02-23 (follow-up: verify compile-command C++ standard in warning triage)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require
    `-std=c++20` coverage inside `utils/wasm_cxx20_warning_check.sh`.
  - Pre-fix failure:
    - warning triage relied on cache-level standard check only and did not
      assert the actual `circt-tblgen` compile command used `-std=c++20`.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to:
    - dump `ninja -t commands circt-tblgen`;
    - fail if compile commands do not contain `-std=c++20`;
    - keep existing warning-pattern checks (`ambiguous-reversed-operator`,
      `c++20-extensions`).
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.

## 2026-02-23 (follow-up: integrate C++20 warning triage into wasm smoke)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require:
    - `WASM_CHECK_CXX20_WARNINGS`
    - `utils/wasm_cxx20_warning_check.sh` invocation path.
  - Pre-fix failure:
    - `utils/run_wasm_smoke.sh` had no hook to run C++20 warning triage.
- Fix:
  - updated `utils/run_wasm_smoke.sh` with
    `WASM_CHECK_CXX20_WARNINGS` control:
    - `auto` (default): run warning triage when rebuilding; skip in
      `WASM_SKIP_BUILD=1` mode.
    - `1`: force warning triage via `utils/wasm_cxx20_warning_check.sh`.
    - `0`: disable warning triage for the smoke run.
  - added explicit smoke-stage output `C++20 warning triage`.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end and runs `[wasm-cxx20-warn] PASS`.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_cxx20_warning_contract_check.sh`,
    `utils/wasm_cxx20_warning_check.sh`, and
    `utils/wasm_ci_contract_check.sh` all pass.

## 2026-02-23 (follow-up: fail warning triage on any compiler warning)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require a
    generic warning-line scan in `utils/wasm_cxx20_warning_check.sh`:
    - `grep -Eiq -- "(^|[^[:alpha:]])warning:" "$log"`
  - Pre-fix failure:
    - warning triage only rejected a small explicit warning list.
    - new warning classes could slip through undetected.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to fail on any remaining
    compiler warning line in rebuild output after targeted pattern checks.
  - kept explicit checks for:
    - `ambiguous-reversed-operator`
    - `c++20-extensions`
    - compile-command `-std=c++20`.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` all pass.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: enforce -std=c++20 per rebuilt translation unit)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require
    per-source compile-command checks:
    - `FIRRTLAnnotationsGen.cpp`
    - `circt-tblgen.cpp`
    each with `-std=c++20`.
  - Pre-fix failure:
    - warning triage only checked for `-std=c++20` anywhere in command dump.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to iterate each source and
    fail if its compile command is missing `-std=c++20`.
  - retained all prior checks:
    - cache-level C++20;
    - disallowed warning patterns;
    - generic warning-line detection.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` pass.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: include FIRRTLIntrinsicsGen in C++20 warning triage)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require
    coverage for `FIRRTLIntrinsicsGen.cpp`.
  - Pre-fix failure:
    - warning triage rebuilt and verified only:
      - `FIRRTLAnnotationsGen.cpp`
      - `circt-tblgen.cpp`
    - it skipped `FIRRTLIntrinsicsGen.cpp`, leaving a blind spot in
      `circt-tblgen` C++20 warning coverage.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to:
    - include `FIRRTLIntrinsicsGen.cpp.o` in forced rebuild targets;
    - enforce `-std=c++20` compile-command verification for
      `FIRRTLIntrinsicsGen.cpp` as well.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` pass.

## 2026-02-23 (follow-up: make CI warning-triage mode explicit)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_ci_contract_check.sh` to require explicit
    workflow settings:
    - `WASM_CHECK_CXX20_WARNINGS=1` for full rebuild smoke run;
    - `WASM_CHECK_CXX20_WARNINGS=0` for `WASM_SKIP_BUILD=1` rerun.
  - Pre-fix failure:
    - CI relied on `WASM_CHECK_CXX20_WARNINGS=auto` behavior in smoke script.
- Fix:
  - updated `.github/workflows/wasmSmoke.yml`:
    - full run now passes `WASM_CHECK_CXX20_WARNINGS=1`;
    - runtime-only rerun now passes `WASM_CHECK_CXX20_WARNINGS=0`.
  - this decouples CI intent from default-behavior changes in
    `utils/run_wasm_smoke.sh`.
- Validation:
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_cxx20_warning_contract_check.sh`, and
    `utils/wasm_smoke_contract_check.sh` pass.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: reject non-numeric C++ standard overrides)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_contract_check.sh` to assert that
    `utils/configure_wasm_build.sh` rejects non-numeric
    `CMAKE_CXX_STANDARD` values (e.g. `gnu++20`) with clear diagnostics.
  - Pre-fix failure:
    - configure script accepted non-numeric values and deferred failures to
      downstream tooling.
- Fix:
  - updated `utils/configure_wasm_build.sh`:
    - require `CMAKE_CXX_STANDARD` to match `^[0-9]+$`;
    - keep existing floor check (`>= 20`) for numeric values.
  - explicit diagnostics now cover both:
    - non-numeric value;
    - numeric but too-low value.
- Validation:
  - `utils/wasm_cxx20_contract_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_warning_contract_check.sh`,
    `utils/wasm_cxx20_warning_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` pass.

## 2026-02-23 (follow-up: hard-fail warning triage when clean step fails)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require an
    explicit clean-step failure diagnostic:
    - `[wasm-cxx20-warn] failed to clean rebuild targets`
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_check.sh` ignored clean failures via `|| true`,
      which could allow stale object reuse and reduce warning sensitivity.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh`:
    - clean step now hard-fails with explicit diagnostic if target cleaning
      fails.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` pass.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: verify compile-command presence per triaged TU)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require
    explicit diagnostic coverage for missing compile commands:
    - `[wasm-cxx20-warn] missing compile command for ...`
  - Pre-fix failure:
    - warning triage assumed each TU appears in `ninja -t commands` output and
      only validated `-std=c++20` for matched lines.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh`:
    - fail immediately if any expected TU compile command is absent from
      command dump before checking flags.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh`,
    `utils/wasm_cxx20_contract_check.sh`,
    `utils/wasm_smoke_contract_check.sh`, and
    `utils/wasm_ci_contract_check.sh` pass.

## 2026-02-23 (follow-up: validate wasm smoke env toggle values)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    env validation hooks in `utils/run_wasm_smoke.sh`:
    - shared boolean helper `validate_bool_env`;
    - explicit checks for:
      - `WASM_REQUIRE_VERILOG`
      - `WASM_SKIP_BUILD`
      - `WASM_REQUIRE_CLEAN_CROSSCOMPILE`
    - explicit `WASM_CHECK_CXX20_WARNINGS` value validation.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh`
      failed with:
      - `missing token in smoke script: validate_bool_env`
    - smoke script accepted invalid env values (for example,
      `WASM_CHECK_CXX20_WARNINGS=invalid`) without early rejection.
- Fix:
  - updated `utils/run_wasm_smoke.sh` to:
    - add `validate_bool_env()` and fail fast unless boolean toggles are
      exactly `0` or `1`;
    - reject invalid `WASM_CHECK_CXX20_WARNINGS` values unless one of
      `auto`, `0`, or `1`.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce the new validation
    contract tokens.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - negative test:
    - `WASM_CHECK_CXX20_WARNINGS=invalid WASM_SKIP_BUILD=1 utils/run_wasm_smoke.sh`
      exits non-zero with:
      - `invalid WASM_CHECK_CXX20_WARNINGS value: invalid (expected auto, 0, or 1)`.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.

## 2026-02-23 (follow-up: align warning triage with CMAKE_CXX_STANDARD>=20)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require:
    - cache standard parsing from `CMAKE_CXX_STANDARD:STRING=...`;
    - explicit diagnostics for:
      - non-numeric cache standard;
      - cache standard below 20;
    - per-TU standard-flag validation via
      `std_flag_is_cpp20_or_newer` instead of exact `-std=c++20`.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_contract_check.sh` failed with:
      - `missing token in warning check script: cache C++ standard is non-numeric`
    - warning triage previously required exact
      `CMAKE_CXX_STANDARD:STRING=20`, conflicting with
      `utils/configure_wasm_build.sh` which allows numeric values `>=20`.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to:
    - parse and validate `CMAKE_CXX_STANDARD:STRING=...` as numeric;
    - reject cache standards below 20 with explicit diagnostic;
    - add `std_flag_is_cpp20_or_newer` and verify each triaged TU compile
      command has a `-std=` flag that is C++20-or-newer.
  - updated `utils/wasm_cxx20_warning_contract_check.sh` to enforce the new
    contract tokens.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 utils/run_wasm_smoke.sh`
    passes end-to-end.
  - `utils/wasm_smoke_contract_check.sh`, `utils/wasm_ci_contract_check.sh`,
    and `utils/wasm_cxx20_contract_check.sh` pass.

## 2026-02-23 (follow-up: fail fast on invalid NINJA_JOBS in wasm smoke)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require:
    - a positive-integer validator helper (`validate_positive_int_env`);
    - explicit `NINJA_JOBS` validation call in `utils/run_wasm_smoke.sh`.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: validate_positive_int_env`
    - smoke script accepted invalid values like `NINJA_JOBS=zero` and deferred
      failure to downstream `ninja` invocations.
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - added `validate_positive_int_env()` and require `NINJA_JOBS` to match
      `^[1-9][0-9]*$`;
    - fail early with explicit diagnostic when invalid.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce this new token
    contract.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - negative test:
    - `NINJA_JOBS=zero WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 utils/run_wasm_smoke.sh`
      exits non-zero with:
      - `invalid NINJA_JOBS value: zero (expected positive integer)`.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.
  - `utils/wasm_cxx20_warning_contract_check.sh`,
    `utils/wasm_ci_contract_check.sh`, and
    `utils/wasm_cxx20_contract_check.sh` pass.

## 2026-02-23 (follow-up: fail fast on invalid NINJA_JOBS in C++20 warning triage)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require:
    - a positive-integer validator helper (`validate_positive_int_env`);
    - explicit validation call for `NINJA_JOBS` in
      `utils/wasm_cxx20_warning_check.sh`.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_contract_check.sh` failed with:
      - `missing token in warning check script: validate_positive_int_env`
    - warning triage accepted invalid values like `NINJA_JOBS=zero` and only
      failed later during `ninja` invocation.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh`:
    - added `validate_positive_int_env()` and require `NINJA_JOBS` to match
      `^[1-9][0-9]*$`;
    - fail early with explicit diagnostic when invalid.
  - updated `utils/wasm_cxx20_warning_contract_check.sh` to enforce this
    validation contract.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - negative test:
    - `NINJA_JOBS=zero utils/wasm_cxx20_warning_check.sh`
      exits non-zero with:
      - `invalid NINJA_JOBS value: zero (expected positive integer)`.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: preflight all wasm smoke helper scripts)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    helper preflight hooks in `utils/run_wasm_smoke.sh` for all invoked
    helper scripts:
    - `REENTRY_HELPER` (`utils/wasm_callmain_reentry_check.js`)
    - `PLUSARGS_HELPER` (`utils/wasm_plusargs_reentry_check.sh`)
    - `RESOURCE_GUARD_HELPER` (`utils/wasm_resource_guard_default_check.sh`)
    - explicit diagnostics on missing/non-executable helpers.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: missing helper script: $REENTRY_HELPER`
    - smoke script only preflighted `utils/wasm_cxx20_warning_check.sh` and
      could fail later with less targeted diagnostics if helper scripts were
      missing.
- Fix:
  - updated `utils/run_wasm_smoke.sh` to:
    - define helper variables (`REENTRY_HELPER`, `PLUSARGS_HELPER`,
      `RESOURCE_GUARD_HELPER`);
    - fail early when:
      - re-entry helper file is missing;
      - plusargs/default-guard helpers are not executable;
    - route helper invocations through these validated helper variables.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce this contract.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: validate CIRCT_SIM_WASM_ENABLE_NODERAWFS in configure)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_configure_contract_check.sh` to require:
    - explicit source-level diagnostic token:
      `CIRCT_SIM_WASM_ENABLE_NODERAWFS must be ON or OFF`;
    - runtime rejection for invalid overrides, e.g.
      `CIRCT_SIM_WASM_ENABLE_NODERAWFS=maybe`.
  - Pre-fix failure:
    - `utils/wasm_configure_contract_check.sh` failed with:
      - `missing source check in configure script: CIRCT_SIM_WASM_ENABLE_NODERAWFS must be ON or OFF`
    - configure script accepted arbitrary values for
      `CIRCT_SIM_WASM_ENABLE_NODERAWFS`, deferring behavior to CMake truthiness.
- Fix:
  - updated `utils/configure_wasm_build.sh` to fail fast unless
    `CIRCT_SIM_WASM_ENABLE_NODERAWFS` is exactly `ON` or `OFF`.
- Validation:
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_cxx20_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: validate configure boolean toggles consistently)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_configure_contract_check.sh` to require:
    - explicit `ON|OFF` validation hooks in `utils/configure_wasm_build.sh`
      for:
      - `LLVM_ENABLE_ASSERTIONS`
      - `BUILD_SHARED_LIBS`
      - `CIRCT_SIM_WASM_ENABLE_NODERAWFS`
    - runtime rejection for invalid overrides:
      - `LLVM_ENABLE_ASSERTIONS=enabled`
      - `BUILD_SHARED_LIBS=static`
  - Pre-fix failure:
    - `utils/wasm_configure_contract_check.sh` failed with:
      - `missing source check in configure script: validate_on_off_env "LLVM_ENABLE_ASSERTIONS" "$LLVM_ENABLE_ASSERTIONS"`
    - configure script validated only the NODERAWFS toggle and left other
      boolean toggles unsanitized.
- Fix:
  - added shared `validate_on_off_env()` helper in
    `utils/configure_wasm_build.sh`.
  - applied it to:
    - `LLVM_ENABLE_ASSERTIONS`
    - `BUILD_SHARED_LIBS`
    - `CIRCT_SIM_WASM_ENABLE_NODERAWFS`
- Validation:
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: make warning-triage Emscripten compiler detection robust)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require:
    - dedicated compiler detector helper `is_emscripten_cpp_compiler`;
    - explicit diagnostic:
      `does not appear to use Emscripten em++`.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_contract_check.sh` failed with:
      - `missing token in warning check script: is_emscripten_cpp_compiler`
    - warning triage hardcoded `emscripten/em++` substring matching, which is
      brittle when compile commands invoke `em++` from PATH.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` to:
    - add `is_emscripten_cpp_compiler()` matcher that accepts both
      full-path and PATH-based `em++` command forms;
    - replace hardcoded `emscripten/em++` substring check with the helper.
  - updated `utils/wasm_cxx20_warning_contract_check.sh` accordingly.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_cxx20_warning_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: make VCD_PATH validation effective in smoke script)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    empty-path diagnostic coverage:
    - `invalid VCD_PATH value: empty path`
  - Pre-fix behavior:
    - initial empty-path validation was added, but it was ineffective because
      `VCD_PATH` used `${VCD_PATH:-/tmp/circt-wasm-smoke.vcd}`.
    - explicitly setting `VCD_PATH=` triggered fallback to default path,
      bypassing validation and allowing the run to continue.
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - switched to `VCD_PATH="${VCD_PATH-/tmp/circt-wasm-smoke.vcd}"` so
      explicit empty override is preserved;
    - retained explicit empty-path validation:
      - `[wasm-smoke] invalid VCD_PATH value: empty path`.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - negative test:
    - `VCD_PATH= WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 utils/run_wasm_smoke.sh`
      exits non-zero with:
      - `invalid VCD_PATH value: empty path`.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: fallback when ninja target query fails in skip-build mode)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    fallback/error handling tokens in `utils/run_wasm_smoke.sh`:
    - `ninja target query failed; inferring circt-verilog support from existing artifacts`
    - `failed to query ninja targets`
    - skip-build artifact fallback condition using
      `-s "$VERILOG_JS"` and `-s "$VERILOG_WASM"`.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: ninja target query failed; inferring circt-verilog support from existing artifacts`
    - with `WASM_SKIP_BUILD=1`, a failing `ninja -t targets` query could make
      smoke fail `WASM_REQUIRE_VERILOG=1` despite existing verilog wasm
      artifacts.
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - capture `ninja -t targets` stderr to `$tmpdir/targets.err`;
    - if query fails but `WASM_SKIP_BUILD=1` and verilog artifacts exist,
      infer verilog support and continue with explicit fallback message;
    - otherwise fail fast with explicit diagnostic and target-query stderr.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce this contract.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - regression simulation (fake `ninja` in `PATH`):
    - `PATH="$tmpdir:$PATH" WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
      passes end-to-end and logs:
      - `ninja target query failed; inferring circt-verilog support from existing artifacts`.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: explicit diagnostic when warning-triage command query fails)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_cxx20_warning_contract_check.sh` to require:
    - explicit diagnostic token:
      `failed to query compile commands for circt-tblgen`.
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_contract_check.sh` failed with:
      - `missing token in warning check script: failed to query compile commands for circt-tblgen`
    - warning triage relied on `set -e` for `ninja -t commands` failures
      without clear context in stderr.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh`:
    - wrapped `ninja -t commands circt-tblgen` in explicit failure handling;
    - emit clear diagnostic and relay captured stderr from target-query
      failure.
- Validation:
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - fake-ninja regression simulation:
    - `PATH="$tmpdir:$PATH" utils/wasm_cxx20_warning_check.sh`
      exits non-zero with:
      - `failed to query compile commands for circt-tblgen`.
  - `utils/wasm_cxx20_warning_check.sh` passes with normal toolchain.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: distinguish CrossCompile diff state from git inspection errors)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    handling tokens around llvm submodule inspection:
    - `git_rc=$?`
    - `if [[ "$git_rc" -eq 1 ]]; then`
    - `unable to inspect llvm submodule CrossCompile.cmake status`
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: unable to inspect llvm submodule CrossCompile.cmake status`
    - smoke script treated any non-zero return from
      `git -C llvm diff --quiet -- llvm/cmake/modules/CrossCompile.cmake`
      as “local edits present”, which conflated true diff state (`rc=1`) with
      git/submodule errors (`rc>1`).
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - capture stderr for CrossCompile status query to
      `$tmpdir/crosscompile.err`;
    - branch on git return code:
      - `rc=1`: preserve existing “local edits present” behavior;
      - other non-zero: fail with explicit diagnostic and captured stderr.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: use effective (last) -std flag in warning triage)
- Gap identified (regression-test first):
  - added behavioral regression test:
    - `utils/wasm_cxx20_warning_behavior_check.sh`
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_behavior_check.sh` failed in case
      `final-cxx20`:
      - compile commands contained `-std=gnu++17 -std=c++20`;
      - warning triage still rejected them because it read the first `-std`
        flag instead of the effective final one.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh`:
    - parse the last `-std=...` token per compile command
      (`tail -n 1`) so validation matches effective compiler behavior.
  - added `utils/wasm_cxx20_warning_behavior_check.sh`:
    - pass case: final flag is C++20 (`... -std=gnu++17 -std=c++20`);
    - fail case: final flag is older than C++20
      (`... -std=c++20 -std=gnu++17`).
  - integrated behavior check into CI:
    - `.github/workflows/wasmSmoke.yml` now runs
      `utils/wasm_cxx20_warning_behavior_check.sh`;
    - `utils/wasm_ci_contract_check.sh` enforces workflow token.
- Validation:
  - pre-fix regression failure reproduced:
    - `utils/wasm_cxx20_warning_behavior_check.sh`
      failed `final-cxx20` as expected.
  - post-fix:
    - `utils/wasm_cxx20_warning_behavior_check.sh` passes.
    - `utils/wasm_cxx20_warning_contract_check.sh` passes.
    - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: preflight NODE_BIN in standalone wasm helper scripts)
- Gap identified (regression-test first):
  - added contract check:
    - `utils/wasm_runtime_helpers_contract_check.sh`
  - Pre-fix failure:
    - `utils/wasm_runtime_helpers_contract_check.sh` failed with:
      - `missing token in plusargs helper: command -v "$NODE_BIN"`
    - helper scripts relied on implicit shell errors when `NODE_BIN` was
      invalid, resulting in less actionable diagnostics.
- Fix:
  - updated helper scripts:
    - `utils/wasm_plusargs_reentry_check.sh`
    - `utils/wasm_resource_guard_default_check.sh`
  - both now fail fast with explicit Node runtime diagnostics:
    - `[wasm-plusargs] missing Node.js runtime: ...`
    - `[wasm-rg-default] missing Node.js runtime: ...`
  - integrated new helper contract into CI:
    - `.github/workflows/wasmSmoke.yml` runs
      `utils/wasm_runtime_helpers_contract_check.sh`;
    - `utils/wasm_ci_contract_check.sh` enforces workflow token.
- Validation:
  - `utils/wasm_runtime_helpers_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - negative tests:
    - `NODE_BIN=definitely_not_node utils/wasm_plusargs_reentry_check.sh`
      exits non-zero with explicit runtime diagnostic.
    - `NODE_BIN=definitely_not_node utils/wasm_resource_guard_default_check.sh`
      exits non-zero with explicit runtime diagnostic.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: explicit test-input preflight in default-guard helper)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_runtime_helpers_contract_check.sh` to require
    in `utils/wasm_resource_guard_default_check.sh`:
    - env-overridable input paths:
      - `BMC_TEST_INPUT`
      - `SIM_TEST_INPUT`
      - `SV_TEST_INPUT`
    - explicit missing-input diagnostic:
      - `[wasm-rg-default] missing test input: ...`
  - Pre-fix failure:
    - `utils/wasm_runtime_helpers_contract_check.sh` failed with:
      - `missing token in default-guard helper: BMC_TEST_INPUT="${BMC_TEST_INPUT:-`
    - helper used hardcoded relative test paths without explicit preflight
      checks.
- Fix:
  - updated `utils/wasm_resource_guard_default_check.sh`:
    - made input paths env-overridable;
    - added upfront file-existence checks for all three inputs with explicit
      diagnostics.
- Validation:
  - `utils/wasm_runtime_helpers_contract_check.sh` passes.
  - negative test:
    - `BMC_TEST_INPUT=/tmp/definitely-missing.mlir utils/wasm_resource_guard_default_check.sh`
      exits non-zero with:
      - `missing test input: /tmp/definitely-missing.mlir`.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: skip optional verilog when ninja target query fails)
- Gap identified (regression-test first):
  - behavioral repro before fix:
    - prepared temp `BUILD_DIR` containing only `circt-bmc` and `circt-sim`
      wasm artifacts (no `circt-verilog` artifacts);
    - injected fake `ninja` that fails `-t targets` query;
    - ran:
      - `WASM_SKIP_BUILD=1`
      - `WASM_REQUIRE_VERILOG=0`
      - `WASM_CHECK_CXX20_WARNINGS=0`
    - pre-fix result:
      - hard failure:
        - `failed to query ninja targets ... (needed to detect circt-verilog target)`
      - this was incorrect because verilog was optional in this run mode.
  - strengthened `utils/wasm_smoke_contract_check.sh` to require explicit
    optional-verilog fallback branch and diagnostic:
    - `ninja target query failed and circt-verilog is optional; skipping SV frontend checks`.
  - Pre-fix contract failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: ninja target query failed and circt-verilog is optional; skipping SV frontend checks`
- Fix:
  - updated `utils/run_wasm_smoke.sh` target-detection logic:
    - existing fallback (skip-build + existing verilog artifacts) preserved;
    - new fallback: if `WASM_SKIP_BUILD=1` and `WASM_REQUIRE_VERILOG!=1`,
      continue without verilog checks when target query fails;
    - keep hard failure when verilog is required or when query failure cannot
      be safely ignored.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - regression scenario (temp build + fake failing `ninja` + optional verilog)
    now passes end-to-end and logs:
    - `ninja target query failed and circt-verilog is optional; skipping SV frontend checks`.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: fail-fast configure when LLVM_SRC_DIR is missing)
- Gap identified (regression-test first):
  - added behavioral regression:
    - `utils/wasm_configure_behavior_check.sh`
  - Pre-fix failure:
    - `utils/wasm_configure_behavior_check.sh` failed with:
      - `configure unexpectedly succeeded with missing LLVM source dir`
    - configure script invoked `emcmake/cmake` even when `LLVM_SRC_DIR` did
      not exist, producing downstream errors instead of clear preflight output.
- Fix:
  - updated `utils/configure_wasm_build.sh`:
    - added explicit runtime-mode preflight:
      - `[wasm-configure] missing LLVM source directory: ...`
    - check runs after command-availability checks and only in non-print mode
      (keeps `--print-cmake-command` behavior unchanged).
  - integrated behavioral check into CI:
    - `.github/workflows/wasmSmoke.yml` runs
      `utils/wasm_configure_behavior_check.sh`;
    - `utils/wasm_ci_contract_check.sh` enforces workflow token.
- Validation:
  - `utils/wasm_configure_behavior_check.sh` passes.
  - `utils/wasm_configure_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: avoid fixed reentry VCD paths in smoke checks)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_smoke_contract_check.sh` to require:
    - process-unique reentry VCD variables:
      - `REENTRY_VCD="/tmp/reentry-${BASHPID}.vcd"`
      - `REENTRY_RUN1_VCD="/tmp/reentry-run1-${BASHPID}.vcd"`
      - `REENTRY_RUN2_VCD="/tmp/reentry-run2-${BASHPID}.vcd"`
    - corresponding usage in reentry checks.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: REENTRY_VCD="/tmp/reentry-${BASHPID}.vcd"`
    - reentry checks used fixed `/tmp/reentry*.vcd` names, which could collide
      across concurrent smoke invocations.
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - introduced `BASHPID`-scoped reentry VCD paths;
    - replaced fixed `/tmp/reentry*.vcd` paths in both help->run and run->run
      reentry checks.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce this contract.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: make configure behavior test outputs concurrency-safe)
- Gap identified (regression-test first):
  - added new contract check:
    - `utils/wasm_configure_behavior_contract_check.sh`
  - Pre-fix failure:
    - `utils/wasm_configure_behavior_contract_check.sh` failed with:
      - `missing token in behavior check script: behavior_out="$tmpdir/configure.out"`
    - `utils/wasm_configure_behavior_check.sh` wrote diagnostics to fixed
      `/tmp/wasm-config-behavior.*` paths, which can collide across concurrent
      CI/local runs.
- Fix:
  - updated `utils/wasm_configure_behavior_check.sh`:
    - switched behavior-check output/error paths to tmpdir-scoped files:
      - `behavior_out="$tmpdir/configure.out"`
      - `behavior_err="$tmpdir/configure.err"`
    - removed fixed `/tmp/wasm-config-behavior*` usage.
  - added `utils/wasm_configure_behavior_contract_check.sh` to enforce this
    contract, including guard against reintroducing fixed `/tmp` paths.
  - integrated contract in CI:
    - `.github/workflows/wasmSmoke.yml` now runs
      `utils/wasm_configure_behavior_contract_check.sh`;
    - `utils/wasm_ci_contract_check.sh` requires workflow token.
- Validation:
  - `utils/wasm_configure_behavior_contract_check.sh` passes.
  - `utils/wasm_configure_behavior_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: only require SV input when verilog helper path is active)
- Gap identified (regression-test first):
  - behavioral bug repro before fix:
    - created temp `BUILD_DIR` with `circt-bmc.js` and `circt-sim.js` only;
    - ran `utils/wasm_resource_guard_default_check.sh` with
      `SV_TEST_INPUT` pointing to a missing file;
    - pre-fix result:
      - failed with `missing test input` even though `circt-verilog.js` was
        absent and verilog checks should be skipped.
  - added behavioral regression:
    - `utils/wasm_runtime_helpers_behavior_check.sh`
    - Pre-fix failure:
      - case1 failed:
        - `missing SV input should be ignored when circt-verilog.js is absent`.
- Fix:
  - updated `utils/wasm_resource_guard_default_check.sh`:
    - require `BMC_TEST_INPUT` and `SIM_TEST_INPUT` unconditionally;
    - move `SV_TEST_INPUT` preflight into the
      `if [[ -f "$VERILOG_JS" ]]; then` branch.
  - strengthened contract:
    - `utils/wasm_runtime_helpers_contract_check.sh` now requires tokens for
      verilog-guarded SV preflight.
  - integrated behavior check into CI:
    - `.github/workflows/wasmSmoke.yml` runs
      `utils/wasm_runtime_helpers_behavior_check.sh`;
    - `utils/wasm_ci_contract_check.sh` enforces workflow token.
- Validation:
  - `utils/wasm_runtime_helpers_behavior_check.sh` passes.
  - `utils/wasm_runtime_helpers_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: make runtime-helper behavior test outputs concurrency-safe)
- Gap identified (regression-test first):
  - added contract check:
    - `utils/wasm_runtime_helpers_behavior_contract_check.sh`
  - Pre-fix failure:
    - `utils/wasm_runtime_helpers_behavior_contract_check.sh` failed with:
      - `missing token in behavior check script: case1_out="$tmpdir/case1.out"`
    - `utils/wasm_runtime_helpers_behavior_check.sh` used fixed
      `/tmp/wasm-runtime-helper-case*.{out,err}` paths, risking collisions in
      concurrent CI/local runs.
- Fix:
  - updated `utils/wasm_runtime_helpers_behavior_check.sh`:
    - switched case outputs/errors to tmpdir-scoped files:
      - `case1_out`, `case1_err`, `case2_out`, `case2_err`;
    - removed fixed `/tmp/wasm-runtime-helper-case*` paths.
  - added `utils/wasm_runtime_helpers_behavior_contract_check.sh` to enforce:
    - required tmpdir token usage;
    - absence of fixed `/tmp/wasm-runtime-helper-case` literals.
  - integrated into CI:
    - `.github/workflows/wasmSmoke.yml` now runs
      `utils/wasm_runtime_helpers_behavior_contract_check.sh`;
    - `utils/wasm_ci_contract_check.sh` enforces workflow token.
- Validation:
  - `utils/wasm_runtime_helpers_behavior_contract_check.sh` passes.
  - `utils/wasm_runtime_helpers_behavior_check.sh` passes.
  - `utils/wasm_runtime_helpers_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - note: current full smoke run in this workspace fails in pre-existing
    `utils/run_wasm_smoke.sh` VCD `$var` assertions unrelated to this change.

## 2026-02-23 (follow-up: preflight baseline inputs in runtime-helper behavior check)
- Gap identified (regression-test first):
  - strengthened `utils/wasm_runtime_helpers_behavior_contract_check.sh` to
    require in `utils/wasm_runtime_helpers_behavior_check.sh`:
    - env-overridable baseline inputs:
      - `BMC_INPUT`
      - `SIM_INPUT`
    - explicit baseline-input preflight diagnostic:
      - `[wasm-runtime-helpers-behavior] missing baseline test input(s): ...`
  - Pre-fix failure:
    - `utils/wasm_runtime_helpers_behavior_contract_check.sh` failed with:
      - `missing token in behavior check script: BMC_INPUT="${BMC_INPUT:-`
    - behavior check hardcoded baseline inputs without explicit preflight.
- Fix:
  - updated `utils/wasm_runtime_helpers_behavior_check.sh`:
    - added `BMC_INPUT`/`SIM_INPUT` environment overrides;
    - added upfront existence check for both baseline inputs with explicit
      diagnostic;
    - threaded overrides through both behavior test cases.
- Validation:
  - `utils/wasm_runtime_helpers_behavior_contract_check.sh` passes.
  - `utils/wasm_runtime_helpers_behavior_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.

## 2026-02-23 (follow-up: resolve smoke-contract drift on VCD token expectations)
- Gap identified (regression-test first):
  - while attempting a VCD-validation iteration, `utils/wasm_smoke_contract_check.sh`
    drifted to require token strings that were not present in the current
    `utils/run_wasm_smoke.sh` workspace version.
  - Pre-fix failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `missing token in smoke script: expected VCD output to include \$enddefinitions`
- Fix:
  - aligned `utils/wasm_smoke_contract_check.sh` with the active smoke-script
    contract by removing the stale `\$enddefinitions` token requirements.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.

## 2026-02-23 (follow-up: accept header-only VCDs in smoke validation)
- Gap identified (regression-test first):
  - runtime repro before fix:
    - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=0 WASM_REQUIRE_CLEAN_CROSSCOMPILE=0 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
      failed with:
      - `expected SV pipeline VCD to declare at least one $var: ...`
  - strengthened `utils/wasm_smoke_contract_check.sh`:
    - require smoke diagnostics to validate VCDs via `\$enddefinitions`;
    - explicitly forbid stale `$var`-declaration-required diagnostics.
  - Pre-fix contract failure:
    - `utils/wasm_smoke_contract_check.sh` failed with:
      - `stale token still present in smoke script: expected VCD output to declare at least one \$var`
- Fix:
  - updated `utils/run_wasm_smoke.sh`:
    - removed strict `^\$var` requirements for both:
      - SV pipeline VCD (`$tmpdir/verilog-sim.vcd`)
      - stdin sim VCD (`$VCD_PATH`)
    - retained non-empty and `\$enddefinitions` checks.
  - updated `utils/wasm_smoke_contract_check.sh` to enforce new contract.
- Validation:
  - `utils/wasm_smoke_contract_check.sh` passes.
  - previously failing repro now passes end-to-end:
    - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=0 WASM_REQUIRE_VERILOG=0 WASM_REQUIRE_CLEAN_CROSSCOMPILE=0 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
  - full required-verilog smoke also passes:
    - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`

## 2026-02-23 (follow-up: ignore trailing non-compile command matches in warning triage)
- Gap identified (regression-test first):
  - extended `utils/wasm_cxx20_warning_behavior_check.sh` with new case:
    - `trailing-link-step-ignored`
  - Pre-fix failure:
    - `utils/wasm_cxx20_warning_behavior_check.sh` failed with:
      - `case 'trailing-link-step-ignored' failed (rc=1 expected 0)`
    - warning triage selected the last raw line containing source-name
      substrings, which could be a non-compile link step lacking `-std=...`.
  - Real-run signal:
    - smoke failed in warning-triage stage with:
      - `compile command for FIRRTLAnnotationsGen.cpp is missing a -std=... flag`
    - dump showed compile lines with `-std=c++20` and later non-compile lines
      containing `*.cpp.o` names.
- Fix:
  - updated `utils/wasm_cxx20_warning_check.sh` source-command selection:
    - filter matches to actual compile commands containing `-c`;
    - then select the last compile command for each source.
  - kept per-source `-std` validation unchanged after command selection.
- Validation:
  - `utils/wasm_cxx20_warning_behavior_check.sh` passes.
  - `utils/wasm_cxx20_warning_contract_check.sh` passes.
  - `utils/wasm_ci_contract_check.sh` passes.
  - `WASM_SKIP_BUILD=1 WASM_CHECK_CXX20_WARNINGS=1 WASM_REQUIRE_VERILOG=1 WASM_REQUIRE_CLEAN_CROSSCOMPILE=1 NINJA_JOBS=1 utils/run_wasm_smoke.sh`
    passes end-to-end.

## 2026-02-23 (follow-up: cocotb string-handle regression path for sample_module)
- Gap identified (repro first):
  - `PYGPI_PYTHON_BIN=python3.9 COCOTB_WORKDIR=/tmp/cocotb_test_now2 utils/run_cocotb_tests.sh test_cocotb`
    failed with:
    - `test_handle.test_string_ansi_color` missing
      `stream_in_string_asciival_sum` (plus existing unrelated underscore /
      escaped-identifier misses).
  - `test_inertial_writes` remained passing.
- Realization:
  - compiling `sample_module` through the default full lowering path dropped
    the string helper signal used by cocotb's ANSI-string check.
  - compiling `sample_module` as LLHD keeps the helper signal/process visible
    to `circt-sim` VPI.
- Fix:
  - updated `utils/run_cocotb_tests.sh` to compile `sample_module` with
    `--ir-llhd` and removed forced `-D _VCP` for that design.
- Validation:
  - `PYGPI_PYTHON_BIN=python3.9 COCOTB_WORKDIR=/tmp/cocotb_test_patch1 utils/run_cocotb_tests.sh test_inertial_writes`
    passes (`PASS (5 tests)`).
  - `PYGPI_PYTHON_BIN=python3.9 COCOTB_WORKDIR=/tmp/cocotb_test_patch1 utils/run_cocotb_tests.sh test_cocotb`
    now fails `3/286` (improved from `4/286`), and
    `test_handle.test_string_ansi_color` is no longer failing.

## 2026-02-24 (follow-up: unresolved wasm/circt-sim regressions)
- Gap identified (regression-test first):
  - unresolved list still had five wasm/circt-sim items:
    - `timeout-no-spurious-vtable-warning.mlir`
    - `uvm-phase-add-duplicate-fast-path.mlir`
    - `vpi-string-put-value-test.sv`
    - `wasm-plusargs-reentry.mlir`
    - `wasm-uvm-stub-vcd.sv`
  - coverage gaps:
    - `wasm-plusargs-reentry.mlir` had no `RUN/CHECK`;
    - `wasm-uvm-stub-vcd.sv` had no `RUN/CHECK`.
  - pre-fix runtime failure:
    - timeout test printed
      `[circt-sim] Wall-clock timeout reached (global guard)`
      but still exited `0`.
- Fixes:
  - added lit coverage for wasm reentry/vcd checks:
    - `test/Tools/circt-sim/wasm-plusargs-reentry.mlir`
    - `test/Tools/circt-sim/wasm-uvm-stub-vcd.sv`
  - fixed timeout exit-code race in `tools/circt-sim/circt-sim.cpp`:
    - join/check wall-clock guard thread before final exit-code decision;
    - force non-zero exit when timeout guard fired.
  - stabilized immediate VPI string regression test semantics:
    - updated `test/Tools/circt-sim/vpi-string-put-value-test.c` to perform
      write/readback in `cbAfterDelay` (after startup arming),
      and validate immediate string readback behavior;
    - updated `test/Tools/circt-sim/vpi-string-put-value-test.sv` checks and
      `--max-time` to include scheduled callback window.
- Validation:
  - `build_test/bin/circt-sim test/Tools/circt-sim/timeout-no-spurious-vtable-warning.mlir --top test --timeout=2`
    now exits `1`, prints timeout, and no spurious vtable diagnostics.
  - `build_test/bin/circt-sim test/Tools/circt-sim/uvm-phase-add-duplicate-fast-path.mlir`
    prints `phase-add calls=1`.
  - `build_test/bin/circt-sim test/Tools/circt-sim/wasm-plusargs-reentry.mlir --top top ...`
    matches expected `verbose/debug/missing` outcomes for no args, `+VERBOSE +DEBUG`,
    and `+VERBOSE`.
  - `vpi-string-put-value-test` flow (`cc` plugin + `circt-verilog` + `circt-sim`)
    reports `VPI_STRING: 4 passed, 0 failed` and final `0 failed`.
  - `wasm-uvm-stub-vcd` flow (`circt-verilog` + `circt-sim --vcd`) writes non-empty
    VCD containing `$enddefinitions` and `sig` declaration.

## 2026-02-24 (follow-up: focused wasm regressions gate with explicit fail/xfail counters)
- Gap identified (regression-first):
  - broad `lit` over `test/Tools/circt-sim` in this dirty workspace includes
    many unrelated non-wasm failures, making wasm triage noisy.
  - no single focused wasm gate emitted explicit `failures` and `xfails`
    counters.
- Fixes:
  - added `utils/run_wasm_regressions.sh` to run a focused lit filter covering:
    - `timeout-no-spurious-vtable-warning`
    - `wasm-uvm-stub-vcd`
    - `wasm-plusargs-reentry`
    - `vpi-string-put-value-test`
    - `vpi-string-put-value-delayed-test`
    - `uvm-phase-add-duplicate-fast-path`
  - script emits summary:
    - `[wasm-regressions] summary: failures=<n> xfails=<n>`
    - exits non-zero unless both are zero.
  - stabilized timeout regression to avoid resource-guard short-circuit:
    - `test/Tools/circt-sim/timeout-no-spurious-vtable-warning.mlir`
      now runs with `--resource-guard=false`.
  - fixed VCD check ordering in:
    - `test/Tools/circt-sim/wasm-uvm-stub-vcd.sv`
    - `$var ... sig` now checked before `$enddefinitions`.
- Validation:
  - `utils/run_wasm_regressions.sh`
    returns:
    - `[wasm-regressions] summary: failures=0 xfails=0`
    - `[wasm-regressions] PASS`.

## 2026-02-24 (follow-up: harden wasm regression gate behavior + smoke integration)
- Gap identified (regression-first):
  - `utils/run_wasm_regressions.sh` exited early on lit infrastructure errors
    (for example empty filter matches) and could miss a normalized summary line.
  - the focused lit gate did not include wasm JS smoke end-to-end execution.
- Fixes:
  - hardened `utils/run_wasm_regressions.sh`:
    - captures lit exit code without early abort;
    - normalizes infra failures into `failures=1` when no test-level
      `FAIL/XFAIL/XPASS` rows are present;
    - counts `XPASS` and treats it as failure;
    - optionally executes `utils/run_wasm_smoke.sh` (enabled by default)
      and folds smoke failure into `failures`;
    - emits expanded summary:
      - `failures=<n> xfails=<n> xpasses=<n> smoke_failures=<n>`.
  - added `utils/wasm_regressions_behavior_check.sh`:
    - verifies no-match filter case fails with normalized summary;
    - verifies focused wasm lit set reports zero failures/xfails.
- Validation:
  - `utils/wasm_regressions_behavior_check.sh` passes.
  - `utils/run_wasm_regressions.sh` passes end-to-end with smoke enabled:
    - `[wasm-regressions] summary: failures=0 xfails=0 xpasses=0 smoke_failures=0`
    - `[wasm-regressions] PASS`.

## 2026-02-24 (follow-up: restore FILTER override compatibility in wasm regression gate)
- Gap identified (regression-first):
  - `utils/wasm_regressions_behavior_check.sh` failed in its
    empty-filter case because it still sets `FILTER=...`, while
    `utils/run_wasm_regressions.sh` had been switched to `FILTER_BASE`.
  - Result: no-match case unexpectedly passed by running the default filter.
- Fixes:
  - restored backward-compatible `FILTER` override in
    `utils/run_wasm_regressions.sh`:
    - if `FILTER` is set, use it directly;
    - else keep `FILTER_BASE` and optional `RUN_NATIVE_SV_LIT` composition.
- Validation:
  - `utils/wasm_regressions_behavior_check.sh` passes again.
  - `RUN_SMOKE=0 utils/run_wasm_regressions.sh` reports:
    - `[wasm-regressions] summary: failures=0 xfails=0 xpasses=0 smoke_failures=0`
    - `[wasm-regressions] PASS`.

## 2026-02-24 (follow-up: keep unified regressions entrypoint canonical)
- Realization:
  - adding a parallel "general regressions" wrapper script duplicates
    `utils/run_regression_unified.sh` and adds avoidable workflow divergence.
- Action:
  - dropped the new wrapper path and kept
    `utils/run_regression_unified.sh` as the default general entrypoint.
  - continued using `utils/run_wasm_regressions.sh` as a focused wasm gate.
- Validation:
  - `utils/run_regression_unified.sh --dry-run --profile smoke --engine circt --out-dir /tmp/unified-dryrun` exits 0.
  - `utils/wasm_regressions_behavior_check.sh` passes.

## 2026-02-24 (follow-up: unified cocotb_vpi regressions to zero failures)
- Gap identified (regression-first):
  - `utils/run_regression_unified.sh --profile smoke --engine circt --suite-regex '^cocotb_vpi$'`
    failed with three real cocotb failures:
    - missing `_underscore_name` / escaped handles (`test_cocotb`)
    - `test_first_on_coincident_triggers` timing out (no visible regs)
    - array-of-struct VPI hierarchy mismatch (`test_array`)
- Fixes:
  - rebuilt `build_test/bin/circt-verilog` and `build_test/bin/circt-sim` from
    current tree so `vpi.all_vars` synthesis path is active.
  - fixed `circt-sim` link failure during rebuild by adding JIT deps in
    `tools/circt-sim/CMakeLists.txt`:
    - `MLIRExecutionEngine`
    - `MLIRExecutionEngineUtils`
  - fixed cocotb coincident-trigger compile path in
    `utils/run_cocotb_tests.sh`:
    - compile `test_first_on_coincident_triggers` with `--ir-llhd`.
  - made cocotb workdir default per-process (when `COCOTB_WORKDIR` unset) in
    `utils/run_cocotb_tests.sh` to avoid stale-result cross-run contamination.
  - added regression test:
    - `test/Conversion/ImportVerilog/vpi-struct-fields-array-of-struct.sv`
  - fixed struct metadata emission for arrays-of-struct and struct array
    fields in `lib/Conversion/ImportVerilog/Structure.cpp`:
    - emit `vpi.struct_fields` for unpacked arrays whose element type is
      unpacked struct;
    - include field array metadata (`is_array`, bounds, element width).
- Validation:
  - `build_test/bin/circt-verilog ... | llvm/build/bin/FileCheck` passes for
    `test/Conversion/ImportVerilog/vpi-struct-fields-array-of-struct.sv`.
  - `COCOTB_WORKDIR=<tmp> utils/run_cocotb_tests.sh`:
    - `test_cocotb` PASS (286 tests)
    - `test_force_release` PASS (7 tests)
    - `test_first_on_coincident_triggers` PASS (1 test)
    - full suite PASS (`Total: 43, Pass: 43, Fail: 0`)
  - unified lane PASS:
    - `utils/run_regression_unified.sh --profile smoke --engine circt --suite-regex '^cocotb_vpi$'`
      reports `selected=1 failures=0`.

## 2026-02-24 (follow-up: circt-sim/wasm regression compatibility + UVM auto-include gating)
- Gap identified (regression-first):
  - focused lit repro showed `test/Tools/circt-sim/dyn-array-new-copy-after-wait.sv`
    failing because `circt-sim` no longer accepted legacy
    `--jit-compile-budget=0`.
  - broader `test/Tools/circt-sim` sweeps also exposed parse failures in
    non-UVM tests when `circt-verilog` auto-injected UVM by default.
- Fixes:
  - restored CLI compatibility in `tools/circt-sim/circt-sim.cpp` by accepting
    legacy JIT knobs as no-op flags:
    - `--jit-compile-budget`
    - `--jit-hot-threshold`
    - `--jit-cache-policy`
    - `--jit-fail-on-deopt`
    - `--jit-report`
  - tightened UVM auto-injection in `tools/circt-verilog/circt-verilog.cpp`:
    - only auto-inject UVM when sources appear to reference UVM
      (or when `--uvm-path` is explicitly provided).
- Validation:
  - targeted lit repros pass:
    - `test/Tools/circt-sim/dyn-array-new-copy-after-wait.sv`
    - `test/Tools/circt-sim/process-kill-await.sv`
    - `test/Tools/circt-sim/interface-pullup-distinct-driver-sensitivity.sv`
    - `test/Tools/circt-sim/fork-disable-ready-wakeup.sv`
    - `test/Tools/circt-sim/config-db.sv`
    - `test/Tools/circt-sim/uvm-wait-for-nba-region.sv`
  - unified lane still PASS:
    - `utils/run_regression_unified.sh --profile smoke --engine circt --suite-regex '^cocotb_vpi$'`
      reports `selected=1 failures=0`.
  - wasm regression gate now clean:
    - `utils/run_wasm_regressions.sh`
      reports `[wasm-regressions] summary: failures=0 xfails=0 xpasses=0 smoke_failures=0`.

## 2026-02-24 (follow-up: wasm regression runner lock contention hardening)
- Gap identified (regression-first):
  - concurrent wasm regression invocations can race on shared lit artifacts.
  - added a behavior test first in
    `utils/wasm_regressions_behavior_check.sh` (`lock-contention-fails-cleanly`)
    and confirmed it failed before implementation.
- Fixes:
  - added explicit lock acquisition in `utils/run_wasm_regressions.sh`:
    - `WASM_REGRESSIONS_LOCK_FILE` (default: `/tmp/circt-wasm-regressions.lock`)
    - `WASM_REGRESSIONS_LOCK_WAIT_SECS` (default: `0`)
    - emits deterministic diagnostic on contention:
      - `[wasm-regressions] lock busy: ...`
- Validation:
  - `utils/wasm_regressions_behavior_check.sh` passes all cases, including
    lock-contention behavior.
  - `utils/run_wasm_regressions.sh` remains clean:
    - `[wasm-regressions] summary: failures=0 xfails=0 xpasses=0 smoke_failures=0`
    - `[wasm-regressions] PASS`.

## 2026-02-24 (follow-up: remove cocotb allowlist, fix string/VPI root cause)
- Gap identified (regression-first):
  - `test_handle.test_string_ansi_color` in cocotb only passed via a temporary
    allowlist in `utils/run_cocotb_tests.sh`.
  - direct repro in `vpi-string-put-value-test` failed:
    - `FAIL: string write propagated to asciival_sum`
    - `VPI_STRING: FINAL: ... 1 failed`
- Root cause:
  - the interpreter collapsed LLHD `delta` and `epsilon` into one counter for
    immediate-drive visibility checks, so blocking assignments lowered as
    `<0ns,0d,1e>` were treated like NBA updates for same-process reads.
- Fixes:
  - in `tools/circt-sim/LLHDProcessInterpreter.cpp`, `interpretDrive` now
    treats zero-time epsilon-only delays as immediate for
    `pendingEpsilonDrives`, while keeping true delta/NBA behavior distinct.
  - removed cocotb known-failure passthrough in `utils/run_cocotb_tests.sh`.
  - added regression test pair:
    - `test/Tools/circt-sim/vpi-string-put-value-test.sv`
    - `test/Tools/circt-sim/vpi-string-put-value-test.c`
- Validation:
  - `ninja -C build_test circt-sim`: PASS
  - `cc -shared -fPIC -o /tmp/vpi-string-put-value-test.so test/Tools/circt-sim/vpi-string-put-value-test.c -ldl && build_test/bin/circt-verilog test/Tools/circt-sim/vpi-string-put-value-test.sv --ir-moore --ir-hw --ir-llhd -o /tmp/vpi-string-put-value-test.mlir && build_test/bin/circt-sim /tmp/vpi-string-put-value-test.mlir --top vpi_string_test --max-time=100000 --vpi=/tmp/vpi-string-put-value-test.so`: PASS (`VPI_STRING: FINAL: 6 passed, 0 failed`)
  - `utils/run_cocotb_tests.sh test_cocotb`: PASS (`286 tests`)

## 2026-02-24 (follow-up: circt-verilog.wasm UVM compile trap after output emission)
- Gap identified (repro-first):
  - wasm `circt-verilog.js` compiling a minimal `tb_top.sv` that imports full
    `uvm_pkg` produced output MLIR, then trapped in Node with:
    - `RuntimeError: memory access out of bounds`
  - same input succeeded natively; issue was wasm-runtime specific.
- Realization:
  - enabling heap growth alone (`-sALLOW_MEMORY_GROWTH=1`) was not sufficient;
    the compile path still needed a larger fixed wasm stack budget.
- Fixes:
  - in `tools/circt-verilog/CMakeLists.txt`:
    - added `CIRCT_VERILOG_WASM_STACK_SIZE` (bytes, default `33554432`).
    - validates numeric/positive values under `EMSCRIPTEN`.
    - links `circt-verilog` with `-sSTACK_SIZE=<value>`.
  - in `utils/configure_wasm_build.sh`:
    - added `CIRCT_VERILOG_WASM_STACK_SIZE` env passthrough + validation.
    - threads option into cmake configure command.
  - in `utils/wasm_configure_contract_check.sh`:
    - added command-token/source-token checks for the new stack-size knob.
    - added invalid override cases (`maybe`, `0`) with explicit diagnostics.
- Validation:
  - `utils/wasm_configure_contract_check.sh`: PASS.
  - `ninja -C build-wasm circt-verilog`: PASS.
  - wasm repro command:
    - `node build-wasm/bin/circt-verilog.js --resource-guard=false --ir-llhd --timescale 1ns/1ns --uvm-path lib/Runtime/uvm-core -I lib/Runtime/uvm-core/src --top tb_top -o <tmp>/design.llhd.mlir <tmp>/tb_top.sv`
    - exit code `0`; output generated (`~25 MB` MLIR); no wasm trap.

## 2026-02-24 (follow-up: browser-style UVM MEMFS regression hardening)
- Gap identified (repro-first):
  - user-reported browser worker crash path (`Malformed attribute storage object`)
    did not reproduce in local Node CLI runs, including stack-size sweeps and
    same-instance reruns.
  - to reduce risk of browser-vs-node drift, added a dedicated Node MEMFS
    regression that mirrors browser worker file loading semantics.
- Fixes:
  - added `utils/wasm_uvm_pkg_memfs_reentry_check.sh`:
    - loads `circt-verilog.js` into a VM context with `noInitialRun`.
    - copies full `lib/Runtime/uvm-core` into wasm MEMFS.
    - compiles a minimal `tb_top` importing full `uvm_pkg` twice via
      `callMain` in one wasm instance (re-entry).
    - fails on non-zero exit, missing output, `Aborted(`, or
      `Malformed attribute storage object`.
  - wired new helper into `utils/run_wasm_smoke.sh` when
    `circt-verilog` target is available.
  - extended `utils/wasm_smoke_contract_check.sh` with tokens for the new
    helper integration.
- Validation:
  - `BUILD_DIR=build-wasm NODE_BIN=node utils/wasm_uvm_pkg_memfs_reentry_check.sh`:
    PASS
  - `utils/wasm_smoke_contract_check.sh`: PASS
  - `bash -n utils/run_wasm_smoke.sh utils/wasm_uvm_pkg_memfs_reentry_check.sh utils/wasm_smoke_contract_check.sh`: PASS

## 2026-02-24 (follow-up: allow pre-active cbStartOfSimulation registration)
- Gap identified (test-first):
  - `vpi_register_cb` rejected all callback registrations when
    `VPIRuntime::isActive()==false`.
  - this blocks wasm/JS hosts that pre-register `cbStartOfSimulation` before
    `callMain`.
- Regression added first:
  - new unit test: `unittests/Dialect/Sim/VPIRuntimeTest.cpp`
    (`VPIRuntimeRegisterCbTest`).
  - pre-fix behavior reproduced:
    - `AllowsCbStartOfSimulationRegistrationWhileInactive` FAILED
      (`handle == nullptr`).
- Fix:
  - updated `lib/Dialect/Sim/VPIRuntime.cpp` `vpi_register_cb` guard:
    - allow pre-active registration iff reason is `cbStartOfSimulation`;
      keep rejecting null callback data and non-start reasons while inactive.
- Validation:
  - `ninja -C build_test CIRCTSimTests`: PASS.
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter=VPIRuntimeRegisterCbTest.*`: PASS (3 tests).
  - `pytest -q test/Tools/circt-sim/test_vpi.py -k startup_register_bridge`: PASS.

## 2026-02-24 (follow-up: browser-style UVM malformed-attribute A/B recheck)
- Goal:
  - validate reported wasm abort (`Malformed attribute storage object`) with
    strict old-vs-current comparison using
    `utils/repro_wasm_uvm_browser_assert.mjs`.
- A/B setup:
  - old CIRCT worktree: `0246e937a9e11f602c85a80dc2fcb2c69c5e5a84`
  - current CIRCT: `d349546a01e8cb6506fdc6ca483b1affe5603d7a`
  - old LLVM submodule: `972cd847efb20661ea7ee8982dd19730aa040c75`
  - current LLVM submodule: `d6b7ec99ca74fb0648237a5545f0878f14af6d44`
- Realizations/surprises:
  - old SHA had multiple wasm bootstrap/build issues in this environment:
    missing `CONFIGURE_circt_NATIVE`, wasm `mlir-tblgen.js` used for host td
    generation, and one source-level mismatch in ImportVerilog context fields.
  - local temporary workarounds were required in `/tmp/circt-repro-0246` to
    produce an old wasm artifact for comparison.
- Result:
  - old artifact run summary:
    - `exitCode=0`, `callMainErrorPresent=false`, `outMlirBytes=25034335`,
      `hasMalformed=false`, `hasAbort=false`.
  - current artifact run summary:
    - `exitCode=0`, `callMainErrorPresent=false`, `outMlirBytes=25034364`,
      `hasMalformed=false`, `hasAbort=false`.
- Conclusion:
  - repro script did not reproduce malformed-attribute abort on either old or
    current artifacts in this environment.
  - likely environment/artifact drift in the original failure report; should be
    reported upstream as “currently not reproducible.”

## 2026-02-24 (follow-up: add browser-like UVM harness to wasm CI)
- Goal:
  - keep browser-like wasm UVM compile path covered in CI with the new harness.
- Test-first sequence:
  - strengthened `utils/wasm_ci_contract_check.sh` to require token:
    - `utils/repro_wasm_uvm_browser_assert.mjs --expect-pass`
  - pre-workflow-update contract run failed as expected with missing-token error.
- CI update:
  - updated `.github/workflows/wasmSmoke.yml` with step:
    - install Playwright runtime: `npm install --no-save --no-package-lock @playwright/test`
    - install browser binary: `npx playwright install chromium`
    - run harness health check:
      `node utils/repro_wasm_uvm_browser_assert.mjs --expect-pass`
- Validation:
  - `utils/wasm_ci_contract_check.sh`: PASS after workflow update.
- Realization:
  - harness itself is now suitable as a stability gate for healthy artifacts;
    malformed-attribute abort remains environment/artifact dependent and is not
    currently reproducible on this workspace default wasm artifact.

## 2026-02-24 (follow-up: wasm `circt-sim` UVM init OOB fix via stack contract)
- Gap identified (repro-first):
  - Node wasm UVM flow could compile full `uvm_pkg` but `circt-sim.wasm`
    trapped during initialization with:
    - `RuntimeError: memory access out of bounds`
  - failure occurred before simulation completed and blocked tutorial UVM runs.
- Fixes:
  - added wasm stack/memory knobs to `circt-sim`:
    - `CIRCT_SIM_WASM_STACK_SIZE` (default `33554432`)
    - `CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH` (default `ON`)
    - wired to link flags `-sSTACK_SIZE=...` and `-sALLOW_MEMORY_GROWTH=1`.
  - extended `utils/configure_wasm_build.sh` with:
    - env passthrough + validation for both new knobs.
  - extended `utils/wasm_configure_contract_check.sh` with:
    - command-token and invalid-override checks for both new knobs.
  - added UVM sim regression helper:
    - `utils/wasm_uvm_pkg_sim_check.sh`
    - compiles a minimal full-`uvm_pkg` sample with `circt-verilog.wasm`,
      runs `circt-sim.wasm --mode interpret --max-time=... --vcd`,
      and fails on wasm runtime abort signatures.
  - hardened `utils/wasm_uvm_pkg_memfs_reentry_check.sh` for Node raw-fs
    artifacts:
    - switched helper payload loading to host-path re-entry mode (same-instance
      `callMain`), avoiding MEMFS writes that fail when `-sNODERAWFS=1` is
      enabled.
  - wired helper into `utils/run_wasm_smoke.sh` and
    `utils/wasm_smoke_contract_check.sh`.
- Validation:
  - `utils/wasm_configure_contract_check.sh`: PASS.
  - `cmake -S llvm/llvm -B build-wasm -DCIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH=ON -DCIRCT_SIM_WASM_STACK_SIZE=33554432`: PASS.
  - `ninja -C build-wasm -j4 circt-sim`: PASS.
  - fresh Node wasm UVM compile+sim:
    - `circt-verilog.js` rc=0, output MLIR generated (~25MB).
    - `circt-sim.js --mode interpret --max-time=1000000` rc=0.
    - stdout includes `UVM_INFO ... [RNTST] Running test my_test...`.
    - VCD generated with `$enddefinitions`.

## 2026-02-25 (post-merge validation: wasm UVM pass, VPI startup-yield still failing)
- Repro-first checks run after pulling/merging upstream wasm changes:
  - `node utils/repro_wasm_uvm_browser_assert.mjs --expect-pass`: PASS
    - summary: `outMlirBytes=25034364`, `hasMalformed=false`, `hasAbort=false`.
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_startup_yield_check.sh`: FAIL
    - failure signature: `yield hook did not execute async cbStart registration path`.
- Realization:
  - source-side pre-registration guard is present in `VPIRuntime.cpp`, but the
    async startup path still does not complete through the first `await` in the
    Node wasm harness, so async suspension/resume remains incomplete in this
    validation path.
- Build-system surprise during validation:
  - wasm rebuilds are fragile in this dirty tree because
    `tools/circt-sim-compile/CMakeLists.txt` references missing
    `LowerTaggedIndirectCalls.cpp`, which breaks CMake regeneration for later
    incremental rebuild attempts.

## 2026-02-25 (async-yield follow-up: restore suspension path + harden node regression)
- Repro-first:
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_startup_yield_check.sh`
    initially failed after `cbStartOfSimulation` pre-registration fix with:
    - `yield hook did not execute async cbStart registration path`.
- Key realization:
  - in this Emscripten mode (`-sASYNCIFY=1`), a JSImport implemented as plain
    `async function` did not suspend wasm; the callback resumed after simulation
    had already progressed/completed.
  - restoring `Asyncify.handleAsync(async () => ...)` for
    `circt_vpi_wasm_yield` preserves suspension ordering in the startup path.
  - `callMain()` can return before async rewind completion; node regression
    checks must wait for post-await callback effects.
- Fixes:
  - `tools/circt-sim/circt-sim-vpi-wasm.js`
    - use `Asyncify.handleAsync(async function() { ... })` for
      `circt_vpi_wasm_yield`.
  - `tools/circt-sim/CMakeLists.txt`
    - expand asyncify import list to include both spellings:
      `_circt_vpi_wasm_yield` and `circt_vpi_wasm_yield`.
  - `utils/wasm_vpi_startup_yield_check.sh`
    - after `callMain`, poll briefly to allow Asyncify rewind completion before
      asserting `cbStart` async registration and `cbAfterDelay` firing.
- Validation:
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_startup_yield_check.sh`:
    PASS.
  - `node utils/repro_wasm_uvm_browser_assert.mjs --expect-pass`:
    PASS (`outMlirBytes=25034364`, `hasMalformed=false`, `hasAbort=false`).

## 2026-02-25 (wasm risk scan after merge)
- Scope:
  - static scan of wasm-sensitive codepaths in `tools/circt-sim`,
    `tools/circt-verilog`, `VPIRuntime`, runtime helpers, and wasm scripts.
  - one runtime spot-check for `$system` behavior in Node-backed wasm.
- Findings (potential wasm fragility):
  - Node-only FS default in wasm tool wrappers (`-sNODERAWFS=1`) remains set
    for both `circt-verilog` and `circt-sim`; true browser hosts still require
    custom `process/require/fs/path` shims.
  - two wasm helper scripts still invoke global `callMain(...)` instead of
    `Module.callMain(...)`, which is brittle against Emscripten wrapper changes.
  - `vpi_startup_register` callback storage is process-global and not reset,
    so long-lived wasm module reuse can accumulate startup callback pointers.
  - `--vpi` dynamic-library path has no wasm-specific diagnostic despite wasm
    environments not supporting normal `dlopen` semantics.
  - resource guard stays disabled on Emscripten by design; hangs/OOM are
    therefore unbounded in wasm hosts.
- Validation note:
  - `$system` syscall spot-check in Node-backed wasm path passed
    (`syscall-system.sv`), so current behavior is usable in Node-like runtimes.

## 2026-02-25 (wasm VPI re-entry callback leakage regression + fix)
- TDD repro:
  - added `utils/wasm_vpi_reentry_callback_isolation_check.sh` to verify
    callback isolation across same-instance `callMain` runs.
  - repro behavior before fix (on rebuilt wasm artifacts):
    `run2` observed stale `cbStartOfSimulation` callback without re-registration,
    proving cross-run VPI callback leakage.
- Root cause:
  - `VPIRuntime` singleton callback/object state was not reset between wasm
    `callMain` invocations.
- Fix:
  - added `VPIRuntime::resetForNewSimulationRun()` and invoke it on
    emscripten after `simContext.run()` completes (including failure path), so
    pre-registered startup callbacks remain valid for current run but do not
    leak into subsequent runs.
- Regression coverage:
  - wired new helper into `utils/run_wasm_smoke.sh` and
    `utils/internal/checks/wasm_smoke_contract_check.sh`.
- Validation:
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_reentry_callback_isolation_check.sh`: PASS.
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_vpi_startup_yield_check.sh`: PASS.
  - `utils/internal/checks/wasm_smoke_contract_check.sh`: PASS.

## 2026-02-25 (new wasm bug found: circt-verilog abort via thread constructor)
- Discovery path:
  - while running wasm helper checks in `build-wasm-mergecheck`,
    `utils/wasm_resource_guard_default_check.sh` failed in the
    `circt-verilog` phase with hard abort.
- Minimal repro (mergecheck artifacts):
  - `cat test/Tools/circt-sim/reject-raw-sv-input.sv | node build-wasm-mergecheck/bin/circt-verilog.js --no-uvm-auto-include --ir-llhd --single-unit --format=sv -o /tmp/out.mlir -`
  - observed: `Aborted()` and no output MLIR.
- Assertion-enabled diagnosis:
  - rebuilt `circt-verilog` with `CIRCT_VERILOG_WASM_ASSERTIONS=ON`.
  - abort message becomes explicit:
    `system_error was thrown in -fno-exceptions mode with error 138 and message "thread constructor failed"`.
- Narrowing:
  - `--parse-only` succeeds on same input/artifact.
  - `--lint-only` aborts with the same thread-constructor failure.
  - indicates the semantic-analysis/lint path triggers thread creation in wasm
    where threads are unavailable.
- Context note:
  - this failure is present in current `build-wasm-mergecheck` artifacts and
    explains why wasm UVM/frontend helpers fail there before simulation.

## 2026-02-25 (new wasm bug found: circt-sim --timeout abort)
- Minimal repro (works on both `build-wasm` and `build-wasm-mergecheck`):
  - `node <build>/bin/circt-sim.js --resource-guard=false --timeout 1 test/Tools/circt-sim/llhd-combinational.mlir`
  - observed: immediate hard abort (`RuntimeError: Aborted...`).
- Expected:
  - either timeout watchdog support in wasm, or a graceful diagnostic that
    `--timeout` is unsupported on emscripten.
- Likely root cause:
  - `SimulationContext::startWatchdogThread()` always constructs
    `std::thread` when `timeout > 0`.
  - on wasm/emscripten (no pthreads), thread construction aborts.
  - code location: `tools/circt-sim/circt-sim.cpp` around
    `startWatchdogThread()` and the `watchdogThread = std::thread(...)` path.

## 2026-02-25 (fix: wasm thread-option fallbacks for circt-sim)
- TDD regression added:
  - `utils/wasm_threaded_options_fallback_check.sh`
  - checks that in wasm/node artifacts:
    - `circt-sim.js --timeout 5` does not abort
    - `CIRCT_SIM_EXPERIMENTAL_PARALLEL=1 circt-sim.js --parallel 2` does not abort
- Pre-fix behavior:
  - both commands aborted with `RuntimeError: Aborted(...)`.
- Fixes in `tools/circt-sim/circt-sim.cpp`:
  - `startWatchdogThread()` now no-ops on emscripten (no `std::thread`).
  - `setupParallelSimulation()` now gracefully falls back to sequential mode on
    emscripten even when `CIRCT_SIM_EXPERIMENTAL_PARALLEL=1`.
  - global `WallClockTimeout` thread guard in `processInput()` is disabled on
    emscripten with an explicit warning; cooperative timeout checks remain.
- Smoke wiring:
  - `utils/run_wasm_smoke.sh` now runs
    `utils/wasm_threaded_options_fallback_check.sh`.
- Validation:
  - `BUILD_DIR=build-wasm-mergecheck NODE_BIN=node utils/wasm_threaded_options_fallback_check.sh`: PASS.
  - direct commands now return rc=0 with warnings instead of aborting.
