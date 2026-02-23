# WASM Engineering Log

## 2026-02-23
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
    - `ninja -C build-test circt-verilog circt-sim`:
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
  - `ninja -C build-test check-circt-conversion-veriftosmt`
  - `ninja -C build-test check-circt-tools-circt-bmc`
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
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim/reject-raw-sv-input.sv` passes.

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
