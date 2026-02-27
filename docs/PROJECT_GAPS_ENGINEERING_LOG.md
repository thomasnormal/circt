# Project Gaps Engineering Log

## 2026-02-27

### ImportVerilog: unconnected inout instance port
- Repro:
  - `module child(inout wire io); endmodule`
  - `module top; child u0(.io()); endmodule`
  - `circt-verilog --ir-moore` failed with `unsupported port 'io' (Port)`.
- Root cause:
  - Unconnected-port materialization handled `In` by synthesizing a placeholder net/var and reading it, but `InOut` fell through to unsupported.
- Fix:
  - For unconnected `InOut`, synthesize the same placeholder net/var and keep it as an lvalue ref (no `moore.read`), then wire that into `portValues`.
- Tests:
  - Added `test/Conversion/ImportVerilog/unconnected-inout-instance-port.sv`.
  - Verified with `llvm-lit` filter for that test and direct `circt-verilog` repro.

### ImportVerilog: capture module-scope interface refs in task/function regions
- Repro:
  - Access a module-scope interface instance (`vif.sig`) from an `automatic` task.
  - Lowering could produce `moore.read` that uses a value defined outside the task
    function region, violating region isolation.
- Root cause:
  - Interface instance refs used by hierarchical member access were read directly
    without ensuring they were captured into the isolated function/task region.
- Fix:
  - Call `context.captureRef(...)` before reading those interface refs in both
    relevant hierarchical access paths.
- Tests:
  - Added `test/Conversion/ImportVerilog/task-interface-instance-capture.sv`.
  - Verified with focused `llvm-lit` run for that test.

### UVM: re-enable `wait_for_state` compile-time regression
- Repro:
  - `test/Runtime/uvm/uvm_phase_wait_for_state_test.sv` was disabled with
    `UNSUPPORTED: true` and did not exercise compile-time API availability.
- Fix:
  - Converted the test to a real parse-only lit check using
    `--uvm-path=%S/../../../lib/Runtime/uvm-core`.
- Tests:
  - `llvm-lit -sv ... --filter 'Runtime/uvm/uvm_phase_wait_for_state_test.sv'`
- Note:
  - `uvm_phase_aliases_test.sv` still fails today on `final_ph` undeclared when
    run against upstream `uvm-core`; fixing that requires a coordinated
    submodule update rather than a superproject-only patch.

### MooreToCore: add explicit `always_comb` / `always_latch` lowering coverage
- Repro:
  - Existing MooreToCore coverage had TODO commentary for `always_comb` and
    `always_latch`, but no focused regression proving the implicit wait-loop
    lowering shape.
- Fix:
  - Added `test/Conversion/MooreToCore/procedure-always-comb-latch.mlir`.
  - The test checks that both procedure kinds lower to `llhd.process` with
    a body block and a wait block (`llhd.wait`) that loops back to the body.
- Tests:
  - `llvm-lit -sv ... --filter 'Conversion/MooreToCore/procedure-always-comb-latch.mlir'`

### UVM: fix missing `final_ph` phase alias and re-enable alias regression
- Repro:
  - `circt-verilog --parse-only --uvm-path=lib/Runtime/uvm-core test/Runtime/uvm/uvm_phase_aliases_test.sv`
  - Failed with undeclared identifier `final_ph`.
- Root cause:
  - `uvm_domain.svh` declared and initialized `build_ph` through `report_ph`
    but omitted the standard `final_ph` alias.
- Fix:
  - Added `uvm_phase final_ph;` global alias in
    `lib/Runtime/uvm-core/src/base/uvm_domain.svh`.
  - Initialized it in `get_common_domain()` with
    `domain.find(uvm_final_phase::get())`.
  - Re-enabled `test/Runtime/uvm/uvm_phase_aliases_test.sv` as a real parse-only
    test against bundled `uvm-core`.
- Tests:
  - Direct repro command now parses successfully.
  - `llvm-lit -sv ... --filter 'Runtime/uvm/uvm_phase_aliases_test.sv'`

### MooreToCore: handle selected format strings in `fstring_to_string`
- Repro:
  - A selected format string lowered to an empty string:
    `arith.select/mux` over `!moore.format_string` then `moore.fstring_to_string`.
  - Before fix, conversion fell through to "unsupported format string type"
    fallback and emitted null/zero string struct.
- Root cause:
  - `FormatStringToStringOpConversion::convertFormatStringToStringStatic`
    only handled direct format producers and concat, not selector wrappers.
  - It also did not unwrap `builtin.unrealized_conversion_cast`, which appears
    around `!sim.fstring` values in this pipeline.
- Fix:
  - Added pass-through handling for `UnrealizedConversionCastOp`.
  - Added handling for `comb.mux` and `arith.select` by recursively converting
    true/false format operands and selecting between the resulting strings.
- Tests:
  - Extended `test/Conversion/MooreToCore/string-ops.mlir` with
    `@FStringToStringSelect`.
  - Verified with focused `llvm-lit` run for that test file.

### Sim: distinguish empty native module-init skips from unsupported skips
- Repro:
  - `circt-compile -v` on a module with no cloneable native-init ops
    (`hw.module @top { hw.output }`) reported:
    `Native module init modules: 0 emitted / 1 total`
    but no skip-reason telemetry.
- Root cause:
  - Native module-init synthesis only incremented `skipReasons` when
    `unsupported == true`; the `opsToClone.empty()` path was silently skipped.
- Fix:
  - Record a dedicated skip reason (`empty`) when a module has no cloneable
    native-init ops and was not rejected as unsupported.
- Tests:
  - Added `test/Tools/circt-sim/aot-native-module-init-empty-telemetry.mlir`.
  - Verified with focused `llvm-lit` runs for:
    - `aot-native-module-init-empty-telemetry.mlir`
    - `aot-native-module-init-skip-telemetry.mlir`
    - `--filter 'Tools/circt-sim/aot-native-module-init'`

### UVM: add package timescale to bundled `uvm-core` and enable simple parse test
- Repro:
  - `Runtime/uvm/uvm_simple_test.sv` failed with:
    `error: design element does not have a time scale defined but others in the design do`
    when parsed against `--uvm-path=.../uvm-core`.
- Root cause:
  - `lib/Runtime/uvm-core/src/uvm_pkg.sv` did not declare a `timescale`, while
    tests compile compilation units that do.
- Fix:
  - Added `` `timescale 1ns/1ps `` to `uvm_pkg.sv`.
  - Updated `test/Runtime/uvm/uvm_simple_test.sv` to use bundled
    `lib/Runtime/uvm-core` in its RUN line.
- Tests:
  - `llvm-lit --filter 'Runtime/uvm/uvm_simple_test.sv' test/Runtime/uvm` now passes.
  - `llvm-lit --filter 'Runtime/uvm/' test/Runtime/uvm` improved to 8 passing / 9 failing
    (remaining failures are API-compatibility mismatches in the test corpus).

### UVM: resolve `check(...)` helper override clash in `config_db_test`
- Repro:
  - `Runtime/uvm/config_db_test.sv` failed against `uvm-core` with:
    `virtual method 'check' has different number of arguments from its superclass method`.
- Root cause:
  - The test class extends `uvm_test` and declared `check(bit,string)`, which
    conflicts with inherited `uvm_component::check()`.
- Fix:
  - Renamed the local helper to `check_result(...)` and updated all call sites.
- Tests:
  - `llvm-lit --filter 'Runtime/uvm/config_db_test.sv' test/Runtime/uvm` passes.
  - Full UVM parse subset now: 9 passing / 8 failing.

### UVM: remove duplicate `get_type_name` overrides in factory tests
- Repro:
  - `Runtime/uvm/uvm_factory_test.sv` and
    `Runtime/uvm/uvm_factory_override_test.sv` failed with
    `redefinition of 'get_type_name'`.
- Root cause:
  - These tests use `` `uvm_object_utils `` / `` `uvm_component_utils `` which
    already provide `get_type_name`, but also declared explicit methods with the
    same signature.
- Fix:
  - Removed the redundant explicit `get_type_name` methods in both tests.
- Tests:
  - `llvm-lit --filter 'Runtime/uvm/uvm_factory_test.sv|Runtime/uvm/uvm_factory_override_test.sv' test/Runtime/uvm` passes.
  - Full `Runtime/uvm` parse subset now: 11 passing / 6 failing.

### Sim: tighten `UNSUPPORTED` classification in sv-tests simulation runner
- Repro:
  - `utils/run_sv_tests_circt_sim.sh` marked a test `UNSUPPORTED` on any
    non-zero sim exit if stderr contained words like `not yet implemented`,
    even without an actual error diagnostic.
- Root cause:
  - Classification used broad substring matching:
    `unsupported|not yet implemented|unimplemented` across the whole log.
- Fix:
  - Require explicit `error:` or `fatal:` context on the same line as
    `unsupported/not yet implemented/unimplemented` before classifying as
    `UNSUPPORTED`.
  - Keep other non-zero exits as `FAIL`.
- Tests:
  - Added `test/Tools/run-sv-tests-sim-unsupported-classification-requires-error.test`
    to cover:
    - non-error note -> `FAIL`
    - explicit unsupported error -> `UNSUPPORTED`
  - Re-ran focused sim-runner tests:
    - `run-sv-tests-sim-should-fail-pass.test`
    - `run-sv-tests-sim-should-fail-elab-compile-pass.test`
    - `run-sv-tests-sim-tag-regex-empty-tags.test`
    - `run-sv-tests-sim-toolchain-derived-from-circt-verilog.test`

### ImportVerilog: update stale sampled-value unsupported expectation
- Repro:
  - `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv`
    expected `$stable(vif)` to be unsupported, but current lowering only rejects
    `$rose(vif)`.
- Root cause:
  - Test expectations drifted behind implementation: sampled-value lowering now
    supports `$stable` on virtual interface handles.
- Fix:
  - Updated STRICT/WARN checks to only expect unsupported diagnostics for
    `$rose(vif)`.
  - Marked the strict run as `not` to correctly assert diagnostic-failure mode.
- Tests:
  - `llvm-lit` for:
    - `sva-immediate-sampled-continue-on-unsupported.sv`
    - `sva-immediate-past-event-continue-on-unsupported.sv`
    - `sva-continue-on-unsupported.sv`

### UVM: align `uvm_reporting_test` with bundled `uvm-core` reporting APIs
- Repro:
  - `llvm-lit --filter 'Runtime/uvm/uvm_reporting_test.sv' test/Runtime/uvm`
  - Failed on multiple API mismatches versus bundled `uvm-core`, including:
    - missing `get_id_verbosity`
    - missing `get_report_max_quit_count`
    - missing `uvm_report_catcher::{add,remove,clear_catchers,get_catcher_count,summarize_catchers}`
    - old `catch_action` signature instead of `catch`.
- Root cause:
  - The test was written against older/helper reporting APIs that are not
    exposed by the current bundled IEEE-oriented `uvm-core` surface.
- Fix:
  - Updated `uvm_reporting_test.sv` to use currently available APIs:
    - implement `uvm_report_catcher::catch()` directly
    - use `get_verbosity_level(UVM_INFO, id)` for id verbosity checks
    - validate max quit count through the report server singleton
    - use severity-count APIs (`get_severity_count`) instead of convenience
      `get_info_count/get_warning_count`.
  - Reworked catcher registration/count/clear logic to use
    `uvm_report_cb` + `uvm_report_cb_iter`.
  - Switched `process_all_report_catchers` usage to the current
    `uvm_report_message`-based API.
  - Cast report server to `uvm_default_report_server` in the one test that
    exercises convenience quit/severity reset helpers.
- Tests:
  - `llvm-lit -sv --filter 'Runtime/uvm/uvm_reporting_test.sv' test/Runtime/uvm`
  - `llvm-lit -sv --filter 'Runtime/uvm/' test/Runtime/uvm`
    - now 12 passing / 5 failing (previously 11 / 6).

### Sim/MooreToCore: fix `$swrite` integer formatting from packed four-state payloads
- Repro:
  - `llvm-lit -sv --filter 'Tools/circt-sim/syscall-swrite.sv' test/Tools/circt-sim`
  - Observed:
    - `swrite=value=180388626432 hex=2a00000000`
  - Expected:
    - `swrite=value=42 hex=2a`
- Root cause:
  - In `FormatStringToStringOpConversion`, formatted integer ops
    (`sim.fmt.dec/hex/oct/bin`) were lowered to runtime string conversion calls
    using the raw packed four-state integer payload when `fourStateWidth` was
    present.
  - The packed encoding stores `{value,unknown}` bits in one integer, so
    passing it directly shifted the logical value by the unknown-width amount.
- Fix:
  - Added unpacking for packed four-state operands in
    `convertFormatStringToStringStatic` before integer-to-string runtime calls:
    - detect `packedWidth == 2 * fourStateWidth`
    - shift right by `fourStateWidth`
    - truncate to `fourStateWidth`
  - Reused the existing integer-to-i64 conversion path after unpacking.
- Tests:
  - Rebuilt `circt-verilog` target.
  - `llvm-lit -sv --filter 'Tools/circt-sim/syscall-swrite.sv' test/Tools/circt-sim` now passes.
