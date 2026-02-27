# Formal Verification Engineering Log

## 2026-02-27 Session: VerifToSMT reg-source clock mapping fallbacks

### What changed
- Updated `lib/Conversion/VerifToSMT/VerifToSMT.cpp` to improve BMC clock
  resolution for multiclock metadata:
  - added `mapRegSourceArgIndexToClockPos` to resolve `bmc_reg_clock_sources`
    `arg_index` via normal source mapping first, then via unique unmapped clock
    position fallback.
  - added `regSourceClockPosByInputArg` + conflict tracking and made
    `inferClockPosFromInputArg` prefer explicit reg-source metadata before
    heuristic inferred mappings.
  - added on-demand assumption-equivalence fallback in
    `inferClockPosFromInputArg` to recover arg->clock mapping through UF roots.
  - relaxed `bmc_reg_clocks` alias insertion conflicts from hard error to
    non-fatal skip when a conflicting alias is encountered.
- Added regressions:
  - `test/Conversion/VerifToSMT/bmc-multiclock-check-name-reg-source-arg-fallback.mlir`
  - `test/Conversion/VerifToSMT/bmc-multiclock-check-name-reg-source-arg-only-fallback.mlir`

### Red-first / validation notes
- Rebuilt `circt-opt` in `build_clang_test` after the conversion change.
- Passed focused checks:
  - new arg fallback regression
  - new arg-only fallback regression
  - existing `bmc-clock-name-via-arg-equivalence.mlir`
  - existing `bmc-unmapped-clock.mlir` (`--verify-diagnostics` mode)

### Realizations
- Some OpenTitan-style flows provide `bmc_reg_clock_sources` entries with only
  `arg_index` while `bmc_clock_sources` only partially covers clock positions;
  using the unique remaining unmapped clock position closes this gap.
- Explicit reg-source metadata should take precedence over inference from
  equivalence heuristics to avoid false conflicts.

### Surprises
- Rebuilding only `circt-opt` succeeded despite unrelated dirty files that had
  blocked larger rebuilds earlier.
- The OpenTitan lane moved from infrastructure-level BMC legalization errors
  to solver-level `FAIL (SAT)` once these clock mapping gaps were addressed.

## 2026-02-27 Session: OpenTitan FPV stopat policy + toplevel fallback

### What changed
- Updated `utils/run_opentitan_fpv_circt_bmc.py` task policy emission:
  - `hw-stopat-symbolic` now adds `allow-unmatched=1` (quoted as one pass token).
  - kept `hw-externalize-modules ... allow-missing=1` behavior unchanged.
- Added/updated task-policy regressions:
  - `test/Tools/run-opentitan-fpv-circt-bmc-task-policy.test`
  - `test/Tools/run-formal-all-opentitan-fpv-bmc-task-policy-forwarding.test`
- Added contract resolver top-level normalization for missing `_tb` tops:
  - new helpers in `utils/resolve_opentitan_formal_compile_contracts.py` scan
    declared `module` names from resolved source files.
  - if configured top is `*_tb` but missing, and the non-`_tb` module exists,
    use the non-`_tb` module as contract `toplevel`.
- Added regression:
  - `test/Tools/resolve-opentitan-formal-compile-contracts-toplevel-tb-fallback.test`

### Red-first / validation notes
- Task-policy tests were made stricter first and failed red:
  - missing stopat unmatched option caused pairwise stub failures.
- After policy patch:
  - focused lit run passed for both task-policy forwarding tests.
- New toplevel fallback regression failed red (`foo_tb` persisted), then passed
  after resolver patch.
- Focused resolver regression set passed:
  - `...-toplevel-tb-fallback.test`
  - `...-basic.test`
  - `...-target-filter.test`
  - `...-partial.test`
  - `...-unknown-task*.test`
  - `...-non-hdl-file-filter.test`
  - `...-include-file-filter.test`

### Realizations
- The correct `hw-stopat-symbolic` option is `allow-unmatched`, not
  `allow-missing`; using the wrong option fails before any model checking.
- OpenTitan sec_cm manifests can encode `_tb` tops that are absent from the
  resolved formal compile unit; contract-level normalization is the right place
  to recover this deterministically.

### Surprises
- After `_tb` fallback, `pinmux_sec_cm` moved from immediate
  `CIRCT_VERILOG_ERROR not_a_valid_top_level_module` to timeout-class behavior,
  confirming the top-level mismatch was a hard blocker.
- `csrng_sec_cm` moved from `CIRCT_OPT_ERROR unmatched stopat selectors` to
  `FAIL (SAT)` after stopat policy correction, exposing real proof outcomes.

## 2026-02-26 Session: OpenTitan FPV-BMC timeout recovery defaults

### What changed
- Updated `utils/run_opentitan_fpv_circt_bmc.py` defaults:
  - `BMC_AUTO_ASSUME_KNOWN_INPUTS` now defaults to `1` unless explicitly set.
  - `BMC_AUTO_TIMEOUT_ASSERTION_GRANULAR` now defaults to `1` unless explicitly set.
- Added regression tests:
  - `test/Tools/run-opentitan-fpv-circt-bmc-auto-assume-known-inputs.test`
  - `test/Tools/run-opentitan-fpv-circt-bmc-timeout-auto-assertion-granular-default-on.test`

### Red-first / validation notes
- Added and ran a red test for assertion-granular default-on behavior first.
  - Before code change: failed (no timeout fallback triggered by default).
  - After code change: passed.
- Focused lit coverage passed for:
  - new wrapper tests above,
  - existing timeout fallback tests (`run-opentitan-fpv-circt-bmc-timeout-*` filter),
  - pairwise retry coverage (`run-pairwise-circt-bmc-timeout-auto-assume-known-inputs.test`).

### Realizations
- OpenTitan wrapper-level defaults materially control whether pairwise recovery
  logic is used at all; recovery existed but was effectively opt-in.
- Toggling wrapper defaults is safer when covered by wrapper-level fake-runner
  tests that validate env propagation and fallback decision points directly.

### Surprises
- Two non-functional failures can masquerade as backend regressions in this
  flow:
  - stale `/tmp` contract workdirs produce file-not-found `CIRCT_VERILOG_ERROR`,
  - missing `build_test/bin/circt-verilog` produces `runner_command_not_found`.
- On a real `rstmgr_sec_cm::rstmgr` replay, both retries now trigger as
  intended (`auto_assume_known_inputs`, then timeout fallback), but the case
  still times out under bounded local settings; this remains an active parity
  gap at solver/performance level.

## 2026-02-26 Session: Contract resolver non-HDL filtering

### What changed
- Updated `utils/resolve_opentitan_formal_compile_contracts.py` file extraction
  to filter out non-Verilog artifacts from `eda.yml` file lists.
- Added entry classification:
  - Verilog sources: `.sv`, `.v` (or explicit Verilog file_type)
  - Verilog headers: `.svh`, `.vh` (include-dir only, not compile unit)
  - Non-HDL entries (for example `.py`, `.sh`) are excluded.
- Added regression:
  - `test/Tools/resolve-opentitan-formal-compile-contracts-non-hdl-file-filter.test`

### Red-first / validation notes
- New regression failed before fix with a contract `files` list containing:
  - `check_tool_requirements.py`
  - `setup_env.sh`
- After fix:
  - new test passes,
  - existing resolver tests still pass:
    - `resolve-opentitan-formal-compile-contracts-include-file-filter.test`
    - `resolve-opentitan-formal-compile-contracts-basic.test`
    - `resolve-opentitan-formal-compile-contracts-partial.test`

### Realizations
- A single non-HDL path in compile contracts can deterministically derail
  frontend parsing and hide deeper formal backend behavior.
- Filtering should happen at contract-generation time, not ad hoc in runners,
  to keep all downstream flows consistent (BMC/LEC/connectivity).

### Surprises
- `rv_core_ibex_sec_cm` moved from hard parser failure
  (`CIRCT_VERILOG_ERROR expected_member`) to timeout after this fix, confirming
  the prior failure was contract contamination, not parser capability.
- The same sweep still shows other independent gaps (`EN_MASKING` define
  absence, `PiRotate` parameter-name issue, multi-clock/clock-key LEC/BMC
  legalization), so this fix removed one blocker but exposed the next layer.

## 2026-02-26 Session: AES masking define parity + PiRotate parameter arrays

### What changed
- Added OpenTitan target-specific define overrides in
  `utils/resolve_opentitan_formal_compile_contracts.py`:
  - `aes_masked_*` -> `EN_MASKING=1`
  - `aes_unmasked_*` -> `EN_MASKING=0`
- Added regression:
  - `test/Tools/resolve-opentitan-formal-compile-contracts-aes-masking-define-overrides.test`
- Added ImportVerilog regression for multidimensional localparam array rvalue:
  - `test/Conversion/ImportVerilog/localparam-unpacked-multidim-dynamic-index.sv`
- Fixed ImportVerilog constant materialization in
  `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - direct parameter/specparam symbol materialization path
  - recursive fixed-size unpacked array constant materialization for nested
    arrays (not just 1-D integral arrays).

### Red-first / validation notes
- AES define override test failed red (no injected `EN_MASKING`) and passed
  after resolver update.
- Localparam multidim test failed red with:
  - `unknown name 'PiRotate'`
  - `no rvalue generated for Parameter`
  and passed after ImportVerilog fix.
- Existing nearby regressions still pass:
  - `fixed-array-constant.sv`
  - resolver basic/include/partial/non-HDL filter tests.

### Realizations
- Contract-level macro completeness is a first-order prerequisite for formal
  parity; missing one OpenTitan macro can completely block frontend progress.
- ImportVerilog parameter materialization needed true recursive unpacked-array
  support to handle common cryptographic helper tables (`int [N][M]`) used with
  dynamic indices.

### Surprises
- After auto-defining `EN_MASKING`, `aes_masked_sec_cm` advanced immediately to
  a different frontend IR issue (`operand_n_does_not_dominate_this_use`),
  confirming the macro blocker was real and distinct.
- After the PiRotate fix, `entropy_src_sec_cm` advanced from
  `unknown_name_pirotate` to timeout-class behavior, which is expectedly a
  later-stage performance/runtime gap rather than name-resolution failure.

## 2026-02-26 Session: Nested ternary dominance bug in ImportVerilog pipeline

### What changed
- Added new regression:
  - `test/Conversion/ImportVerilog/always-comb-nested-ternary-dominance.sv`
- Updated `lib/Conversion/ImportVerilog/ImportVerilog.cpp` in
  `populateMooreToCorePipeline`:
  - replaced the post-`convert-moore-to-core` bottom-up canonicalizer
    (`createBottomUpSimpleCanonicalizerPass`) with top-down canonicalizer
    (`createSimpleCanonicalizerPass`).

### Red-first / validation notes
- Red reproduction first:
  - `circt-verilog --ir-hw --top=repro_aes_dom` on a minimized nested-ternary
    always-comb module failed with
    `operand #0 does not dominate this use`.
- New regression failed before the pipeline change and passed after it.
- Lit validation passed:
  - `always-comb-nested-ternary-dominance.sv`
  - `fixed-array-constant.sv`
  - `localparam-unpacked-multidim-dynamic-index.sv`

### Realizations
- The failure is not a frontend parse/elaboration issue; it is introduced by the
  first canonicalization stage immediately after `convert-moore-to-core` when
  running bottom-up traversal in this pipeline.
- Switching this specific stage to top-down traversal avoids the invalid
  use-before-def rewrite while keeping the rest of the lowering stack intact.

### Surprises
- The same minimal expression shape reproduces both standalone and OpenTitan
  (`aes_masked_sec_cm`) and deterministically moves after this fix from
  `CIRCT_VERILOG_ERROR operand_n_does_not_dominate_this_use` to a later
  backend class (`CIRCT_BMC_ERROR`
  `for_smtlib_export_does_not_support_llvm_dialect_operations_insid`).
- This confirms the dominance crash was a hard blocker masking downstream BMC
  gaps.

## 2026-02-26 Session: VerifToSMT alloca-load legalization for SMT-LIB export

### What changed
- Added a new VerifToSMT regression:
  - `test/Conversion/VerifToSMT/bmc-for-smtlib-llvm-alloca-load.mlir`
- Extended `legalizeSMTLIBSupportedLLVMOps` in
  `lib/Conversion/VerifToSMT/VerifToSMT.cpp`:
  - added `AllocaLoadAccessInfo` and `resolveAllocaLoadAccess` to trace
    alloca-rooted pointer chains through constant-index `llvm.getelementptr`
    plus simple cast wrappers.
  - added alloca-aware `llvm.load` legalization for `for-smtlib-export`:
    - forward from the latest same-block store to the same alloca access path
      when available.
    - otherwise materialize a stable nondet scalar value (same strategy used by
      malloc-load legalization).
    - erase the legalized load and dead pointer address chain ops.

### Red-first / validation notes
- Red reproduction first:
  - `circt-opt` on a minimal alloca/store/load BMC case failed with:
    - `for-smtlib-export does not support LLVM dialect operations inside
      verif.bmc regions; found 'llvm.alloca'`.
- New regression failed before code change and passes after.
- Focused regression checks passed after the patch:
  - `bmc-for-smtlib-llvm-alloca-load.mlir`
  - `bmc-for-smtlib-llvm-nested-insert-extract.mlir`
  - `bmc-for-smtlib-malloc-load-nondet.mlir`
  - `bmc-for-smtlib-llvm-insert-extract.mlir`
  - plus existing LEC/LLHD spot checks (`lower-lec-llvm-*`, `mem2reg-llvm-zero`).

### Realizations
- SMT-LIB export had malloc-based fallback coverage but no parallel path for
  stack-local alloca memory, making many otherwise-simple BMC properties fail
  early on unsupported LLVM ops.
- A conservative same-block store-forwarding rule closes a practical parity gap
  without requiring a full memory SSA pass inside VerifToSMT.

### Surprises
- Targeted formal test targets (`test/check-circt-*`) are currently blocked by
  an unrelated link failure in `unittests/Support/CIRCTSupportTests`
  (`TypeIDResolver<circt::llhd::...>::id` undefined), so direct per-test
  `circt-opt | FileCheck` remained the reliable validation path.

### Follow-up expansion (same session)
- Extended the alloca legalization to aggregate reads used via
  `llvm.extractvalue`:
  - added alloca nondet caching keyed by `(alloca, element-path, type)` to keep
    repeated reads stable.
  - for alloca-backed loads, reuse the latest same-block store when possible,
    then fold `extractvalue` through stored aggregate builders; otherwise fall
    back to cached nondet scalar leaves.
- Added regression:
  - `test/Conversion/VerifToSMT/bmc-for-smtlib-llvm-alloca-aggregate-extract.mlir`
- Validation:
  - new alloca-load + alloca-aggregate tests pass.
  - existing neighboring malloc/insert-extract tests still pass.

## 2026-02-27 - Rebuild + toolchain acceleration

### What I changed
- Reconfigured `build_clang_test` to use `ccache` launchers:
  - `CMAKE_C_COMPILER_LAUNCHER=ccache`
  - `CMAKE_CXX_COMPILER_LAUNCHER=ccache`
- Verified `lld` remained enabled (`LLVM_ENABLE_LLD=ON`).
- Rebuilt `circt-opt` and `circt-bmc` successfully after reconfigure.

### Build blocker encountered
- Rebuild failed in `lib/Conversion/ExportVerilog/PruneZeroValuedLogic.cpp`
  due to ambiguous `Type`/`Value` (`llvm::Type/Value` vs `mlir::Type/Value`).
- Applied a minimal fix by explicitly qualifying these references as
  `mlir::Type` and `mlir::Value`.

### Validation
- Object-level compile check for
  `PruneZeroValuedLogic.cpp.o` passed after the fix.
- Full target rebuild succeeded:
  - `build_clang_test/bin/circt-bmc`
  - `build_clang_test/bin/circt-opt`

### Realizations
- Switching launcher settings in an existing Ninja tree invalidates command
  signatures and effectively triggers a large one-time rebuild.
- This tree has many `ccache`-uncacheable invocations, so short-term speedup is
  modest; the main win is for repeated identical rebuild loops.
