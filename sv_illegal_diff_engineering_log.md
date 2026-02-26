# Illegal SV Differential Engineering Log

## 2026-02-25

### Task
Build a differential test suite to find illegal SystemVerilog that Xcelium rejects but CIRCT accepts.

### Realizations
- `xmvlog` alone is not enough for this objective. Many legality checks (especially multi-driver constraints around `always_comb`, `always_ff`, `always_latch`, and mixed continuous/procedural drivers) are enforced during elaboration.
- `xrun -sv <case> -elaborate` provides significantly better ground truth for semantic illegality checks than `xmvlog -sv` alone.
- CIRCT currently accepts a broad set of illegal-driver cases that Xcelium rejects.

### Surprises
- `input var` with procedural assignment compiles in `xmvlog` but is rejected during `xrun -elaborate` due to conflicting drivers.
- Multiple-driver violations that are hard errors in Xcelium elaboration are often silently accepted by `circt-verilog --lint-only`.

### Current Differential Snapshot
Using `utils/run_illegal_sv_xcelium_diff.py` with `--xcelium-mode xrun-elaborate` on the seeded corpus:
- `xcelium_reject_circt_accept`: 12
- `both_reject`: 2

### Artifacts Added
- `utils/illegal_sv_diff_cases/*.sv` (seed illegal corpus)
- `utils/run_illegal_sv_xcelium_diff.py` (differential runner + TSV/JSON reports)

### Next Steps
- Promote high-value gap cases into CIRCT regression tests as fixes land.
- Improve CIRCT diagnostics for each gap category (illegal lvalue, conflicting drivers, always_* single-driver semantics).

## 2026-02-25 (Fix pass)

### Goal
Close all currently known `xcelium_reject_circt_accept` gaps in the seeded illegal-SV corpus before adding official regression tests.

### Implementation
- Wired slang semantic analysis into CIRCT import flow by invoking `driver.runAnalysis(*compilation)` in `lib/Conversion/ImportVerilog/ImportVerilog.cpp`.
- Kept CIRCT's own mode handling (`OnlyLint`, `OnlyParse`, `Full`) for IR-mapping behavior, but disabled slang `CompilationFlags::LintMode` so semantic legality checks run consistently in `--lint-only` mode.

### Realizations
- The missing diagnostics were emitted by slang's analysis phase (driver tracking), not by initial parse/elaboration diagnostics.
- CIRCT previously did not run `driver.runAnalysis`, so those checks never surfaced.

### Surprise and fix
- A first attempt using manual `AnalysisManager` wiring caused a teardown crash in `slang::driver::Driver::~Driver`.
- Switching to slang's built-in `driver.runAnalysis` path fixed the crash and kept diagnostics stable.

### Verification
- Ran:
  - `utils/run_illegal_sv_xcelium_diff.py --circt-verilog /home/thomas-ahle/circt/build_test/bin/circt-verilog --xcelium-mode xrun-elaborate --fail-on-xcelium-accept --fail-on-gap`
- Result:
  - `both_reject: 14`
  - `xcelium_reject_circt_accept: 0`

### Follow-up
- Official regression tests can now be added from the corpus once the team is ready.

### Scope note
- The strict analysis pass is currently enabled for CIRCT `--lint-only` mode.
- Full import / lowering paths are intentionally left unchanged for now to avoid broad frontend behavior churn while we harden diagnostics incrementally.

## 2026-02-25 (Full-mode closure + expanded corpus)

### Goal
Eliminate remaining gaps where Xcelium rejects illegal SV but CIRCT accepts in normal (non-`--lint-only`) mode.

### Realizations
- The differential gap remained in normal import mode because semantic analysis was still gated to `OnlyLint`.
- Enabling analysis for full mode closes driver-legality holes without changing parser-only behavior.

### Implementation
- Updated `ImportDriver::importVerilog` to run `driver.runAnalysis(*compilation)` for all modes except `OnlyParse`.
- Kept temporary override of slang `CompilationFlags::LintMode` around analysis to ensure analysis executes even when lint mode is enabled by options.
- Extended differential harness:
  - Added `--circt-mode {full,lint-only,parse-only}` to `utils/run_illegal_sv_xcelium_diff.py`.
  - Script now defaults to `--circt-mode full`.
  - Added `circt_mode` to TSV/JSON artifacts.
- Expanded illegal corpus from 14 to 22 cases with additional patterns:
  - `always_ff`/`always_latch` and `always_comb`/`always_latch` mixed drivers.
  - `always_ff` plus `always @*` mixed driver.
  - Whole-struct plus field-level conflicting drivers.
  - Generate-block multi-driver conflict.
  - Invalid `always_comb`/`always_ff`/`always_latch` event-control forms.

### Verification
- Full mode:
  - `utils/run_illegal_sv_xcelium_diff.py --circt-verilog /home/thomas-ahle/circt/build_test/bin/circt-verilog --xcelium-mode xrun-elaborate --circt-mode full --fail-on-xcelium-accept --fail-on-gap`
  - Result: `both_reject: 22`, `xcelium_reject_circt_accept: 0`.
- Lint mode:
  - Same command with `--circt-mode lint-only`.
  - Result: `both_reject: 22`, `xcelium_reject_circt_accept: 0`.
- Additional synthetic scan (20 generated cases) found no remaining `xrun reject / circt accept` gaps.

### Artifacts
- `/tmp/circt-illegal-sv-diff-20260225-125558/summary.json`
- `/tmp/circt-illegal-sv-diff-20260225-125604/summary.json`

### Notes
- `check-circt-conversion-importverilog` in this workspace is currently blocked by unrelated dirty-tree failures in `circt-sim` sources.
- Direct `llvm-lit` invocation in this sandbox hit Python multiprocessing semaphore permission errors.

## 2026-02-25 (Cleanup / refactor pass)

### Goal
Refactor the illegal-SV differential work for maintainability before promoting to official regression tests.

### Changes
- `ImportVerilog.cpp`:
  - Extracted slang diagnostic severity policy into helper functions.
  - Added RAII `ScopedCompilationFlagOverride` for temporary compilation flag changes.
  - Unified lint-mode policy with helper functions shared between setup and analysis paths.
- Differential harness `run_illegal_sv_xcelium_diff.py`:
  - Refactored command construction and output writing into dedicated helpers.
  - Added case-level expectation parsing from `// EXPECT_CIRCT_DIAG:` and `// EXPECT_XCELIUM_DIAG:` tags.
  - Added expectation reporting in TSV / JSON summaries.
  - Added `--fail-on-expect-mismatch`.
- Corpus:
  - Added `EXPECT_CIRCT_DIAG` tags to all 22 illegal SV cases.

### Realizations
- Centralizing severity policy and flag override behavior makes frontend legality behavior easier to review and less error-prone.
- Embedding expected diagnostic substrings directly in each test case gives lightweight guardrails against regressions in diagnostic quality.
- Running two harness instances in the same second can collide on `/tmp` output directory names; switched default timestamp format to include microseconds.

### Surprises
- Setting slang `CompilationFlags::LintMode=true` for CIRCT `--lint-only` caused analysis legality checks to disappear again. The unified policy now explicitly keeps slang lint-mode disabled, preserving pre-refactor behavior while still centralizing the policy.
