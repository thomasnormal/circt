# BMC and LEC Cleanup Refactor Plan

Date: March 1, 2026
Owner: Formal track (BMC/LEC)
Primary target: full OpenTitan BMC/LEC parity on real workloads, with reproducible Z3-backed results.

## 1. Objective

Close the remaining architecture and maintainability gaps in CIRCT formal flows so we can:

1. Run real OpenTitan BMC and LEC workloads end-to-end with stable outcomes.
2. Remove brittle script-layer policy and move correctness into tool internals.
3. Improve velocity by refactoring monolithic harnesses into testable modules.
4. Make regressions diagnosable with consistent machine-readable output.

This plan is for cleanup plus refactor work that directly improves BMC/LEC correctness, scale, and maintainability.

## 2. Current Baseline Snapshot

Known from current workspace and recent runs:

1. `sv-tests` BMC chapter 16/20 slice is green in current config (`101/101 PASS`).
2. Runner complexity is high and concentrated in a few large scripts:
   - `utils/run_sv_tests_circt_bmc.sh` (~1430 LOC)
   - `utils/run_opentitan_circt_lec.py` (~1013 LOC)
   - `utils/run_opentitan_connectivity_circt_lec.py` (~3670 LOC)
3. BMC core still has hard unsupported paths for multiclock and some register-init forms.
4. LEC still has many unsupported paths concentrated in LLHD signal/ref stripping and aggregate conversion.
5. Tool and harness outputs are not fully unified for metrics, trend analysis, and frontier tracking.

## 3. Principles

1. TDD first: add failing test or expected-error regression before each fix.
2. Keep strict semantics default; gate compatibility behavior behind explicit options.
3. Prefer internal semantic fixes over harness workarounds.
4. Keep changes sliceable: one behavior change per commit where possible.
5. Preserve stable user interfaces where possible; deprecate rather than silently change.

## 4. Plan Structure

Execution is split into six workstreams plus one cross-cutting stabilization stream:

1. WS0: Baseline freeze and instrumentation
2. WS1: Harness modularization and dedup
3. WS2: BMC multiclock closure
4. WS3: BMC register-init closure
5. WS4: LEC LLHD/ref-path closure
6. WS5: LEC LLVM aggregate conversion closure
7. WS6: Unified formal result schema and observability

## 5. WS0 - Baseline Freeze and Instrumentation

### Goal

Create a repeatable baseline and lock it before large refactors.

### Deliverables

1. Baseline manifest with exact commands, tool paths, timeout settings, and seeds.
2. Frozen reference outputs for:
   - OpenTitan LEC connectivity frontier set
   - OpenTitan BMC smoke and Z3 subsets
   - `sv-tests` BMC tagged slices
3. Minimal parser that summarizes pass/fail/timeout/error by reason code.

### Tests

1. Add lit tests for baseline parser behavior.
2. Add CI check that baseline summary format remains stable.

### Exit Criteria

1. Same commands produce identical classification summaries across 3 consecutive runs (allowing bounded timing variance).

## 6. WS1 - Harness Modularization and Dedup

### Goal

Reduce script complexity and policy drift by extracting shared formal runner logic.

### Scope

Refactor:

1. `utils/run_sv_tests_circt_bmc.sh`
2. `utils/run_opentitan_circt_lec.py`
3. `utils/run_opentitan_connectivity_circt_lec.py`

into shared modules under `utils/lib/formal/` (or equivalent) for:

1. Tool execution with retries/timeouts/resource limits
2. Case selection and filtering
3. Result classification and reason coding
4. Artifact retention and path mapping
5. Summary rendering (text plus machine-readable)

### Work Items

1. Define stable internal APIs for command execution and classification.
2. Extract pure helper functions first (no behavior change).
3. Move retry/timeout logic to shared implementation.
4. Replace duplicated reason parsing with common reason classifier table.
5. Keep old CLI behavior via thin compatibility wrappers.

### Tests

1. Expand existing `run-sv-tests-bmc-*` lit coverage to assert no behavior drift.
2. Add new unit-style tests for classifier functions using canned logs.
3. Add regression tests for environment-variable policy toggles.

### Exit Criteria

1. No output behavior change on baseline suite.
2. >=30 percent reduction in duplicated retry/classification logic.
3. New helper modules fully covered by regression tests.

## 7. WS2 - BMC Multiclock Closure

### Goal

Replace current multiclock hard-stop behavior with sound multiclock handling in `circt-bmc`.

### Current Gap

Hard unsupported diagnostics remain in:

1. `lib/Tools/circt-bmc/LowerToBMC.cpp`
2. `lib/Tools/circt-bmc/ExternalizeRegisters.cpp`

### Design Direction

1. Introduce explicit per-check clock-domain metadata in lowered BMC IR.
2. Track clock event sampling per domain.
3. Support multiple domain steps with sound cross-domain sequencing semantics.
4. Keep unsupported only for truly ambiguous mixed-clock constructs, with precise diagnostics.

### Work Items

1. Add failing multiclock lit cases beyond current auto-retry wrappers.
2. Implement domain-aware clock extraction and externalization.
3. Update non-final/final check aggregation for multiclock check sets.
4. Add multiclock counterexample metadata fields.

### Tests

1. New `test/Tools/circt-bmc/*multiclock*.mlir` semantic tests.
2. `sv-tests` multiclock-tagged subset in real Z3 mode.
3. OpenTitan multiclock-heavy shards.

### Exit Criteria

1. Remove generic multiclock hard-stop for supported patterns.
2. Multiclock subset moves from fallback/retry paths to direct PASS/FAIL semantics.

## 8. WS3 - BMC Register-Init Closure

### Goal

Support real-world register initialization shapes that currently fail externalization.

### Current Gap

Unsupported path remains for non-direct `seq.initial` init in:

1. `lib/Tools/circt-bmc/ExternalizeRegisters.cpp`

### Design Direction

1. Normalize init sources before externalization.
2. Recognize equivalent init forms (constant folds, canonicalized wrappers).
3. Preserve initialization semantics in generated SMT model.

### Work Items

1. Add red tests for currently unsupported init patterns.
2. Implement normalization pass or localized canonicalization in externalizer.
3. Enforce explicit diagnostic reasons when truly unsupported.

### Tests

1. New lit regressions under `test/Tools/circt-bmc/externalize-registers-*`.
2. Real OpenTitan BMC cases known to include non-trivial inits.

### Exit Criteria

1. Eliminate avoidable init-related unsupported errors on target OpenTitan set.
2. Keep strict expected-error tests for truly unsupported illegal forms.

## 9. WS4 - LEC LLHD Ref-Path Closure

### Goal

Systematically reduce unsupported LLHD signal/ref patterns in LEC.

### Current Gap

High concentration of unsupported diagnostics in:

1. `lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp`
2. `lib/Tools/circt-lec/LowerLLHDRefPorts.cpp`

### Design Direction

1. Build a typed rewrite matrix by unsupported class:
   - probe path shape
   - drive path shape
   - cast/GEP path shape
   - CFG predecessor ambiguity
2. Implement deterministic canonical forms before rejection.
3. Keep high-signal diagnostics with actionable reason codes.

### Work Items

1. Inventory unsupported messages and map each to a category.
2. Triage categories into:
   - C1: straightforward rewrite
   - C2: CFG-sensitive but bounded
   - C3: true unsupported (documented)
3. Burn down C1 first, then C2.

### Tests

1. Add one focused test per category under `test/Tools/circt-lec/`.
2. Add OpenTitan connectivity repros for each fixed category.
3. Assert no regressions on existing `58/58` connectivity lit set.

### Exit Criteria

1. Measurable reduction in unsupported LLHD/ref errors on OpenTitan runs.
2. Remaining unsupporteds are documented C3 with explicit reasons.

## 10. WS5 - LEC LLVM Aggregate Conversion Closure

### Goal

Close aggregate conversion gaps that block real LEC cones.

### Current Gap

Unsupported aggregate conversion handling remains in:

1. `lib/Tools/circt-lec/LowerLECLLVM.cpp`

### Design Direction

1. Canonicalize common aggregate conversions (extract/insert/load/store patterns).
2. Lower to explicit bit-precise operations before SMT export.
3. Add strict verifier to reject only non-canonical leftovers.

### Work Items

1. Add failing lit tests for aggregate conversions seen in OpenTitan runs.
2. Implement conversion legalization for common path families.
3. Tighten final unsupported check to include reason code and op context.

### Tests

1. Extend existing `lec-lower-llvm-array-*` test family.
2. Add OpenTitan-derived minimal repros in `test/Tools/circt-lec/`.

### Exit Criteria

1. Frontier parser errors move deeper (or vanish) due to fewer aggregate conversion failures.

## 11. WS6 - Unified Formal Result Schema and Observability

### Goal

Provide one stable machine-readable schema across BMC and LEC workflows.

### Schema Fields (minimum)

1. `suite`
2. `mode` (`BMC`, `LEC`)
3. `case_id`
4. `status` (`PASS`, `FAIL`, `ERROR`, `TIMEOUT`, `UNKNOWN`, `XFAIL`, `XPASS`)
5. `reason_code`
6. `stage` (`frontend`, `lowering`, `solver`, `postprocess`)
7. `solver`
8. `solver_time_ms`
9. `frontend_time_ms`
10. `log_path`
11. `artifact_dir`

### Work Items

1. Define schema contract in docs.
2. Emit JSONL from runners.
3. Add TSV projection tool for human summaries.
4. Update baseline tooling and blog/table generators to consume schema.

### Tests

1. Schema validation test with strict required-field checks.
2. Backward-compat parser tests for old logs while migration is active.

### Exit Criteria

1. All primary runners emit schema-compliant JSONL.
2. Frontier and timeout dashboards consume only schema outputs.

## 12. Sequencing and Milestones

## Milestone M1 (Stabilize and Refactor Foundation)

Includes: WS0 + WS1

Definition:

1. Baseline frozen and reproducible.
2. Harnesses modularized with no behavior drift.

## Milestone M2 (BMC Core Semantic Closure)

Includes: WS2 + WS3

Definition:

1. Multiclock support lands for supported classes.
2. Register-init unsupported class significantly reduced.

## Milestone M3 (LEC Core Semantic Closure)

Includes: WS4 + WS5

Definition:

1. LLHD/ref unsupported count reduced with category burn-down evidence.
2. Aggregate conversion blockers reduced on OpenTitan frontier.

## Milestone M4 (Unified Reporting)

Includes: WS6

Definition:

1. Unified schema in all main runners.
2. Trend/frontier reporting fed by schema only.

## 13. Risk Register

1. Risk: semantic regressions during harness refactor.
   - Mitigation: golden baseline compare and lit gate for each extracted function.
2. Risk: multiclock support introduces unsoundness.
   - Mitigation: add adversarial tests and cross-check against known solver behavior.
3. Risk: ref-path rewrites in LEC overfit OpenTitan only.
   - Mitigation: keep generic category tests not tied to one design.
4. Risk: performance regressions from more precise lowering.
   - Mitigation: include timeout and solver-time trend gates in CI.

## 14. Definition of Done

This cleanup/refactor initiative is complete when:

1. Harnesses are modular and behavior-compatible.
2. BMC multiclock and init unsupported classes are materially reduced.
3. LEC LLHD/ref and aggregate unsupported classes are materially reduced.
4. OpenTitan formal frontier is deeper and stable in real Z3 mode.
5. Unified machine-readable schema is used by all major formal workflows.
6. All changes are documented in engineering logs with command-level validation.

## 15. Immediate Next Slice

Start with WS1 in a narrow, low-risk cut:

1. Extract shared retry/timeout execution helper from `run_sv_tests_circt_bmc.sh`.
2. Keep wrapper behavior identical.
3. Add or update lit tests to lock behavior.
4. Land in small commits before broader extraction.

