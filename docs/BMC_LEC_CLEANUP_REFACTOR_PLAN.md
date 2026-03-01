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

## 16. Status Snapshot (as of March 1, 2026)

Current workstream status in this branch:

1. WS0: in progress
   - baseline manifest writer landed
   - baseline manifest contract validator landed
     (`validate_baseline_manifest.py`)
   - unsupported diagnostics audit scaffold landed
   - baseline capture runner landed (`capture_formal_baseline.py`)
   - schema drift comparator landed (`compare_formal_results_drift.py`)
   - per-lane manifest timeout controls landed in WS0 writer (`timeout_secs`)
   - baseline capture now supports default/per-command timeout budgets and
     emits timeout budgets in `execution.tsv`
   - timeout capture path is now robust to `TimeoutExpired` byte
     stdout/stderr payloads (no Python type crash)
   - baseline capture now supports per-command log size caps via
     `--max-log-bytes` to bound artifact growth
   - baseline capture can now gate command outputs with strict JSONL schema
     validation (`--validate-results-schema`)
   - baseline capture now supports strict cross-row schema gating via
     `--validate-results-schema-strict-contract`
   - strict schema-gating CLI contract is regression-covered (strict flag
     requires base schema validation flag)
   - baseline manifests now support per-command expected return-code contracts
     (`expected_returncodes`) for bounded-timeout frontier lanes
   - WS0 manifest writer now supports per-lane expected return-code flags
     (`--*-expected-returncodes`) to generate timeout-frontier baselines
     without manual manifest editing
   - latest real WS0 mini-baseline (`out/ws0-baseline-live-20260301-162812`)
     validates schema for AES LEC + sv-tests BMC lanes across 2 runs with zero
     status/reason/stage drift
   - first 3-run real mini-baseline completed for AES LEC + sv-tests BMC with zero status/reason/stage drift
   - OpenTitan connectivity LEC lane remains the timeout frontier blocker for full WS0 parity capture
2. WS1: in progress
   - shared formal schema helper landed
   - shared formal schema constants module landed for required-field + enum
     policy (`formal_results_schema.py`) and adopted by schema validator and
     dashboard aggregator
   - shared schema v1 row-validation helper now adopted by schema validator and
     dashboard aggregator (type/enum/required-field checks de-duplicated)
   - shared case-row result writer helpers landed in `formal_results.py`
     (`write_results_tsv`, `write_results_jsonl_from_case_rows`)
   - OpenTitan LEC runners now use shared case-row TSV/JSONL projection
     helpers (duplicated local writer logic removed from hot paths)
   - pairwise BMC runner now supports schema JSONL output lane
     (`--results-jsonl-file`, env `FORMAL_RESULTS_JSONL_OUT`)
   - OpenTitan AES BMC wrapper now forwards schema JSONL output lane to the
     pairwise backend (`--results-jsonl-file`)
   - OpenTitan connectivity BMC wrapper now forwards schema JSONL output lane
     to the pairwise backend (`--results-jsonl-file`)
   - connectivity BMC wrapper no-case paths now emit empty requested schema
     JSONL artifacts (selected-groups-empty and generated-cases-empty paths)
   - OpenTitan FPV BMC wrapper now emits merged schema JSONL output lane
     (`--results-jsonl-file`) from final wrapper-level merged rows
   - OpenTitan FPV LEC runner now emits schema JSONL output lane
     (`--results-jsonl-file`, env `FORMAL_RESULTS_JSONL_OUT`)
   - OpenTitan FPV LEC runner now uses shared env retry helper
     (`runner_common.run_command_logged_with_env_retry`)
   - connectivity BMC status governance now consumes shared allowlist/status
     helpers from `runner_common` with local fallback for copied-runner tests
   - FPV BMC summary-drift allowlist parsing/matching now consumes shared
     allowlist helpers from `runner_common` with local fallback for copied-runner tests
   - FPV BMC summary-drift allowlist/row-allowlist paths now validate file
     existence before load, eliminating traceback-prone missing-file paths
   - shared env-driven retry launcher landed in `runner_common`
   - OpenTitan LEC runners migrated off duplicated retry parsing
   - shared drop-reason parser landed in `runner_common` and adopted in LEC runners
   - shared log-writer truncation controls landed in `runner_common`
     and baseline capture now consumes shared log writing path
   - runner shared-library extraction still partial
3. WS2: in progress
   - multiclock unsupported root causes identified
   - diagnostic inventory coverage added for lower-to-bmc single-clock
     rejection paths, including explicit clock-name and unresolved-expression
     diagnostics
   - externalize-registers unsupported clock-shape diagnostic inventory test
     landed (`seq.clock_div`-derived non-traceable clock case)
   - core multiclock semantic lowering changes still pending
4. WS3: in progress
   - register-init unsupported inventory now includes passthrough-initial
     shape coverage (`externalize-registers-initial-passthrough.mlir`)
   - init normalization implementation still pending
5. WS4: in progress
   - several LLHD/ref compatibility rewrites exist in runner/tooling flow
   - systematic category burn-down matrix not completed
6. WS5: in progress
   - several array/aggregate LEC lowering tests exist
   - full aggregate legalization plan still incomplete
7. WS6: in progress
   - JSONL schema emission landed in major runners
   - strict schema validation tooling landed (`validate_formal_results_schema.py`)
   - schema validator now supports `--strict-contract` for cross-row
     invariants (sorted row order and solver-stage non-empty solver enforcement)
   - WS0 capture can now forward strict schema contract checks to the validator
     (`--validate-results-schema-strict-contract`)
   - JSONL->TSV migration adapter landed (`formal_results_jsonl_to_tsv.py`)
   - WS0 capture integration now supports per-command schema validation gating
     and surfaces per-run schema validation rc in `execution.tsv`
   - timeout frontier summary utility landed
     (`summarize_formal_timeout_frontier.py`) with percentile/reason/cumulative
     solver-time reporting from schema JSONL
   - schema-only dashboard input aggregator landed
     (`build_formal_dashboard_inputs.py`) with status/reason/timeout TSV+JSON
     outputs over multi-file JSONL inputs
   - dashboard aggregator now enforces strict schema enums and reason-code
     invariants to prevent malformed rows from polluting trend reports
   - dashboard aggregator now enforces full required-field presence for schema
     rows before aggregation
   - baseline capture now supports optional schema-only dashboard emission via
     `--dashboard-*` outputs, closing WS0 capture -> WS6 dashboard handoff
   - dashboard aggregation now includes expected-returncode frontier lanes
     (for example bounded timeout `124`) when JSONL outputs are present
   - schema validation now covers expected nonzero-return lanes when they emit
     JSONL payloads, preserving schema gates on bounded-timeout frontier data
   - schema contract/versioning doc landed (`docs/FormalResultsSchema.md`)

## 17. Execution Backlog (Ticket-Level)

The following backlog is the concrete implementation queue. IDs are stable handles for engineering logs and commits.

### WS0 Tickets

1. WS0-T1: baseline manifest contract v1
   - define required keys and invariants
   - add parser validation tool
2. WS0-T2: baseline capture scripts for:
   - OpenTitan LEC connectivity frontier
   - OpenTitan AES LEC suite
   - sv-tests BMC chapter slices
3. WS0-T3: three-run reproducibility gate
   - classify drift by status, reason_code, and stage
4. WS0-T4: baseline artifact retention policy
   - max artifact size
   - stable per-case log naming

### WS1 Tickets

1. WS1-T1: shared command launcher module
   - retry policy
   - timeout policy
   - resource guard handling
2. WS1-T2: shared reason classifier module
   - timeout reason normalization
   - frontend error taxonomy
   - solver error taxonomy
3. WS1-T3: shared result writer
   - JSONL schema writer
   - TSV projection helper
4. WS1-T4: script-level dead code cleanup
   - remove duplicated fallback paths
   - delete stale environment toggles no longer used

### WS2 Tickets

1. WS2-T1: multiclock unsupported inventory
   - map each unsupported message to one failing lit test
2. WS2-T2: domain-aware check lowering
   - represent per-domain step semantics in BMC lowering
3. WS2-T3: cross-domain scheduling semantics
   - deterministic step relation for multi-domain checks
4. WS2-T4: multiclock counterexample metadata
   - include clock-domain event traces in witness output

### WS3 Tickets

1. WS3-T1: register-init shape inventory
   - direct init
   - folded init
   - cast/aggregate wrapped init
2. WS3-T2: init normalization in externalizer path
3. WS3-T3: unsupported diagnostics tightening
   - emit exact unsupported init form and op location

### WS4 Tickets

1. WS4-T1: LLHD/ref unsupported matrix
   - category C1/C2/C3 ownership mapping
2. WS4-T2: C1 rewrite completion
3. WS4-T3: C2 bounded CFG-sensitive rewrites
4. WS4-T4: C3 documentation and user-facing diagnostics

### WS5 Tickets

1. WS5-T1: aggregate conversion pattern inventory
2. WS5-T2: legalization for common extract/insert/load/store families
3. WS5-T3: verifier pass for residual non-canonical aggregate forms
4. WS5-T4: OpenTitan-derived regression pack for aggregate cones

### WS6 Tickets

1. WS6-T1: schema contract doc and versioning policy
2. WS6-T2: strict schema validator CLI
3. WS6-T3: migration adapters for old TSV pipelines
4. WS6-T4: dashboard inputs from schema-only data

## 18. Timeout Frontier Program (Z3 Mode)

This is the dedicated program to push the timeout frontier deeper with real OpenTitan workloads.

### Definition

A timeout frontier case is a case that:

1. passes preprocessing/lowering
2. reaches solver stage in real Z3 mode
3. returns `TIMEOUT` at or above a configured timeout budget

### Method

1. Build per-case timing table with:
   - frontend_time_ms
   - solver_time_ms
   - status
   - reason_code
2. Bin cases by runtime percentiles:
   - P50
   - P90
   - P99
3. Focus optimization on:
   - top 10 cumulative solver-time contributors
   - top 10 frequent timeout reason_code families

### Optimization Loop

1. pick one reason_code cluster
2. create minimal reproducer
3. profile generated SMT/IR size and solver behavior
4. implement one change
5. re-run frontier table
6. keep change only if:
   - timeout count decreases, or
   - same timeout count but lower total solver time

### Exit Criteria

1. no regression in pass/fail semantics
2. timeout frontier shifts forward by at least 25 percent in case count at fixed budget
3. solver-time P90 reduced by at least 20 percent on frontier suite

## 19. Test Strategy Matrix

Every ticket should land with one test from each relevant layer.

1. unit-level
   - helper/parser/classifier behavior
2. lit regression
   - exact failing lowering or runner behavior
3. integration runner
   - end-to-end harness flow over synthetic fixture
4. real-workload smoke
   - OpenTitan shard or sv-tests slice

Minimum policy:

1. no semantic change without a red-to-green lit test
2. no runner output change without schema/TSV regression updates
3. no timeout-classification change without reason-code regression tests

## 20. CI and Automation Plan

### Presubmit Gates

1. `check-circt` formal-focused lit subset:
   - `test/Tools/circt-bmc/*`
   - `test/Tools/circt-lec/*`
   - runner `test/Tools/run-*formal*`
2. schema tests:
   - formal result schema generation
   - manifest generation and parsing

### Periodic Gates (nightly)

1. OpenTitan connectivity LEC shard suite in Z3 mode
2. OpenTitan AES LEC suite in Z3 mode
3. sv-tests BMC tagged suites with fixed seeds

### Reporting

Nightly report should publish:

1. pass/fail/error/timeout counts by suite
2. timeout frontier top cases by solver_time_ms
3. unsupported diagnostic counts by category
4. drift vs previous baseline

## 21. Rollout and Compatibility

1. keep existing TSV outputs during migration
2. treat JSONL as source-of-truth once adapters are complete
3. announce schema version bumps with migration note
4. deprecate legacy fields in two-step process:
   - first release: emit both old and new
   - second release: remove old after consumers migrate

## 22. Quality and Refactor Cleanup Targets

These are explicit cleanup targets to avoid long-term code rot.

1. `utils/run_sv_tests_circt_bmc.sh`
   - split into reusable library + thin CLI
   - reduce line count by at least 30 percent
2. `utils/run_opentitan_connectivity_circt_lec.py`
   - isolate case generation from execution
   - isolate reason classification from command launch
3. dead-path cleanup
   - remove duplicate fallback paths superseded by common helpers
4. reason code normalization
   - one canonical enum source
   - no ad-hoc mixed-case reason names

## 23. Four-Week Execution Calendar

### Week 1

1. complete WS0-T1/T2/T3
2. freeze baseline artifacts
3. publish first drift report

### Week 2

1. complete WS1-T1/T2
2. move runners to shared launcher/classifier
3. verify no behavior drift on baseline

### Week 3

1. complete WS2-T1/T2 and WS3-T1/T2
2. land first multiclock and init-form closure patches
3. run OpenTitan BMC shard validation in Z3 mode

### Week 4

1. complete WS4-T1/T2 and WS5-T1/T2
2. run timeout frontier loop on OpenTitan LEC
3. publish milestone assessment for M1/M2/M3 readiness

## 24. Open Questions to Resolve Early

1. Should multiclock semantics in BMC be lockstep, event-driven, or hybrid by default?
2. Which aggregate legalization boundary belongs in LEC tool vs conversion pass?
3. Do we keep legacy runner-specific reason codes or enforce global enum now?
4. What is the strict resource budget for nightly OpenTitan Z3 runs?
5. Which dashboard consumers need compatibility adapters before schema-only switch?

## 25. Updated Next Action

Execute WS0-T2 plus WS0-T3 immediately:

1. capture frozen baseline manifests for OpenTitan AES LEC, connectivity LEC, and sv-tests BMC
   - use `write_ws0_baseline_manifest.py` for canonical command presets
2. run each baseline three times using `capture_formal_baseline.py`
3. emit drift report from schema rows via `compare_formal_results_drift.py`
4. block further semantic refactors until baseline drift is understood

## 26. Formal Result Schema Contract v1 (Normative)

Normative schema/versioning details are now maintained in:
`docs/FormalResultsSchema.md`

This plan keeps only WS6 execution references:

1. required fields and enum contracts are versioned under schema v1
2. strict-contract mode is opt-in and enforces cross-row invariants
3. version bumps are required for enum/required-field contract changes

## 27. Reason Code Taxonomy v1

Reason codes are the central unit for drift and timeout analysis.

### BMC Families

1. frontend:
   - `FRONTEND_COMMAND_TIMEOUT`
   - `FRONTEND_PARSE_ERROR`
   - `FRONTEND_RESOURCE_GUARD_RSS`
   - `FRONTEND_OUT_OF_MEMORY`
2. lowering:
   - `BMC_MULTICLOCK_UNSUPPORTED`
   - `BMC_REGISTER_INIT_UNSUPPORTED`
   - `BMC_SMT_EXPORT_UNSUPPORTED`
3. solver:
   - `SOLVER_COMMAND_TIMEOUT`
   - `SOLVER_UNKNOWN`
   - `SOLVER_RESOURCE_EXHAUSTED`
4. result:
   - `ASSERTION_VIOLATION`
   - `PROVEN_UNSAT`
   - `COVER_WITNESS_FOUND`
   - `NO_PROPERTY`

### LEC Families

1. frontend:
   - `FRONTEND_COMMAND_TIMEOUT`
   - `TOK_PACKAGESEP_PARSE_ERROR`
   - `TIMESCALE_REQUIRED`
2. lowering:
   - `LLHD_REF_UNSUPPORTED`
   - `LLVM_AGGREGATE_CONVERSION_UNSUPPORTED`
   - `VPI_ATTR_PARSE_ERROR`
3. solver:
   - `SOLVER_COMMAND_TIMEOUT`
   - `SMTLIB_EXPORT_ERROR`
4. result:
   - `EQ`
   - `NEQ`
   - `UNKNOWN`
   - `XPROP_ONLY`

### Mapping Policy

1. every non-pass row must map to one taxonomy code
2. legacy ad-hoc reason text is allowed only behind compatibility projection
3. taxonomy changes require update to:
   - schema validator tests
   - drift classifier tests
   - nightly dashboard mapping

## 28. Reproducibility and Drift Protocol

This protocol governs WS0 baseline freeze.

### Inputs to Freeze

1. tool binaries and versions
2. runner scripts and CLI args
3. timeout budgets
4. resource limits
5. seed/shard selection
6. target manifests/rules manifests

### Drift Computation

For each case, compare run A vs run B on:

1. `status`
2. `reason_code`
3. `stage`

Classify drift:

1. `NO_DRIFT`
2. `STATUS_DRIFT`
3. `REASON_DRIFT`
4. `STAGE_DRIFT`
5. `MISSING_CASE`
6. `NEW_CASE`

### Gate Criteria

1. required for WS0 completion:
   - zero `STATUS_DRIFT`
   - zero `MISSING_CASE`
2. allowed temporarily:
   - bounded `REASON_DRIFT` if reason normalization is in flight
3. blocked:
   - any increase in `TIMEOUT` count without explicit waiver

## 29. Baseline Command Templates

These are canonical templates used for baseline capture. Paths can be overridden by environment.

### OpenTitan AES LEC (Z3)

```bash
OUT=out/aes_lec.tsv \
FORMAL_RESULTS_JSONL_OUT=out/aes_lec.jsonl \
LEC_RUN_SMTLIB=1 \
LEC_SMOKE_ONLY=0 \
utils/run_opentitan_circt_lec.py \
  --opentitan-root ~/opentitan \
  --results-file "$OUT" \
  --results-jsonl-file "$FORMAL_RESULTS_JSONL_OUT"
```

### OpenTitan Connectivity LEC (Z3)

```bash
OUT=out/connectivity_lec.tsv \
FORMAL_RESULTS_JSONL_OUT=out/connectivity_lec.jsonl \
LEC_RUN_SMTLIB=1 \
LEC_SMOKE_ONLY=0 \
utils/run_opentitan_connectivity_circt_lec.py \
  --target-manifest <target.tsv> \
  --rules-manifest <rules.tsv> \
  --opentitan-root ~/opentitan \
  --results-file "$OUT" \
  --results-jsonl-file "$FORMAL_RESULTS_JSONL_OUT"
```

### sv-tests BMC (Z3)

```bash
OUT=out/sv_bmc.tsv \
FORMAL_RESULTS_JSONL_OUT=out/sv_bmc.jsonl \
BMC_SMOKE_ONLY=0 \
BMC_RUN_SMTLIB=1 \
utils/run_sv_tests_circt_bmc.sh ~/sv-tests
```

## 30. Dependency Map and Critical Path

Dependencies:

1. WS0-T1 is prerequisite for WS0-T2 and WS0-T3.
2. WS0-T3 is prerequisite for broad WS1 refactors.
3. WS1-T1 and WS1-T2 are prerequisite for WS1-T3.
4. WS2-T1 is prerequisite for WS2-T2 and WS2-T3.
5. WS3-T1 is prerequisite for WS3-T2.
6. WS4-T1 is prerequisite for WS4-T2 and WS4-T3.
7. WS5-T1 is prerequisite for WS5-T2.
8. WS6-T1 is prerequisite for WS6-T2 and WS6-T3.

Critical path for near-term parity:

1. WS0-T1 -> WS0-T2 -> WS0-T3
2. WS1-T1 -> WS1-T2
3. WS2-T1 -> WS2-T2
4. WS4-T1 -> WS4-T2
5. WS5-T1 -> WS5-T2
6. WS6-T2 -> nightly schema-only reporting

## 31. Milestone Readiness Checklists

### M1 Checklist (WS0 + WS1)

1. baseline manifests committed and reproducible
2. runner behavior drift report is clean
3. shared launcher + classifier integrated in all primary runners
4. no net loss in lit coverage

### M2 Checklist (WS2 + WS3)

1. multiclock unsupported count reduced on tracked suite
2. register-init unsupported count reduced on tracked suite
3. no soundness regressions in existing bmc lit tests
4. all new behavior guarded by red-to-green tests

### M3 Checklist (WS4 + WS5)

1. LLHD/ref C1 bucket burn-down complete
2. aggregate legalization covers top OpenTitan patterns
3. timeout frontier depth improves at fixed budget
4. unsupported diagnostics are explicit and categorized

### M4 Checklist (WS6)

1. schema validator runs in presubmit
2. schema-only nightly reports generated successfully
3. legacy TSV consumers migrated or adapter-shimmed
4. dashboard and drift tooling consume JSONL contract only

## 32. PR Slicing Strategy

Each slice should be reviewable and reversible.

1. one behavior change per PR when possible
2. tests first:
   - red lit test
   - implementation
   - green lit + integration slice
3. include engineering-log entry with:
   - realization
   - root cause
   - command-level validation
4. avoid mixing:
   - semantic fixes
   - large refactors
   in the same PR unless required

Suggested first 8 PRs:

1. PR1: WS0-T1 schema/manifest validation hardening
2. PR2: WS0-T2 baseline capture scripts
3. PR3: WS0-T3 drift comparator + report
4. PR4: WS1-T1 shared launcher extraction
5. PR5: WS1-T2 reason classifier extraction
6. PR6: WS2-T1 multiclock unsupported inventory tests
7. PR7: WS3-T1 register-init inventory tests
8. PR8: WS6-T2 strict schema validator CLI

## 33. Performance Guardrails

To prevent regressions while enabling deeper proofs:

1. define fixed timeout budgets per suite:
   - AES LEC
   - connectivity LEC
   - sv-tests BMC
2. track:
   - total solver time
   - P50/P90/P99 solver time
   - timeout count
3. block merges that:
   - increase timeout count by more than 5 percent
   - increase solver-time P90 by more than 15 percent
   unless approved with a documented tradeoff

## 34. Decision Record Template

For all material semantic decisions (especially WS2/WS4/WS5), include:

1. context
2. options considered
3. selected approach
4. soundness implications
5. performance implications
6. test evidence
7. rollback strategy

Decision records should live in engineering log entries and reference ticket IDs.

## 35. Next 72-Hour Plan

1. Implement WS0-T2:
   - add concrete baseline capture wrappers
   - produce first baseline artifacts for AES LEC, connectivity LEC, sv-tests BMC
2. Implement WS0-T3:
   - build drift report from schema JSONL
   - run three baseline repetitions
3. Start WS1-T1:
   - extract shared launcher utility
   - migrate one runner behind compatibility interface
4. Publish short status update:
   - drift summary
   - blocked items
   - first proposed WS2/WS3 test list
