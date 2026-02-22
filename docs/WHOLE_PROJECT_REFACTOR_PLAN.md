# Whole-Project Refactor Plan (Execution Plan)

Last updated: February 20, 2026

## Goals
1. Reduce maintenance cost and merge conflict rate across `tools/`, `utils/`, and regression harnesses.
2. Make behavior deterministic and observable via stable runner/report contracts.
3. Preserve throughput and correctness while improving extensibility.
4. Keep formal, simulation, and mutation flows aligned with shared infrastructure.
5. Introduce a unified full-fleet regression harness for AVIP/sv-tests/OpenTitan/Ibex and related suites.

## Non-Goals
1. Rewriting all shell runners in one change.
2. Changing user-facing semantics without explicit migration paths.
3. Large architectural churn without regression gates.

## Refactor Principles
1. Behavior-preserving first, behavior-changing only behind explicit gates.
2. Small PR slices with measurable acceptance criteria.
3. Shared library extraction before feature expansion.
4. Contract-first: CLI, schema, and artifact stability documented and tested.
5. Each extraction step must keep focused lit/regression coverage green.

## Current Pain Points
1. Monolithic runner scripts duplicate parsing, retries, timeout, schema, and drift logic.
2. Similar formal/sim/mutation workflows are implemented in separate script stacks.
3. `circt-sim` implementation hotspots are concentrated in very large files.
4. Test harness logic is repeated in many lit tests, increasing brittleness.
5. Report schema/version handling is partially duplicated across workflows.

## Target End State
1. Shared runner libraries in `utils/lib/` and `utils/<domain>/lib/`.
2. Stable API entrypoints for runner profiles and policy bundles.
3. Modular `circt-sim` runtime internals grouped by subsystem.
4. Shared schema compatibility checks for summary/report artifacts.
5. Reusable lit fixtures/helpers replacing duplicated inline harness scripts.

## Execution Phases

### Phase 0: Contract Freeze and Baseline Gates
Objective: lock external behavior before heavy internal refactor.

Steps:
1. Document runner contract for each major workflow:
   - mutation (`run_mutation_*`),
   - formal (`run_*_circt_bmc.py`, `run_*_circt_lec.py`),
   - simulation (`run_avip_circt_sim.sh`, `run_sv_tests_circt_sim.sh`).
2. Define stable artifact contract files:
   - required columns,
   - schema versioning fields,
   - failure semantics.
3. Add focused "golden behavior" tests that assert contract invariants.

Exit criteria:
1. Contract docs checked in.
2. Golden tests for each workflow pass in CI/local lit slices.

---

### Phase 1: Runner Infrastructure Unification
Objective: remove duplicate runner plumbing and centralize policy mechanics.

Steps:
1. Create shared shell utilities in `utils/lib/`:
   - tool resolution,
   - timeout/retry wrappers,
   - numeric parsing/validation,
   - canonical path handling,
   - hash helpers.
2. Introduce stable profile-based API launchers where missing:
   - `default`,
   - strict policy variants (`native-real`, strict formal gates, etc.).
3. Route existing convenience wrappers through API entrypoints.
4. Keep compatibility shims for old scripts until migration complete.

Exit criteria:
1. No direct duplicate implementations of core helpers across runner files.
2. Existing wrapper UX unchanged.
3. Focused runner lit suites pass.

---

### Phase 2: Mutation Stack Refactor (Current Workstream)
Objective: complete decomposition of `run_mutation_mcy_examples.sh` and sibling scripts.

Steps:
1. Complete module extraction:
   - `args`/validation,
   - manifest parsing,
   - worker lifecycle,
   - baseline/drift evaluation,
   - reporting/schema writers.
2. Keep mutation planning and rewrite logic in dedicated modules.
3. Expand shared test harness helper usage in mutation lit tests.
4. Add cross-script shared modules between `run_mutation_mcy_examples.sh`,
   `run_mutation_matrix.sh`, and `run_mutation_cover.sh`.

Exit criteria:
1. `run_mutation_mcy_examples.sh` reduced to orchestration + module wiring.
2. Mutation lit slices pass at parity with pre-refactor behavior.
3. Real `~/mcy/examples` native-real run remains green.

---

### Phase 3: Formal Workflow Unification (BMC/LEC)
Objective: unify suite orchestration behind shared engine and declarative suite metadata.

Steps:
1. Define suite manifests for:
   - `~/sv-tests/`,
   - `~/verilator-verification/`,
   - `~/yosys/tests/`,
   - OpenTitan targets.
2. Refactor duplicated launch/retry/drift code into shared formal libs.
3. Standardize baseline schema and drift checks across all formal runners.
4. Add migration tooling for existing baselines.

Exit criteria:
1. Shared formal orchestration helpers used by all suite drivers.
2. Formal parity checks and drift gates retain existing behavior.

---

### Phase 4: `circt-sim` Internal Modularization
Objective: reduce churn concentration and improve testability of simulator runtime.

Steps:
1. Identify high-churn zones in `LLHDProcessInterpreter*`.
2. Split into subsystem units (behavior-preserving extraction):
   - memory model and backing blocks,
   - drive/store/update semantics,
   - call/call_indirect interceptors,
   - UVM helper layers,
   - tracing/diagnostics.
3. Introduce local unit targets per subsystem where feasible.
4. Keep public behavior and command line unchanged.

Exit criteria:
1. Core runtime functionality remains parity-stable.
2. File-level ownership boundaries are clearer and conflict rate decreases.

---

### Phase 5: Schema and Reporting Convergence
Objective: standardize reporting for mutation/formal/sim workflows.

Steps:
1. Define a common schema utility for:
   - version emission,
   - contract fingerprints,
   - compatibility checks.
2. Standardize summary/report field naming conventions.
3. Add strict schema migration tooling and tests.

Exit criteria:
1. Shared schema helper adopted by major workflows.
2. Schema drift behavior is deterministic and documented.

---

### Phase 6: Test Harness Infrastructure Refactor
Objective: reduce repeated inline shell logic in lit tests.

Steps:
1. Add reusable fake tools and fixture helpers under `test/Tools/Inputs/`.
2. Convert high-duplication tests first (mutation/formal wrappers).
3. Add helper contracts (env knobs, required flags, failure injection).

Exit criteria:
1. Significant reduction in duplicated test shell bodies.
2. Better readability and lower fixture maintenance cost.

---
### Phase 6.5: Unified Regression Orchestrator
Objective: provide one entrypoint to run AVIP, sv-tests, OpenTitan, Ibex, and related suites with consistent policy/reporting, including CIRCT vs Xcelium baseline parity lanes.

Steps:
1. Define a declarative suite-manifest schema covering:
   - AVIP suites,
   - sv-tests,
   - verilator-verification,
   - yosys/tests,
   - OpenTitan,
   - Ibex.
2. Add a unified orchestrator entrypoint in `utils/` with profile modes:
   - smoke,
   - nightly,
   - full.
3. Add backend-engine modes for parity runs:
   - `--engine circt`,
   - `--engine xcelium`,
   - `--engine both` (paired baseline comparison).
4. Add shared per-suite adapters with common preflight/timeout/retry policy.
5. Add sharding, resume, and bounded-retry support for long-running fleets.
6. Emit unified summary/report artifacts with schema/contract checks, including engine-parity summaries.

Exit criteria:
1. A single command can launch selected suite sets with deterministic config.
2. The same suite manifest can run under CIRCT and Xcelium engines.
3. Unified reports include per-suite + aggregate pass/fail/infra metrics and engine-parity deltas.
4. Focused orchestrator contract tests pass.

---

### Phase 7: Ownership, Docs, and Maintenance Model
Objective: make long-term maintenance predictable.

Steps:
1. Add subsystem ownership map for `utils/`, `tools/circt-sim/`, formal/mutation runners.
2. Keep `CHANGELOG.md` concise and move deep iteration logs to linked artifacts.
3. Add playbooks for common maintenance operations (baseline updates, schema migrations).

Exit criteria:
1. Ownership map and maintenance playbooks are in-tree.
2. New contributor path for runner/sim/formal work is clearer.

## Risk Controls
1. Every phase uses focused regression gates before/after extraction.
2. Keep compatibility wrappers until adoption is complete.
3. Avoid multi-domain refactors in a single PR.
4. Keep migration scripts for any schema changes.

## Validation Matrix (Minimum)
1. Mutation:
   - focused mutation lit slices,
   - `~/mcy/examples` native-real smoke/regression.
2. Formal:
   - focused BMC/LEC wrapper tests,
   - representative suite dry-runs.
3. Simulation:
   - focused `circt-sim` lit + unit slices,
   - bounded AVIP smoke lanes.
4. Unified Fleet:
   - orchestrator smoke profile across AVIP/sv-tests/OpenTitan/Ibex subsets,
   - paired CIRCT vs Xcelium baseline lane checks on selected suites.

## Suggested PR Sequence
1. PR-1: shared runner helpers + mutation script adoption (behavior-preserving).
2. PR-2: mutation manifest/worker/baseline module extraction.
3. PR-3: formal shared helper extraction.
4. PR-4: schema utility convergence.
5. PR-5: unified regression orchestrator skeleton + manifest schema.
6. PR-6+: staged `circt-sim` source modularization.

## Status Tracking
Execution status is tracked in:
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`
