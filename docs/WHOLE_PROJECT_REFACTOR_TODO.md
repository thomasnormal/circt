# Whole-Project Refactor TODO Tracker

Last updated: February 20, 2026

Status legend:
- `[ ]` not started
- `[~]` in progress
- `[x]` done

## Phase 0: Contracts and Baselines
- [x] Document stable runner contract for mutation workflows.
- [x] Document stable runner contract for formal workflows.
- [x] Document stable runner contract for simulation workflows.
- [x] Add golden contract lit tests for mutation.
- [x] Add golden contract lit tests for formal.
- [x] Add golden contract lit tests for simulation.

## Phase 1: Runner Infrastructure Unification
- [x] Create `utils/lib/common.sh` for shared helper functions.
- [~] Move tool/timeout/retry helpers to shared libs.
- [~] Move numeric parsing/validation helpers to shared libs.
- [~] Move canonicalization/hash helpers to shared libs.
- [~] Convert mutation scripts to shared runner helpers.
- [x] Adopt shared tool-resolution wrappers across mutation global-filter runners.
- [~] Convert formal scripts to shared runner helpers.
- [x] Adopt shared hash wrappers across formal BMC/LEC runners.
- [x] Adopt shared z3 preflight tool-resolution wrappers across formal runners.
- [~] Convert simulation wrappers to shared runner helpers.

## Phase 2: Mutation Stack Refactor
- [x] Extract native mutation planner to `utils/mutation_mcy/lib/native_mutation_plan.py`.
- [x] Extract native mutation planner shell glue to `utils/mutation_mcy/lib/native_mutation_plan.sh`.
- [x] Externalize native mutation rewrite template.
- [x] Introduce profile API entrypoint (`run_mutation_mcy_examples_api.sh`).
- [x] Route native-real wrapper through API entrypoint.
- [x] Add shared fake mutation runner helper in `test/Tools/Inputs/`.
- [x] Extract manifest parsing from `run_mutation_mcy_examples.sh`.
- [x] Extract args/validation from `run_mutation_mcy_examples.sh`.
- [x] Extract worker lifecycle from `run_mutation_mcy_examples.sh`.
- [x] Extract baseline/drift logic from `run_mutation_mcy_examples.sh`.
- [x] Share extracted modules with `run_mutation_matrix.sh`.
- [x] Share extracted modules with `run_mutation_cover.sh`.

## Phase 3: Formal Workflow Unification
- [x] Define declarative suite manifests for `sv-tests`.
- [x] Define declarative suite manifests for `verilator-verification`.
- [x] Define declarative suite manifests for `yosys/tests`.
- [x] Define declarative target manifests for OpenTitan formal flows.
- [x] Extract shared formal launch/retry/drift helpers.
- [x] Migrate BMC wrappers to shared formal helpers.
- [x] Migrate LEC wrappers to shared formal helpers.

## Phase 4: `circt-sim` Internal Modularization
- [x] Map high-churn code zones in `LLHDProcessInterpreter*`.
- [~] Extract memory model subsystem into dedicated sources.
- [~] Extract drive/update subsystem into dedicated sources.
- [~] Extract call/call_indirect interception subsystem.
- [~] Extract UVM adapter/interceptor subsystem.
- [~] Extract tracing/diagnostics subsystem.
- [x] Extract fork/join diagnostic emission helpers into tracing source.
- [x] Extract disable_fork diagnostic emission helpers into tracing source.
- [x] Extract fork intercept/create diagnostics into tracing source.
- [x] Extract process-finalize diagnostics into tracing source.
- [x] Extract wait-sensitivity diagnostics into tracing source.
- [x] Extract wait-event cache/no-op diagnostics into tracing source.
- [x] Extract call-indirect site-cache diagnostics into tracing source.
- [x] Extract interface-sensitivity diagnostics into tracing source.
- [x] Extract interface field-shadow scan diagnostics into tracing source.
- [x] Extract interface-overwrite diagnostics into tracing source.
- [x] Extract multi-driver post-NBA diagnostics into tracing source.
- [x] Extract module-drive execution/trigger diagnostics into tracing source.
- [x] Extract module-drive multi-driver diagnostics into tracing source.
- [x] Extract child-instance discovery/registration diagnostics into tracing source.
- [x] Extract child-instance input mapping/mismatch diagnostics into tracing source.
- [x] Extract signal discovery/registration diagnostics into tracing source.
- [x] Extract per-signal init/registration diagnostics into tracing source.
- [x] Extract signal export/top-process registration diagnostics into tracing source.
- [x] Extract interface-propagation diagnostics into tracing source.
- [x] Extract interface tri-state rule diagnostics into tracing source.
- [x] Extract conditional-branch diagnostics into tracing source.
- [x] Extract firreg update diagnostics into tracing source.
- [x] Extract instance-output diagnostics into tracing source.
- [x] Extract instance-output dependency diagnostics into tracing source.
- [x] Extract combinational trace-through diagnostics into tracing source.
- [x] Extract continuous fallback diagnostics into tracing source.
- [x] Extract drive schedule diagnostics into tracing source.
- [x] Extract array-drive remap diagnostics into tracing source.
- [x] Extract drive failure diagnostics into tracing source.
- [x] Extract array-drive scheduling diagnostics into tracing source.
- [x] Extract I3C address-bit diagnostics into tracing source.
- [x] Extract I3C ref-cast diagnostics into tracing source.
- [x] Extract I3C cast-layout diagnostics into tracing source.
- [x] Extract ref-arg resolve diagnostics into tracing source.
- [x] Extract I3C field-drive signal-struct diagnostics into tracing source.
- [x] Extract I3C field-drive mem-struct diagnostics into tracing source.
- [x] Extract I3C field-drive array/mem diagnostics into tracing source.
- [x] Extract I3C config-handle diagnostics into tracing source.
- [x] Extract I3C handle-call diagnostics into tracing source.
- [x] Extract I3C call-stack-save diagnostics into tracing source.
- [x] Extract I3C to-class argument diagnostics into tracing source.
- [x] Extract config-db func.call diagnostics into tracing source.
- [x] Extract sequencer func.call diagnostics into tracing source.
- [x] Extract analysis write func.call diagnostics into tracing source.
- [x] Extract UVM run_test entry diagnostics into tracing source.
- [x] Extract get_name loop diagnostics into tracing source.
- [x] Extract baud fast-path reject/hit/null-self-stall diagnostics into tracing source.
- [x] Extract baud fast-path gep/missing-fields diagnostics into tracing source.
- [x] Extract baud fast-path batch-parity/batch-mismatch/batch-schedule diagnostics into tracing source.
- [x] Extract get_name LLVM loop diagnostics into tracing source.
- [x] Extract function-cache shared hit/store diagnostics into tracing source.
- [x] Extract phase-order diagnostics into tracing source.
- [x] Extract mailbox tryput/get diagnostics into tracing source.
- [x] Extract randomize diagnostics into tracing source.
- [x] Extract tail-wrapper fast-path diagnostics into tracing source.
- [x] Extract on-demand-load diagnostics into tracing source.
- [x] Extract struct-inject X diagnostics into tracing source.
- [x] Extract AHB transaction field diagnostics into tracing source.
- [x] Extract fread diagnostics into tracing source.
- [x] Extract function-progress diagnostics into tracing source.
- [x] Extract function-step-overflow diagnostics into tracing source.
- [x] Extract assertion-failure diagnostics into tracing source.
- [x] Extract UVM run_test re-entry diagnostics into tracing source.
- [x] Extract func.call internal-failure warning diagnostics into tracing source.
- [x] Extract execute-loop step-limit/overflow diagnostics into tracing source.
- [x] Extract sim.terminate trigger diagnostics into tracing source.
- [x] Extract UVM JIT promotion-candidate diagnostics into tracing source.
- [x] Extract interface propagation-map diagnostics into tracing source.
- [x] Extract interface auto-link diagnostics into tracing source.
- [x] Extract interface intra-link diagnostics into tracing source.
- [x] Extract interface tri-state candidate diagnostics into tracing source.
- [x] Extract interface copy/signal-copy diagnostics into tracing source.
- [x] Extract interface auto-link discovery diagnostics into tracing source.
- [x] Extract interface field-signal dump diagnostics into tracing source.
- [~] Add focused unit/lit slices per extracted subsystem.

## Phase 5: Schema and Reporting Convergence
- [ ] Add shared schema utility library.
- [ ] Unify schema version/contract emission.
- [ ] Unify summary field naming conventions.
- [ ] Add schema migration compatibility tests.
- [ ] Add strict schema drift gates for all major workflows.

## Phase 6: Test Harness Infrastructure
- [ ] Expand reusable fake tool helpers (`test/Tools/Inputs`).
- [ ] Migrate mutation tests off inline shell fixtures.
- [ ] Migrate formal wrapper tests off inline shell fixtures.
- [ ] Migrate simulation wrapper tests off inline shell fixtures.


## Phase 6.5: Unified Regression Orchestrator
- [x] Define unified suite-manifest schema for AVIP, sv-tests, verilator-verification, yosys/tests, OpenTitan, and Ibex.
- [x] Add unified orchestrator entrypoint (e.g. `utils/run_regression_unified.sh`) with `smoke|nightly|full` profiles.
- [x] Add engine modes (`circt|xcelium|both`) with shared suite/adaptor contract.
- [x] Add paired CIRCT-vs-Xcelium baseline comparison report lanes.
- [x] Add sharding/resume/retry support for long-running full-fleet runs.
- [x] Add orchestrator contract tests and CLI golden tests.
- [x] Add bounded parallel lane execution (`--jobs`) for orchestrator throughput.
- [x] Add adapter catalog + adapter-driven manifest lanes for suite command normalization.
- [x] Expand CVDP dataset lanes (commercial/non-commercial, smoke-limited, compile-only, golden smoke variants).
- [x] Add bounded smoke-only lanes for sv-tests BMC/LEC, verilator BMC/LEC, yosys BMC/LEC, and AVIP verilog.
- [ ] Add per-suite dataset budgeting policy (profile-level caps and runtime budgets) to keep `full` runs scalable.

## Phase 7: Ownership and Docs
- [x] Add subsystem ownership map for runner/formal/sim domains.
- [x] Add maintenance playbook for baseline/schema updates.
- [x] Keep changelog concise and link deep iteration notes.

## Recurring Validation Tasks
- [ ] Run focused mutation lit slices after each mutation refactor PR.
- [ ] Run focused formal lit slices after each formal refactor PR.
- [ ] Run focused `circt-sim` lit/unit slices after each sim refactor PR.
- [ ] Run `~/mcy/examples` native-real regression periodically.
- [ ] Run representative suite checks for `~/sv-tests/`, `~/verilator-verification/`, `~/yosys/tests/`, and OpenTitan according to phase scope.
- [ ] Run unified orchestrator smoke profile with paired CIRCT/Xcelium checks periodically.

## Notes
- Keep refactor PRs behavior-preserving unless migration flags are explicitly introduced.
- Update this file in every refactor PR.
