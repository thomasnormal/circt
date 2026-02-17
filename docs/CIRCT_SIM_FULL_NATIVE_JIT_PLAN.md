# circt-sim Full Native JIT Program (Big-Bang Rollout, Strict End-State)

## Summary
Build a full native JIT execution stack for `circt-sim` in one major program, with these locked decisions:

1. Timeline: Big-bang program (broad JIT coverage, not tiny incremental pilots).
2. Correctness bar: Bit-exact parity versus interpret mode before default-on.
3. Strictness: Strict native-only is the end-state, not day-one; initial compile mode allows controlled deopt/fallback to preserve parity while coverage closes.

This plan is decision-complete and implementation-ready.

---

## Scope and Success Criteria

### In Scope
1. Native JIT for `circt-sim` process execution (LLHD/Seq/Moore/Core execution paths used in simulation).
2. Scheduler-compatible resumable execution (wait/delta/call-stack suspension semantics).
3. Full runtime integration: VPI, DPI, UVM dynamic dispatch behavior, coverage/reporting hooks.
4. Compile-mode telemetry and deopt governance.
5. Cross-suite parity/performance gating:
   - `~/mbit/*avip*`
   - `~/sv-tests/`
   - `~/verilator-verification/`
   - `~/yosys/tests/`
   - `~/opentitan/`

### Out of Scope
1. Removing interpreter code immediately.
2. Strict-native from first release.
3. AVIP-specific hardcoded hacks as primary architecture.

### Exit Criteria (Program Complete)
1. Compile mode is default-on candidate with parity green.
2. Strict-native mode exists and passes target strict gates (zero allowed deopts on governed suite).
3. Performance and memory targets met on AVIP-heavy workloads (especially AHB-class stress).

---

## Why Strict Day-One Is Not Viable (Grounded Constraints)
Current codepaths show strict-native first cut would be high risk because:
1. Dynamic call paths and vtable-heavy UVM behaviors currently rely on fallback/intercept patterns.
2. Scheduler-coupled wait/resume semantics are complex and must be compiled as resumable state machines.
3. Some paths are explicitly partial/unsupported today (for example, combinational registration TODO and unsupported verif clocked handling behavior in sim path).
4. VPI/DPI callback boundary behavior is tightly coupled to runtime scheduler phases.

Therefore: strict-native is feasible as convergence phase, not first activation phase.

---

## Architecture Plan

## 1) Execution Model
1. Add a JIT execution tier for process bodies/functions.
2. Keep interpreter as controlled deopt target during convergence.
3. Add strict-native mode later: deopt becomes error under configured policy.

## 2) Core Components
1. JITCompileManager
   - Owns module/function lowering, ORC engine sessions, compiled thunk cache.
2. ProcessThunk ABI
   - Stable ABI for process activation with resume state, scheduler hooks, runtime calls.
3. DeoptBridge
   - Transfers process state between compiled frame and interpreter frame.
4. JITDispatchRegistry
   - Canonical dispatch table for `func.call`, `call_indirect`, and runtime helpers.
5. Compile-Mode Governor
   - Hotness thresholds, compile budget, fail-on-deopt policy, telemetry export.

## 3) Compilation Strategy
1. First-class compilation unit: hot function/process regions with resumable control-flow lowering.
2. Cache key: function symbol + type specialization + relevant runtime guard version.
3. Guards: runtime assumptions (vtable shape/class id/type checks/signal encoding invariants).
4. On guard failure: deopt with state transfer (until strict phase).

## 4) Scheduler/Wait Semantics
1. Compile `wait`/delay/event-sensitivity as explicit state-machine checkpoints.
2. Preserve exact delta-cycle ordering and wake conditions.
3. Preserve call-stack resume semantics for nested call/call_indirect suspension.

## 5) Runtime/Foreign Interface Compatibility
1. Keep VPI callback phase points identical to interpret mode.
2. Preserve DPI ABI and shared-lib behavior.
3. Preserve coverage/runtime call semantics and report timing.
4. Preserve `sim.terminate` and finish-grace behavior.

---

## Public APIs / Interfaces / Types to Add or Change

## CLI / Env Contracts
1. `--mode=compile` becomes true JIT mode (not alias behavior).
2. Add:
   - `--jit-hot-threshold=<N>`
   - `--jit-compile-budget=<N>`
   - `--jit-cache-policy=<...>`
   - `--jit-fail-on-deopt` (strict gate)
   - `--jit-report=<path>`
3. Env equivalents:
   - `CIRCT_SIM_JIT_HOT_THRESHOLD`
   - `CIRCT_SIM_JIT_COMPILE_BUDGET`
   - `CIRCT_SIM_JIT_CACHE_POLICY`
   - `CIRCT_SIM_JIT_FAIL_ON_DEOPT`
   - `CIRCT_SIM_JIT_REPORT_PATH`

## Telemetry Schema (machine-readable, stable)
1. `jit_compiles_total`
2. `jit_cache_hits_total`
3. `jit_exec_hits_total`
4. `jit_deopts_total`
5. `jit_deopt_reason_*`
6. `jit_deopt_processes[]` (`process_id`, `process_name`, `reason`)
7. `jit_compile_wall_ms`
8. `jit_exec_wall_ms`
9. `jit_strict_violations_total`

## Internal Runtime ABI
1. Introduce explicit compiled thunk signature for process activation/resume.
2. Introduce stable deopt state container (values, memory refs, resume PC, wait state).

---

## Work Breakdown (Decision-Complete)

## Implementation Progress Snapshot (February 17, 2026)
1. Phase A telemetry artifact writer started:
   - `circt-sim` now supports `--jit-report=<path>` and
     `CIRCT_SIM_JIT_REPORT_PATH`.
   - report includes stable scheduler/control sections, placeholder top-level
     JIT counters, and existing UVM fast-path/JIT-promotion counters.
2. Phase A deterministic parity harness started:
   - `utils/run_avip_circt_sim.sh` supports deterministic mode selection
     (`CIRCT_SIM_MODE=interpret|compile`) and optional extra args.
   - `utils/check_avip_circt_sim_mode_parity.py` compares interpret vs compile
     matrices on fixed key fields.
   - `utils/run_avip_circt_sim_mode_parity.sh` provides one-command parity run.
3. Regression coverage added for telemetry/parity harness smoke behavior.
4. Compile-mode governor scaffolding landed:
   - introduced `JITCompileManager` with stable counters and deopt reasons.
   - introduced initial process-thunk cache API
     (`install/has/execute/invalidate`) as the future ORC handoff seam.
   - compile manager now tracks per-process activation hotness and applies
     compile-budget gating for thunk installation attempts.
   - added CLI/env controls:
     - `--jit-hot-threshold` / `CIRCT_SIM_JIT_HOT_THRESHOLD`
     - `--jit-compile-budget` / `CIRCT_SIM_JIT_COMPILE_BUDGET`
     - `--jit-cache-policy`
     - `--jit-fail-on-deopt` / `CIRCT_SIM_JIT_FAIL_ON_DEOPT`
   - strict policy path now marks compile-mode thunk misses as deopt and
     enforces non-zero exit when `--jit-fail-on-deopt` is enabled.
   - deopt accounting now occurs at process dispatch when no compiled thunk is
     available (`missing_thunk` reason), rather than as a coarse end-of-run
     placeholder.
   - first real native thunk path landed for trivial terminating process bodies
     (`llhd.halt`-only, including halt-yield process-result paths,
     `sim.proc.print` + `llhd.halt`) and trivial initial-block shapes
     (`seq.yield`-only and `sim.proc.print` + `seq.yield`), including
     compile/install/dispatch counter updates.
   - introduced initial deopt-bridge seam:
     - explicit per-thunk execution-state ABI
     - process-state snapshot/restore on thunk-requested deopt
     - `guard_failed` telemetry path exercised by regression hook
       (`CIRCT_SIM_JIT_FORCE_DEOPT_REQUEST=1`).
   - first resumable native thunk pattern landed:
     - supports two-block `wait(delay) -> halt` process bodies with native
       suspend/resume across activations.
     - extended to support two-block
       `wait(delay) -> sim.proc.print -> halt` process bodies.
     - extended to support process-result terminal shapes with
       `wait yield (...)` and destination block operands feeding terminal
       `halt` yield operands.
     - extended to support event-sensitive two-block
       `wait(observed...) -> sim.proc.print -> halt` process bodies when the
       observed value is produced by a pre-wait `llhd.prb`.
     - extended to support multi-observed event waits when the observed list
       is produced by matching pre-wait probe sequences.
     - added dedicated regression coverage for event-sensitive process-result
       shapes (`wait yield (...)` + dest operands + terminal `halt` yields) in
       both single-observed and multi-observed wait forms, including strict
       guard-failed deopt variants.
     - matcher/executor now also support event-sensitive waits where observed
       operands are derived by pure pre-wait computations in the entry block,
       rather than only direct probe result operands.
     - matcher now enforces that pre-wait derived-observed preludes are pure
       (side-effect-free; `llhd.prb` allowed), so impure shapes are classified
       as `unsupported_operation` rather than being JIT-thunked.
     - wired per-process native resume tokens through thunk dispatch and deopt
       snapshot/restore to keep resumable state-machine handoff explicit.
     - periodic toggle clock native thunk now also uses explicit token-guarded
       activation phases and deopts on token/state mismatches.
   - deopt classification now distinguishes `missing_thunk` from
     `unsupported_operation` when compile is attempted but process shape is not
     yet supported.
   - cache policy governance now has explicit behavior:
     - `memory`: keep compiled process thunks resident (default).
     - `none`: evict process thunks after each execution to force
       re-install/recompile and expose no-cache behavior in telemetry.
   - `--jit-cache-policy` now validates accepted values (`memory`/`none`) and
     honors `CIRCT_SIM_JIT_CACHE_POLICY` with warning+fallback on invalid input.
   - JIT report now includes per-process first deopt reasons in
     `jit_deopt_processes`, including both `process_id` and `process_name`,
     enabling strict-mode triage to pinpoint which processes are still leaving
     native coverage.
   - strict fail-on-deopt diagnostics now print per-process deopt details in
     stderr (`id`, `name`, `reason`) to reduce triage latency in strict lanes.
5. Bounded integration parity smoke executed:
   - `AVIPS=jtag`, `SEEDS=1`, `COMPILE_TIMEOUT=120`, `SIM_TIMEOUT=120`.
   - mode-parity checker passed with one row per mode; both lanes hit the
     configured timeout bound.
6. Bounded compile-mode profiling smokes executed (`AVIPS=jtag`, `SEEDS=1`,
   `CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1`):
   - 90s and 180s bounds both reached timeout before graceful summary emission.
7. Refreshed bounded compile-mode smoke executed (`AVIPS=jtag`, `SEEDS=1`,
   `SIM_TIMEOUT=90`):
   - compile `OK`; bounded sim `TIMEOUT`.
8. Parallel runtime hardening gap identified:
   - minimal LLHD process tests under `--parallel=4` currently exhibit hangs
     and allocator corruption aborts in both interpret and compile modes.
   - strict-native convergence now explicitly depends on fixing parallel
     scheduler/runtime stability before multi-threaded parity gates can pass.
9. Parallel safety gate mitigation landed:
   - `--parallel` now defaults to stable sequential fallback with explicit
     warning in `circt-sim`.
   - experimental scheduler path remains available for hardening via:
     `CIRCT_SIM_EXPERIMENTAL_PARALLEL=1`.
   - added parallel-mode regression for resumable wait/yield process-result
     thunk shape to keep CLI compatibility covered while hardening continues.
   - added parallel-mode regression for derived-observed event-wait resumable
     process-result thunk shape to protect compile-mode coverage while the
     experimental parallel scheduler remains gated.

## Phase A: Foundation and Correctness Harness
1. Implement compile-mode telemetry framework and result artifact writer.
2. Create deterministic parity harness:
   - Interpret vs compile comparison on identical seeds.
   - Diff on exit status, messages, coverage summaries, critical traces.
3. Establish baseline dashboards for all five downstream suites.

Acceptance:
1. One-command parity run produces deterministic artifact bundle.
2. No manual parsing needed for regression verdict.

## Phase B: JIT Infrastructure (Broad, Not Pilot)
1. Land JITCompileManager + ORC integration.
2. Land thunk ABI + ProcessExecutionState mapping.
3. Land cache manager and invalidation semantics.
4. Wire compile mode to invoke compiled thunks where eligible.

Acceptance:
1. Compile mode executes native thunks on representative hot functions.
2. Deopt bridge works without semantic drift in covered tests.

## Phase C: Semantics Closure for Big-Bang Coverage
1. Close unsupported/partial execution semantics that block strict convergence:
   - combinational registration gap
   - clocked assertion/concurrent behavior policy in simulation path
   - any unhandled-op routes that would silently degrade results
2. Replace ad-hoc fallback branches with explicit guarded dispatch/deopt reasons.

Acceptance:
1. No silent semantic degradation in compile mode.
2. Every deopt has explicit reason code.

## Phase D: Runtime Integration Hardening
1. Preserve VPI callback order/phase behavior.
2. Preserve DPI interception and symbol resolution behavior.
3. Validate coverage/report lifecycle parity.

Acceptance:
1. VPI/DPI regression suites pass in interpret and compile.
2. Coverage summaries match parity expectations.

## Phase E: Big-Bang Rollout Gates
1. Run full matrix over AVIP/sv-tests/verilator/yosys/opentitan.
2. Tune thresholds/budget/cache and memory policy.
3. Produce release candidate with compile mode as supported production path.

Acceptance:
1. Bit-exact parity gates pass for governed deterministic workloads.
2. Performance gain and memory stability targets met.

## Phase F: Strict-Native Convergence
1. Add strict-native policy:
   - `--jit-fail-on-deopt` hard-gates.
2. Burn down deopt reasons until governed strict suite is zero-deopt.
3. Declare strict-native readiness when all strict gates are green.

Acceptance:
1. Strict mode passes required suite with zero deopts.
2. Strict violations fail CI by policy.

---

## Test Plan (Required for Every Fix/Feature)

## Unit/Component Tests
1. JIT cache key correctness and invalidation.
2. Thunk resume/deopt state transfer correctness.
3. Wait/delay/event wake semantics under compiled execution.
4. Runtime ABI function boundary checks.

## lit / Tool Regressions
1. Compile mode smoke tests for basic process execution.
2. Compile vs interpret parity tests for:
   - wait-condition loops
   - call_indirect/vtable dispatch
   - sim.terminate and finish-grace
   - DPI/VPI interaction
   - coverage runtime calls
3. New tests for every bug fix/deopt reason class.

## Integration Sweeps (gated)
1. `~/mbit/*avip*`: fixed-seed matrix, parity + perf + memory.
2. `~/sv-tests/`: pass/xfail drift and compile-vs-interpret parity.
3. `~/verilator-verification/`
4. `~/yosys/tests/`
5. `~/opentitan/`: targeted plus expanded profiles as stability increases.

---

## Rollout, CI, and Governance

1. CI lanes:
   - `circt-sim-interpret-parity`
   - `circt-sim-compile-parity`
   - `circt-sim-strict-native` (initially non-blocking, then blocking)
2. Changelog/process:
   - Update `docs/CHANGES.md` for each milestone and significant behavior change.
   - Update `PROJECT_PLAN.md` milestone status with dates/commit IDs.
3. Regression policy:
   - Any parity regression blocks rollout.
   - Any strict-native deopt in strict lane is failure once lane is promoted.

---

## Risks and Mitigations

1. Risk: semantic drift in dynamic UVM behavior.
   - Mitigation: explicit guards, deopt reasons, parity harness first.
2. Risk: compile overhead dominates.
   - Mitigation: hotness threshold, compile budgets, cache persistence policy.
3. Risk: memory blowup in AVIP-heavy runs.
   - Mitigation: JIT/heap telemetry + memory gates + AHB stress lane.
4. Risk: strict-native never closes.
   - Mitigation: deopt reason burn-down plan with ranked closure queue and ownership.

---

## Assumptions and Defaults (Locked)

1. Big-bang program scope is accepted.
2. Bit-exact parity is mandatory for rollout gates.
3. Strict-native is an explicit end-state phase, not day-one.
4. Initial compile mode may deopt; strict mode later forbids deopt.
5. Interpreter remains available during convergence for correctness.
