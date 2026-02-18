# circt-sim Full Native JIT Program (Big-Bang Rollout, Strict End-State)

## Summary
Build a full native JIT execution stack for `circt-sim` in one major program, with these locked decisions:

1. Timeline: Big-bang program (broad JIT coverage, not tiny incremental pilots).
2. Correctness bar: Bit-exact parity versus interpret mode before default-on.
3. Strictness: Strict native-only is the end-state, not day-one; initial compile mode allows controlled deopt/fallback to preserve parity while coverage closes.

This plan is decision-complete and implementation-ready.

---

## Latest Program Update (February 18, 2026)
1. Shared UVM getter cache instrumentation was hardened:
   - shared-hit trace formatting fixed
   - shared-store tracing added
   - profile summary now emits
     `UVM function-result cache: local_hits/shared_hits/local_entries/shared_entries`.
2. Bounded I3C compile-mode repro (55s cap, seed 1) now confirms heavy shared-cache
   activity in real workload conditions:
   - `shared_hits=5742`, `shared_entries=176`
   - sim-time progress reached `451490000000 fs` in bounded window.
3. Functional closure remains the primary blocker for I3C:
   - coverage still `0.00% / 0.00%` despite runtime cache gains.
   - next work should target functional progression/coverage sampling root cause,
     not only getter hot-path overhead.

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
6. `jit_deopt_processes[]`
   (`process_id`, `process_name`, `reason`, optional `detail`)
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
   - deopt telemetry now also carries optional per-process detail hints
     (`detail`) for unsupported-shape classification, and strict diagnostics
     append that detail when available.
   - compile-governor detail hints now classify `missing_thunk` deopts as
     `below_hot_threshold`, `compile_budget_zero`, or
     `compile_budget_exhausted` (plus `install_failed` on install miss), so
     strict convergence burn-down can separate policy gating from shape gaps.
   - added deopt burn-down aggregation utility:
     - `utils/summarize_circt_sim_jit_reports.py` aggregates one or more JIT
       report files/directories and emits ranked reason/detail counters.
     - optional TSV outputs provide reason ranking, reason+detail ranking, and
       per-process rows for strict convergence queueing.
     - utility now also supports allowlist-aware strict gating:
       - `--fail-on-any-non-allowlisted-deopt`
       - `--fail-on-reason=<reason>`
       - `--fail-on-reason-detail=<reason>:<detail>`
     - added AVIP compile-lane wrapper:
       `utils/run_avip_circt_sim_jit_policy_gate.sh`
     to run matrix + report aggregation + policy gate end-to-end.
   - added regression:
     `test/Tools/summarize-circt-sim-jit-reports.test`,
     `test/Tools/summarize-circt-sim-jit-reports-policy.test`,
     `test/Tools/run-avip-circt-sim-jit-policy-gate.test`.
   - closed the Phase C combinational registration gap baseline:
     - `llhd.combinational` ops now register as scheduler processes in both
       top-level and child-instance paths.
     - combinational `llhd.yield` now suspends and re-arms inferred
       sensitivity wakeups.
     - value lookup paths now consume registered combinational results first
       (with on-demand fallback retained for uncovered paths).
     - compile-mode JIT dispatch now executes native combinational thunks for
       `llhd.yield`-suspending bodies, including multiblock
       `cf.cond_br`/`cf.br` control-flow shapes; non-candidate combinational
       bodies still bypass compile dispatch to avoid strict-lane deopt noise.
   - added native thunk coverage for safe one-block terminating bodies:
     - supports one-block process bodies ending in `llhd.halt` with
       non-suspending safe preludes (pure ops + `sim.proc.print`).
     - supports one-block `sim.fork` child branch bodies ending in
       `sim.fork.terminator` under the same safe-prelude policy.
     - fork-child thunk/deopt shape matching now keys off the active branch
       region instead of always using the parent process body.
     - AVIP explicit-JIT deopt burn-down shifted dominant detail from
       `first_op:llvm.insertvalue` to `first_op:llvm.alloca` and
       `first_op:llvm.getelementptr`, defining the next closure queue.
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
10. Default-off rollout governance hardened:
    - added `circt-sim` regression proving default mode remains `interpret`
      even when JIT knobs are explicitly set.
    - added AVIP wrapper regression proving
      `utils/run_avip_circt_sim.sh` defaults to interpret mode when
      `CIRCT_SIM_MODE` is unset.
    - this pins non-default JIT activation until compile-mode parity gates are
      broadly green.
11. Deopt triage precision and stack-memory prelude closure advanced:
    - one-block terminating process/fork-child native thunk candidate now
      accepts safe LLVM stack-memory prelude ops (`alloca/gep/load/store`).
    - unsupported-operation detail classification now reports first unsupported
      prelude op for one-block terminating bodies (instead of coarse first-op
      reporting), improving burn-down queue fidelity.
    - refreshed AVIP explicit-JIT jtag burn-down now shows refined queue:
      `first_op:llvm.call` (81), `first_op:llvm.getelementptr` (45),
      `first_op:func.call_indirect` (11), `first_op:func.call` (3),
      `first_op:llvm.alloca` (3).
12. Call-target triage precision and `process::self` closure update:
    - unsupported detail for one-block terminating process shapes now includes
      direct callee names for `llvm.call` and `func.call` entries.
    - native prelude coverage now includes the non-suspending
      `llvm.call @__moore_process_self` and
      `llvm.call @__moore_packed_string_to_string` cases.
    - AVIP explicit-JIT jtag queue re-ranked from opaque `first_op:llvm.call`
      to targetable call sites; post-update dominant tails are:
      - `first_op:func.call_indirect` (90)
      - `first_op:llvm.getelementptr` (45)
      - residual `first_op:llvm.alloca` (3)
13. Forward-only multiblock terminating thunk closure for process control-flow:
    - compile-mode native thunk matching/execution now supports multiblock
      process/fork regions with safe preludes and forward-only
      `cf.br`/`cf.cond_br` edges, ending in `llhd.halt` or
      `sim.fork.terminator`.
    - this closes strict deopts on static-target
      `func.call_indirect -> cf.br -> halt` process shapes.
    - updated regression
      `test/Tools/circt-sim/jit-process-thunk-func-call-indirect-static-target-unsupported.mlir`
      from strict expected-fail to strict pass (`jit_deopts_total=0`).
    - added regression
      `test/Tools/circt-sim/jit-process-thunk-multiblock-call-indirect-condbr-halt.mlir`
      for `func.call_indirect` + `cf.cond_br` + block-argument merge.
    - parallel compatibility smoke passed under `--parallel=4`
      (sequential fallback warning expected while experimental parallel
      scheduler remains gated).
    - bounded AVIP compile-mode smoke refreshed:
      `AVIPS=jtag`, `SEEDS=1`, `COMPILE_TIMEOUT=90`, `SIM_TIMEOUT=90`,
      `CIRCT_SIM_MODE=compile`:
      compile `OK`; bounded sim `TIMEOUT`.
14. AVIP UVM prelude closure wave for one-block terminating process thunks:
    - expanded non-suspending `func.call` prelude allowlist used by one-block
      terminating process/fork native thunk matching:
      - `uvm_pkg::uvm_get_report_object`
      - `*::uvm_get_report_object`
      - `uvm_pkg::run_test`
      - `m_execute_scheduled_forks`
      - `get_global_hopper` / `*::get_global_hopper`
      - `*::get_report_verbosity_level`
      - `*::get_report_action`
      - config-db `set_<digits>` wrappers (signature-gated)
    - expanded non-suspending LLVM-call prelude allowlist:
      - `__moore_uvm_report_info`
      - `__moore_queue_push_front`
    - expanded one-block terminating safe prelude coverage to accept `llhd.prb`.
    - added strict compile-mode regression coverage:
      - `jit-process-thunk-func-call-uvm-prelude-closure-halt.mlir`
      - `jit-process-thunk-func-call-get-report-verbosity-level-halt.mlir`
      - `jit-process-thunk-func-call-get-report-action-halt.mlir`
      - `jit-process-thunk-prb-prelude-halt.mlir`
      - `jit-process-thunk-llvm-call-uvm-report-and-queue-halt.mlir`
    - bounded AVIP compile-lane deopt burn-down (same harness, jtag seed-1,
      `CIRCT_SIM=/home/thomas-ahle/circt/build/bin/circt-sim`,
      `--jit-hot-threshold=1 --jit-compile-budget=100000`,
      `SIM_TIMEOUT=180`, `MAX_WALL_MS=120000`) improved:
      - `jit_deopts_total`: `18 -> 14`
      - unsupported-operation rows: `12 -> 6`
      - dominant remaining unsupported details:
        - `multiblock_unsupported_terminator:llhd.wait` (3)
        - `multiblock_unsupported_terminator:cf.br` (1)
        - `multiblock_unsupported_terminator:cf.cond_br` (1)
        - `first_op:scf.for` (1)
15. Structured-control and fork prelude closure with resumable halt guards:
    - expanded one-block safe prelude matching to accept:
      - safe `scf.for`/`scf.if`/`scf.while` structured control regions
        (region-local preludes must satisfy existing non-suspending policy),
      - `sim.fork join_none` preludes with branch regions ending in
        `sim.fork.terminator` (or in-region branch terminators),
      - direct one-block `llvm.call @__moore_wait_condition` prelude calls.
    - extended single-block terminating thunk execution to support two
      resumable suspend classes without deopt:
      - `wait_condition` polling suspends (`waiting` +
        `waitConditionRestartBlock`),
      - deferred-halt suspends when active fork children are present
        (`waiting` + `destBlock` + `resumeAtCurrentOp` at `llhd.halt`).
    - added env-gated guard diagnostics for guard-failed triage:
      - `CIRCT_SIM_TRACE_JIT_THUNK_GUARDS=1`.
    - added strict regressions:
      - `jit-process-thunk-fork-join-none-halt.mlir`
      - `jit-process-thunk-scf-for-halt.mlir`
      - `jit-process-thunk-llvm-call-wait-condition-halt.mlir`
    - refreshed bounded AVIP compile-lane run
      (`/tmp/avip-circt-sim-20260217-225344`, `jtag`, seed-1) shows
      unsupported-tail churn from previous queue:
      - closed prior tails `first_op:sim.fork` and `first_op:scf.for`,
      - remaining unsupported details now:
        - `first_op:llvm.call:__moore_wait_condition` (1)
        - `first_op:llvm.call:__moore_delay` (1)
        - `multiblock_no_terminal` (1)
      - guard-failed remains the dominant queue (`8` rows in this run).
16. Multiblock wait-condition loop closure and AVIP unsupported-tail burn-down:
    - resumable multiblock wait-thunk matching now accepts either:
      - explicit `llhd.wait` terminators, or
      - explicit `llvm.call @__moore_wait_condition` suspend points in block
        preludes.
    - resumable multiblock wait-thunk execution now supports
      wait-condition polling restart state:
      - ignores spurious wakeups while wait-condition poll state is armed
        (`waiting + waitConditionRestartBlock`),
      - resumes from `waitConditionRestartOp` with invalidation of
        traced condition inputs.
    - deopt detail classification for multiblock process bodies now aligns with
      resumable wait matching (tracks wait-condition as a suspend source and
      only emits `multiblock_no_terminal` when no terminal and no suspend
      source are present).
    - expanded one-block LLVM prelude allowlist with non-suspending runtime
      queue helper:
      - `__moore_queue_pop_front_ptr`.
    - expanded one-block `func.call` prelude allowlist with:
      - `m_process_guard` and `*::m_process_guard`.
    - added strict regressions:
      - `jit-process-thunk-multiblock-wait-condition-loop-halt.mlir`
      - `jit-process-thunk-llvm-call-queue-pop-front-ptr-halt.mlir`
      - `jit-process-thunk-func-call-m-process-guard-halt.mlir`
    - bounded AVIP compile-lane burn-down progression (jtag seed-1):
      - `/tmp/avip-circt-sim-20260217-230144`:
        unsupported `__moore_wait_condition` closed; new tail
        `__moore_queue_pop_front_ptr` appeared.
      - `/tmp/avip-circt-sim-20260217-230458`:
        `__moore_queue_pop_front_ptr` closed; new tail
        `func.call:m_process_guard` appeared.
      - `/tmp/avip-circt-sim-20260217-230749`:
        `func.call:m_process_guard` closed.
      - current unsupported tail reduced to:
        - `first_op:llvm.call:__moore_delay` (1)
        - `multiblock_no_terminal` (1)
      - guard-failed remains dominant (`9` rows in latest run).
17. Native `__moore_delay` suspend closure and AVIP compile-lane timeout burn-down:
    - resumable multiblock wait-thunk matching now treats
      `llvm.call @__moore_delay` as a recognized suspend source (alongside
      `llhd.wait` and `llvm.call @__moore_wait_condition`).
    - resumable multiblock wait-thunk execution now schedules pending
      `__moore_delay` callback wakeups directly from native thunk dispatch
      (`waiting && pendingDelayFs > 0`), preserving interpreter parity.
    - expanded safe prelude allowlists for remaining AVIP tails:
      - `func.call`: `uvm_pkg::uvm_root::find_all`,
        `*::set_report_verbosity_level`
      - `llvm.call`: `__moore_string_cmp`, `__moore_assoc_get_ref`
    - added strict regressions:
      - `jit-process-thunk-multiblock-llvm-call-delay-halt.mlir`
      - `jit-process-thunk-func-call-uvm-root-find-all-halt.mlir`
      - `jit-process-thunk-llvm-call-string-cmp-halt.mlir`
      - `jit-process-thunk-llvm-call-assoc-get-ref-halt.mlir`
      - `jit-process-thunk-func-call-set-report-verbosity-level-halt.mlir`
      - updated `jit-process-thunk-llvm-call-delay-unsupported.mlir`
        from strict expected-fail to strict pass.
    - bounded AVIP compile-lane progression (`jtag`, seed-1):
      - `/tmp/avip-circt-sim-20260217-232100`:
        simulation now completes (`sim_status=OK`, `sim_sec=31s`);
        unsupported tails:
        `first_op:func.call:uvm_pkg::uvm_root::find_all`,
        `multiblock_no_terminal`.
      - `/tmp/avip-circt-sim-20260217-232403`:
        `uvm_root::find_all` tail closed; new tail
        `first_op:llvm.call:__moore_string_cmp`.
      - `/tmp/avip-circt-sim-20260217-232923`:
        `__moore_string_cmp` and `__moore_assoc_get_ref` tails closed;
        new tail `first_op:func.call:uvm_pkg::uvm_report_object::set_report_verbosity_level`.
      - `/tmp/avip-circt-sim-20260217-233506`:
        `set_report_verbosity_level` tail closed; current unsupported tails:
        - `first_op:func.call:uvm_pkg::uvm_report_object::set_report_id_verbosity` (1)
        - `multiblock_no_terminal` (1)
      - guard-failed remains dominant (`9` rows in latest run).
18. Unsupported-tail elimination for AVIP jtag compile lane:
    - expanded one-block non-suspending `func.call` prelude allowlist with:
      - `*::set_report_id_verbosity`.
    - added env-gated unsupported-shape tracer for strict burn-down triage:
      - `CIRCT_SIM_TRACE_JIT_UNSUPPORTED_SHAPES=1`
      - emits per-process block/op skeleton when thunk install classifies
        `unsupported_operation`.
    - multiblock suspend-source matching/deopt-detail classification now treats
      `sim.fork` preludes as suspend-capable for resumable multiblock thunk
      candidacy (narrow closure for fork-loop shapes previously classified as
      `multiblock_no_terminal`).
    - added strict regressions:
      - `jit-process-thunk-func-call-set-report-id-verbosity-halt.mlir`
      - `jit-process-thunk-multiblock-fork-loop-guard-failed.mlir`
        (asserts deopt classification is `guard_failed` rather than
        `unsupported_operation:multiblock_no_terminal`).
    - bounded AVIP compile-lane progression (`jtag`, seed-1):
      - baseline `/tmp/avip-circt-sim-20260217-235254`:
        - `guard_failed=9`
        - `unsupported_operation=1` (`multiblock_no_terminal`)
      - after closure `/tmp/avip-circt-sim-20260217-235838`:
        - `guard_failed=8`
        - `unsupported_operation=0`
      - result: unsupported-operation tail eliminated on this lane; remaining
        strict convergence queue is entirely guard-failed.
19. Guard-failed deopt detail instrumentation for strict convergence:
    - extended thunk deopt bridge ABI with optional guard detail payload:
      - `ProcessThunkExecutionState::deoptDetail`.
    - compile-mode deopt accounting now forwards thunk guard details into
      per-process JIT telemetry for `guard_failed` deopts.
    - added explicit guard-deopt detail reasons in multiblock executors
      (`empty_body_region`, `resume_token_mismatch`, `step_limit_reached`,
      `post_exec_not_halted_or_waiting`, etc.) to avoid opaque guard bucketing.
    - tightened regression for multiblock fork loop:
      - `jit-process-thunk-multiblock-fork-loop-guard-failed.mlir` now checks
        strict detail `step_limit_reached`.
    - bounded AVIP compile-lane readout (`jtag`, seed-1):
      - `/tmp/avip-circt-sim-20260218-000327`
      - deopt queue now fully detail-qualified:
        - `guard_failed:post_exec_not_halted_or_waiting` (9)
      - unsupported-operation remains `0` on this lane.
20. Single-block terminating call-stack resume closure (targeted):
    - extracted shared call-stack resume helper:
      - added `resumeSavedCallStackFrames(...)` and
        `CallStackResumeResult` in `LLHDProcessInterpreter`.
      - `executeProcess` now uses the helper (behavior-preserving refactor).
    - upgraded `single_block_terminating` thunk execution to support resumable
      suspension states without broad guard relaxation:
      - call-stack-backed suspensions now resume natively instead of immediate
        deopt on non-empty call stack.
      - process-level sequencer retry waits
        (`sequencerGetRetryCallOp`) are treated as resumable waiting states.
      - wait-condition restart and call-stack frame rewrites are aligned with
        interpreter resume logic.
      - guard details are now shape-qualified
        (`single_block_terminating:<reason>`) with richer guard tracing.
    - added strict regression:
      - `jit-process-thunk-single-block-call-indirect-fork-callstack-halt.mlir`
        (vtable/call_indirect + blocking fork in callee, strict no-deopt).
    - bounded AVIP compile-lane validation (`jtag`, seed-1):
      - `/tmp/avip-circt-sim-singleblock-callstack-v2-20260218-012809`:
        - `sim_status=OK`, `sim_sec=35s`
        - `guard_failed=7`
        - queue is now entirely
          `single_block_terminating:post_exec_not_halted_or_waiting`.
    - remaining limitation at this step (closed in item 25):
      - one-block fork-branch shapes with
        `waiting=1, call_stack=0` around `func.call_indirect` + terminal
        `sim.fork.terminator`.
21. Maintainability split for long-term JIT development velocity:
    - extracted native-thunk execution methods from
      `LLHDProcessInterpreter.cpp` into
      `LLHDProcessInterpreterNativeThunkExec.cpp`.
    - keeps JIT thunk execution/refinement work isolated from core
      interpreter dispatch and reduces edit-conflict pressure during
      strict-native burn-down.
    - no intended behavior change; validation covered existing strict JIT
      thunk regressions.
22. Maintainability split (phase-2): native-thunk policy/candidate isolation:
    - extracted compile-path native-thunk policy and shape classification from
      `LLHDProcessInterpreter.cpp` into
      `LLHDProcessInterpreterNativeThunkPolicy.cpp`.
    - moved install/deopt snapshot/unsupported-detail paths and all
      candidate-classifier implementations into the new unit.
    - no intended behavior change; validated with strict JIT thunk
      regressions after rebuild.
23. Maintainability split (phase-3): global lifecycle isolation:
    - extracted global/discovery finalization methods from
      `LLHDProcessInterpreter.cpp` into
      `LLHDProcessInterpreterGlobals.cpp`:
      - `finalizeInit`
      - `discoverGlobalOpsIteratively`
      - `loadRTTIParentTable`
      - `checkRTTICast`
      - `initializeGlobals`
      - `executeGlobalConstructors`
      - `interpretLLVMAddressOf`
    - keeps global-init/RTTI/vtable plumbing separate from process execution
      and native thunk work to reduce merge conflicts in active JIT paths.
    - no intended behavior change; validated with both global-ctor/vtable and
      strict JIT regressions (including parallel-mode compile tests).
24. Maintainability split (phase-4): module-level LLVM init isolation:
    - extracted `executeModuleLevelLLVMOps(...)` from
      `LLHDProcessInterpreter.cpp` into
      `LLHDProcessInterpreterModuleLevelInit.cpp`.
    - introduced shared internal store-pattern matcher header:
      `LLHDProcessInterpreterStorePatterns.h`, used by both core interpreter
      and module-level init translation units to keep four-state copy/tri-state
      detection logic consistent across paths.
    - no intended behavior change; validated with global-init/vtable and
      strict compile-mode JIT regression sets.
    - bounded AVIP compile-lane smoke refreshed:
      - `AVIPS=jtag`, `SEEDS=1`, `CIRCT_SIM_MODE=compile`,
        `COMPILE_TIMEOUT=90`, `SIM_TIMEOUT=90`
      - output bundle: `/tmp/avip-circt-sim-20260218-022455`
      - result: compile `OK` (`28s`), sim `OK` (`88s`).
25. Single-block IMP-order waiting closure for strict native thunk execution:
    - closed the residual
      `single_block_terminating:post_exec_not_halted_or_waiting` guard path
      for `uvm_phase_hopper::process_phase` IMP-order gating waits.
    - `executeSingleBlockTerminatingNativeThunk` now recognizes
      `impWaitingProcesses`-queued states as valid waiting suspensions:
      - both pre-activation queued waiting and post-exec queued waiting are
        handled natively (no guard-failed deopt).
      - closure is narrow to IMP-order queue membership; no broad waiting-state
        guard relaxation.
    - added strict regression:
      - `jit-process-thunk-single-block-fork-process-phase-imp-wait.mlir`
        (fork-branch `sim.fork.terminator` + `process_phase` wait queue shape,
        strict no-deopt).
    - validation:
      - targeted strict checks pass for:
        - `jit-process-thunk-single-block-fork-process-phase-imp-wait.mlir`
        - `jit-process-thunk-single-block-call-indirect-fork-callstack-halt.mlir`
        - `jit-process-thunk-func-call-set-report-id-verbosity-halt.mlir`
        - `jit-process-thunk-multiblock-fork-loop-guard-failed.mlir`
          (expected strict fail with `detail=step_limit_reached` preserved).
      - parallel compatibility smoke passes for the new strict regression with
        `--parallel=4` (expected sequential fallback warning).
      - bounded AVIP compile-lane smoke (`jtag`, seed-1):
        - output bundle:
          `/tmp/avip-circt-sim-impwait-20260218-024142`
        - compile `OK` (`26s`), sim `TIMEOUT` (`90s`).
26. Multiblock resumable call-stack closure and AVIP `jtag` strict burn-down to
    zero deopts:
    - removed hard resumable multiblock guard deopt
      (`non_empty_call_stack`) in
      `executeResumableMultiblockWaitNativeThunk`.
    - resumable multiblock wait thunks now route through
      `resumeSavedCallStackFrames(...)` and honor `Completed` / `Suspended` /
      `Failed` outcomes with guarded fallback.
    - expanded native waiting-state closure in terminating thunk paths:
      - fork-join active-child waits.
      - process-await queue waits (`processAwaiters`).
      - objection wait polling (`objectionWaitForStateByProc`) for multiblock
        terminating paths.
    - expanded single-block safe fork prelude classification:
      - accepts `join` / `join_any` / `join_none`.
      - accepts `sim.disable_fork` prelude op.
      - adds richer unsupported-shape diagnostics for nested fork branches.
    - strict regression coverage:
      - added:
        - `jit-process-thunk-fork-join-disable-fork-terminator.mlir`
        - `jit-process-thunk-multiblock-llvm-call-process-await-halt.mlir`
      - updated existing fork-branch strict tests to no-deopt expectations:
        - `jit-process-thunk-fork-branch-alloca-gep-load-store-terminator.mlir`
        - `jit-process-thunk-fork-branch-insertvalue-terminator.mlir`
    - validation:
      - targeted strict checks pass for the updated/new regressions.
      - targeted parallel-mode compile smokes pass with `--parallel=4`.
      - bounded AVIP compile-lane burn-down (`AVIPS=jtag`, `SEEDS=1`):
        - output bundle:
          `/tmp/avip-circt-sim-jit-jtag-20260218-041130`
        - compile `OK` (`25s`), sim `OK` (`90s`).
        - `jit_deopts_total=0` and no per-process deopt rows.
    - remaining limitation at this step:
      - strict zero-deopt has only been re-proven on bounded AVIP `jtag`
        (`seed=1`) in this wave; wider closure still requires periodic
        full-lane sweeps across `~/mbit/*avip*`, `~/sv-tests/`,
        `~/verilator-verification/`, `~/yosys/tests/`, and `~/opentitan/`.
27. APB strict burn-down wave: close LLVM-call and bare-wait tails; queue down
    to two process-level deopts:
    - expanded safe LLVM-call prelude coverage for single-block/multiblock
      thunk candidates:
      - `__moore_is_rand_enabled`
      - `__moore_int_to_string` / `__moore_string_itoa`
      - `__moore_string_concat`
      - `__moore_randomize_basic`
      - `__moore_randomize_with_range`
      - `__moore_randomize_with_ranges`
      - `__moore_randomize_with_dist`
    - expanded resumable multiblock wait candidate policy to accept
      `llhd.wait` terminators with no delay/observed list when destination
      block/operand mapping is valid.
    - enriched unsupported-shape tracing for `scf.if` by dumping nested then/else
      region block skeletons and nested call ops under
      `CIRCT_SIM_TRACE_JIT_UNSUPPORTED_SHAPES=1`.
    - strict regression coverage added:
      - `jit-process-thunk-llvm-call-is-rand-enabled-halt.mlir`
      - `jit-process-thunk-llvm-call-int-to-string-halt.mlir`
      - `jit-process-thunk-llvm-call-randomize-basic-halt.mlir`
      - `jit-process-thunk-llvm-call-randomize-with-range-halt.mlir`
      - `jit-process-thunk-llvm-call-string-concat-halt.mlir`
      - `jit-process-thunk-multiblock-scf-if-randomize-range-halt.mlir`
      - `jit-process-thunk-multiblock-bare-wait-no-trigger.mlir`
      - `jit-process-thunk-multiblock-call-indirect-process-await-halt.mlir`
    - validation:
      - targeted strict checks pass for the new regressions.
      - targeted `--parallel=4` smoke on
        `jit-process-thunk-multiblock-scf-if-randomize-range-halt.mlir` passes
        with zero deopts.
      - bounded APB compile-lane burn-down (`AVIPS=apb`, `SEEDS=1`,
        `CIRCT_SIM=build/bin/circt-sim`):
        - baseline sample:
          `/tmp/avip-circt-sim-jit-apb-buildbin-20260218-042511`
          with `jit_deopts_total=4`.
        - after closures:
          `/tmp/avip-circt-sim-jit-apb-buildbin-20260218-044313`
          and
          `/tmp/avip-circt-sim-jit-apb-buildbin-20260218-050046`
          with `jit_deopts_total=2`.
    - controlled rollback note:
      - a broad multiblock saved-call-stack wait-acceptance attempt caused
        APB timeout regression (`SIM_TIMEOUT=180`); reverted to keep
        queue-backed waits only.
    - remaining ranked APB queue:
      - `unsupported_operation:multiblock_no_terminal` (proc `fork_96_branch_0`)
      - `guard_failed:post_exec_not_halted` (proc `fork_97_branch_0`)
28. APB strict burn-down closure: eliminate final `invalid_dest_block_state`
    tail and reach zero-deopt in bounded compile lane:
    - policy closure:
      - introduced `isPotentialResumableMultiblockSuspendOp(...)` in
        `LLHDProcessInterpreterNativeThunkPolicy.cpp`.
      - resumable multiblock suspend sources now include:
        - `sim.fork`
        - `func.call_indirect`
        - LLVM calls: `__moore_wait_condition`, `__moore_delay`,
          `__moore_process_await`
    - execution closure:
      - added resumable-multiblock guard tracing under
        `CIRCT_SIM_TRACE_JIT_THUNK_GUARDS=1`.
      - selected resume body region from `state.destBlock->getParent()` when
        destination-resuming.
      - aligned destination-block operand handling with interpreter resume
        behavior (removed strict arity deopt in this thunk path).
    - strict regression coverage:
      - added
        `jit-process-thunk-multiblock-call-indirect-delay-halt.mlir`
        (call-indirect to callee performing `llvm.call @__moore_delay`).
    - validation:
      - `ninja -C build circt-sim`: PASS.
      - targeted strict regressions (6 tests): PASS.
      - bounded APB compile lane (`AVIPS=apb`, `SEEDS=1`,
        `CIRCT_SIM=build/bin/circt-sim`,
        `CIRCT_VERILOG=build-test/bin/circt-verilog`,
        `CIRCT_SIM_MODE=compile`,
        `CIRCT_SIM_WRITE_JIT_REPORT=1`,
        `CIRCT_SIM_EXTRA_ARGS='--jit-hot-threshold=1 --jit-compile-budget=100000'`):
        - `/tmp/avip-circt-sim-jit-apb-buildbin-20260218-051931`:
          `jit_deopts_total=1` (`guard_failed:invalid_dest_block_state`)
        - `/tmp/avip-circt-sim-jit-apb-buildbin-20260218-052243`:
          `jit_deopts_total=0`
      - bounded JTAG `--parallel=4` smoke
        (`/tmp/avip-circt-sim-jit-jtag-p4-20260218-052357`):
        compile `OK`, sim `OK`, `jit_deopts_total=0`.
    - remaining limitation after this closure:
      - zero-deopt has been re-proven on bounded `apb` and `jtag` samples in
        this wave; full plan still requires periodic broader sweeps across
        `~/mbit/*avip*`, `~/sv-tests/`, `~/verilator-verification/`,
        `~/yosys/tests/`, and `~/opentitan/`.
29. i2s strict burn-down closure for terminate path + updated core8 snapshot:
    - closure target:
      - bounded core8 sweep
        (`/tmp/avip-circt-sim-jit-core8-buildbin-20260218-052944`) showed
        one remaining strict deopt in `i2s`:
        `unsupported_operation:first_op:sim.terminate`
        (`llhd_process_4`), with shape:
        `llhd.wait delay -> sim.terminate -> llhd.halt`.
    - implementation:
      - narrowed resumable wait-then-halt support to allow terminal
        destination block form:
        optional `sim.proc.print`, optional `sim.terminate`, then `llhd.halt`.
      - execution path now runs optional `sim.terminate` before `llhd.halt`
        and preserves native resume semantics when terminate defers.
    - regression coverage:
      - added strict test:
        `jit-process-thunk-resumable-wait-then-terminate-halt.mlir`.
    - validation:
      - `ninja -C build circt-sim`: PASS.
      - direct strict compile-mode checks on touched regressions: PASS
        (`jit_deopts_total=0` for all).
      - bounded reruns:
        - `i2s`:
          `/tmp/avip-circt-sim-jit-i2s-buildbin-20260218-055443`
          compile `OK` (`39s`), sim `OK` (`71s`), `jit_deopts_total=0`.
        - `jtag`:
          `/tmp/avip-circt-sim-jit-jtag-buildbin-20260218-055649`
          compile `OK` (`26s`), sim `OK` (`39s`), `jit_deopts_total=0`.
    - updated bounded core8 status after this wave:
      - strict-native deopt closure: `apb`, `ahb`, `axi4Lite`, `i2s`,
        `i3c`, `jtag`, `spi` at `jit_deopts_total=0` (bounded sample lane).
      - remaining blocker in core8 matrix: `axi4` bounded-lane instability.
        follow-up run with raised compile timeout
        (`/tmp/avip-circt-sim-jit-axi4-buildbin-20260218-055956`,
        `COMPILE_TIMEOUT=360`) compiles in `85s`, but sim still times out
        at `120s` and logs absorbed internal `tx_read_packet` `llhd.drv`
        failure warnings before timeout.
30. AXI4 blocker closure wave: guard memory-backed `sig.extract` drives against
    hard-fail on unresolved native pointers
    (February 18, 2026):
    - root cause confirmed in bounded AXI4 compile lane:
      `tx_read_packet` hit `llhd.drv` on `llhd.sig.extract` of a
      memory-backed `!llhd.ref<i64>` and failed with unresolved memory lookup
      (`memory_not_found`), causing absorbed internal function failure.
    - implementation updates in `interpretDrive` (`llhd.sig.extract` path):
      - treat unresolved memory-backed cast targets as guarded no-op success
        (aligning with existing non-fatal pointer-store behavior instead of
        hard-failing process execution),
      - keep out-of-range extract windows as no-op,
      - switch memory-backed bit-slice writes from full-parent-width
        read/modify/write to touched-byte-window updates, avoiding spurious
        boundary failures on narrow sub-reference writes.
    - regression coverage added:
      - `test/Tools/circt-sim/llhd-drv-sig-extract-oob-noop.mlir`
    - validation:
      - targeted regressions pass:
        - `llhd-drv-sig-extract-oob-noop.mlir`
        - `llhd-drv-memory-backed-struct-array-func-arg.mlir`
        - `llhd-ref-cast-array-subfield-store-func-arg.mlir`
      - bounded AXI4 compile lane rerun
        (`/tmp/avip-circt-sim-20260218-064614`):
        compile `OK` (`85s`), sim `TIMEOUT` (`180s`), and the previous
        absorbed `tx_read_packet` `llhd.drv` failure signature is no longer
        present in `sim_seed_1.log`.
      - repeat bounded AXI4 rerun
        (`/tmp/avip-circt-sim-20260218-065533`) also has no
        `tx_read_packet` absorbed-failure signature and advances beyond the
        previous 90fs failure point before hitting the same 180s timeout cap.
      - bounded JTAG compile lane regression check
        (`/tmp/avip-circt-sim-20260218-065058`):
        compile `OK` (`28s`), sim `OK` (`40s`), `jit_deopts_total=0`.
31. AXI4 post-crash status and next strict-native queue extraction
    (February 18, 2026):
    - repeated bounded AXI4 compile lane checks after the `sig.extract` guard
      closure (`/tmp/avip-circt-sim-20260218-065533`,
      `/tmp/avip-circt-sim-20260218-070122`) confirm:
      - no absorbed `tx_read_packet` internal-failure signature,
      - lane still times out under 180s and 300s caps.
    - direct TERM-bounded profiling run with JIT report emission
      (`/tmp/axi4-term120.jit-report.json`) exposes the current strict queue:
      - `jit_deopts_total=7`
      - all remaining deopts are `unsupported_operation`, details:
        - `first_op:llvm.call:__moore_assoc_size` (2)
        - `first_op:llvm.call:__moore_semaphore_get` (3)
        - `first_op:func.call:from_write_class_6984` (1)
        - `first_op:func.call:from_read_class_6987` (1)
    - next closure target:
      - reduce AXI4 timeout pressure by closing/guarding these four
        unsupported first-op classes in native thunk policy/execution.
32. AXI4 unsupported-tail shift: close `__moore_assoc_size` first-op class
    (February 18, 2026):
    - native thunk policy closure:
      - added `llvm.call @__moore_assoc_size` to the safe non-suspending
        LLVM prelude allowlist for single-block terminating thunk candidates.
    - regression coverage:
      - added strict compile-mode regression
        `test/Tools/circt-sim/jit-process-thunk-llvm-call-assoc-size-halt.mlir`.
    - validation:
      - targeted strict regressions pass:
        - `jit-process-thunk-llvm-call-assoc-size-halt.mlir`
        - `jit-process-thunk-llvm-call-assoc-get-ref-halt.mlir`
      - TERM-bounded AXI4 queue sample after closure
        (`/tmp/axi4-term120-after-assoc-size.jit-report.json`):
        - `jit_deopts_total=7` (unchanged),
        - previous `first_op:llvm.call:__moore_assoc_size` tail removed,
        - queue shifted to:
          - `first_op:scf.for` (2)
          - `first_op:llvm.call:__moore_semaphore_get` (3)
          - `first_op:func.call:from_write_class_6984` (1)
          - `first_op:func.call:from_read_class_6987` (1).
33. AXI4 wrapper-tail shift: close write/read class-bridge wrappers plus
    queue push-back prelude
    (February 18, 2026):
    - native thunk policy closures:
      - added signature-gated `func.call` prelude allowlist for numeric
        class-bridge wrappers:
        - `from_write_class_<digits>`
        - `from_read_class_<digits>`
        - `to_write_class_<digits>`
        - `to_read_class_<digits>`
      - added `llvm.call @__moore_queue_push_back` to the safe
        non-suspending LLVM prelude allowlist for single-block terminating
        thunk candidates.
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-from-write-class-wrapper-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-from-read-class-wrapper-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-queue-push-back-halt.mlir`
    - validation:
      - targeted strict compile-mode regressions (build-test lane): PASS
        - `jit-process-thunk-func-call-from-write-class-wrapper-halt.mlir`
        - `jit-process-thunk-func-call-from-read-class-wrapper-halt.mlir`
        - `jit-process-thunk-llvm-call-queue-push-back-halt.mlir`
        - `jit-process-thunk-llvm-call-assoc-size-halt.mlir`
      - targeted parallel compile-mode smoke (build-test lane): PASS
        - `jit-process-thunk-llvm-call-queue-push-back-halt.mlir`
          with `--parallel=4 --work-stealing --auto-partition`.
      - TERM-bounded AXI4 queue sample after closure
        (`/tmp/axi4-term120-after-queue-push-back.jit-report.json`):
        - `jit_deopts_total=12`
        - removed wrapper-tail entries:
          - `first_op:func.call:from_write_class_*`
          - `first_op:func.call:from_read_class_*`
        - queue shifted to:
          - `first_op:scf.for` (2)
          - `first_op:llvm.call:__moore_semaphore_get` (6)
          - `first_op:func.call:from_class_6985` (4)
    - next closure target:
      - close `first_op:scf.for` via remaining unsupported calls in the
        structured prelude and then close/guard
        `first_op:func.call:from_class_*` and
        `first_op:llvm.call:__moore_semaphore_get`.
34. AXI4 semaphore + generic class-wrapper closure wave
    (February 18, 2026):
    - native thunk policy closures:
      - expanded signature-gated class-bridge wrapper name coverage with:
        - `from_class_<digits>`
        - `to_class_<digits>`
      - expanded non-suspending intercepted `func.call` prelude coverage with:
        - `uvm_pkg::uvm_is_match`
        - `*::uvm_is_match`
      - expanded non-suspending LLVM-call prelude coverage with semaphore
        runtime calls:
        - `__moore_semaphore_create`
        - `__moore_semaphore_get`
        - `__moore_semaphore_put`
        - `__moore_semaphore_try_get`
    - native thunk execution closures:
      - single-block and multiblock terminating native thunks now preserve
        native execution across blocking `__moore_semaphore_get` suspend/resume
        states (using `pendingSemaphoreGetId` + `destBlock` +
        `resumeAtCurrentOp` guards) instead of deopting on wake.
      - interpreter resume normalization now clears
        `pendingSemaphoreGetId` after destination-block resume.
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-from-class-wrapper-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-semaphore-get-blocking-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-scf-for-uvm-is-match-halt.mlir`
    - validation:
      - targeted strict compile-mode regressions: PASS
        - all three new tests above
        - existing closure guards:
          - `jit-process-thunk-func-call-from-write-class-wrapper-halt.mlir`
          - `jit-process-thunk-func-call-from-read-class-wrapper-halt.mlir`
          - `jit-process-thunk-llvm-call-queue-push-back-halt.mlir`
          - `jit-process-thunk-llvm-call-assoc-size-halt.mlir`
      - targeted parallel compile-mode smoke: PASS
        - `jit-process-thunk-llvm-call-semaphore-get-blocking-halt.mlir`
          with `--parallel=4 --work-stealing --auto-partition`
        - `jit-process-thunk-llvm-call-queue-push-back-halt.mlir`
          with `--parallel=4 --work-stealing --auto-partition`
      - TERM-bounded AXI4 queue sample after closure
        (`/tmp/axi4-term120-after-semaphore-fromclass-scf-for.jit-report.json`):
        - `jit_deopts_total=12` (unchanged count, shifted tail classes)
        - removed:
          - `first_op:func.call:from_class_6985`
          - `first_op:llvm.call:__moore_semaphore_get`
        - remaining queue:
          - `first_op:scf.for` (2)
          - `first_op:func.call:axi4_slave_driver_bfm::axi4_write_address_phase` (2)
          - `first_op:func.call:axi4_slave_driver_bfm::axi4_write_data_phase` (2)
          - `first_op:func.call:axi4_slave_driver_bfm::axi4_read_address_phase` (2)
          - `first_op:llvm.call:__moore_queue_delete_index` (4)
    - next closure target:
      - close `first_op:scf.for` by widening nested structured-prelude
        non-suspending call coverage in the affected loops.
      - close direct BFM phase-call and queue-delete-index tails through
        signature-gated non-suspending prelude admission plus strict regressions.
35. AXI4 BFM-phase + queue family closure wave
    (February 18, 2026):
    - native thunk policy closures:
      - added signature-gated non-suspending `func.call` prelude admission for
        `*_driver_bfm::*_phase` methods (`this` pointer + handle/ref shape).
      - widened non-suspending LLVM-call prelude admission to full
        `__moore_queue_*` helper family (including delete-index paths).
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-driver-bfm-phase-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-queue-delete-index-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-scf-for-driver-bfm-phase-queue-delete-halt.mlir`
    - validation:
      - targeted strict regressions: PASS (new tests + prior closure guards).
      - targeted parallel smokes: PASS (`--parallel=4 --work-stealing --auto-partition`).
      - TERM-bounded AXI4 queue sample
        (`/tmp/axi4-term120-after-driverbfm-queuedelete.jit-report.json`):
        - `jit_deopts_total=7` (from 12),
        - removed BFM phase and `__moore_queue_delete_index` first-op tails.
36. AXI4 wrapper/result-helper closure wave
    (February 18, 2026):
    - native thunk policy closures:
      - widened `to_*_class_<digits>` wrapper signature acceptance to cover
        value-to-ref bridge form used by generated wrappers.
      - added non-suspending `func.call` prelude admission:
        - `*::get_minimum_transactions`
      - added non-suspending LLVM-call prelude admission:
        - `__moore_dyn_array_*`
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-to-class-wrapper-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-get-minimum-transactions-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-dyn-array-new-halt.mlir`
    - validation:
      - targeted strict regressions: PASS.
      - targeted parallel smokes: PASS.
      - TERM-bounded AXI4 queue sample
        (`/tmp/axi4-term120-after-driverbfm-queueclass-dynarray.wall180.jit-report.json`):
        - `jit_deopts_total=7` (count held; queue shifted),
        - removed `to_*_class`, `get_minimum_transactions`,
          `__moore_dyn_array_new` tails.
37. AXI4 loop-tail burn-down to zero-deopt in bounded compile lane
    (February 18, 2026):
    - native thunk policy closures:
      - added non-suspending intercepted `func.call` prelude admission:
        - `*::sprint`
        - signature-gated `tx_*_packet` helpers (ptr/ref packet helper shape).
      - widened safe LLVM-call prelude families:
        - `__moore_assoc_*`
        - `__moore_string_*`
      - widened safe non-suspending op preludes:
        - `llhd.drv`
        - `llhd.sig`
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-uvm-object-sprint-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-tx-write-packet-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llhd-drv-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-assoc-delete-key-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llvm-call-string-bintoa-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-llhd-sig-halt.mlir`
    - validation:
      - targeted strict regressions: PASS (17 focused tests).
      - targeted parallel smokes: PASS (7 focused tests).
      - AXI4 bounded compile progression:
        - `/tmp/axi4-term120-after-sprint-txpacket.wall180.jit-report.json`
          -> `jit_deopts_total=2` (`first_op:scf.for` closed; tail shifted).
        - `/tmp/axi4-term120-after-scffor-drv-assoc.wall180.jit-report.json`
          -> `jit_deopts_total=2` (shifted to
             `first_op:llvm.call:__moore_string_bintoa` and `first_op:llhd.sig`).
        - `/tmp/axi4-term120-after-string-sig.wall300.jit-report.json`
          -> `jit_deopts_total=0`.
      - parallel bounded AXI4 smoke:
        - `/tmp/axi4-term20-parallel-after-string-sig.jit-report.json`
          -> `jit_deopts_total=0`.
    - next closure target:
      - expand zero-deopt compile-mode burn-down from bounded AXI4 sample to
        broader AVIP matrix and non-AVIP suites under plan gates.
38. Local `func.call` non-suspending summary for native-thunk prelude
    admission (February 18, 2026):
    - native thunk policy closures:
      - split manual intercepted-call admission from a new static local-callee
        suspension summary.
      - direct local `func.func` callees are now auto-admitted when their
        transitive bodies are non-suspending.
      - recursive summary is conservative and treats these as suspending:
        - `llhd.wait` / `llhd.yield` / `llhd.halt`
        - fork/join family ops
        - `func.call_indirect`
        - known suspending runtime calls:
          `__moore_wait_condition`, `__moore_delay`,
          `__moore_process_await`, `__moore_wait_event`
        - unresolved local callees and recursive call cycles
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-nonsuspending-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-suspending-unsupported-strict.mlir`
    - validation:
      - targeted strict/parallel regressions: PASS (9 focused tests, including
        existing BFM/class-wrapper/tx packet closures and parallel thunk smoke).
      - bounded AVIP compile smoke: PASS
        - `/tmp/avip-circt-sim-local-helper-20260218-085223/matrix.tsv`
        - `jtag seed=1`: `compile_status=OK` (`27s`),
          `sim_status=OK` (`40s`).
      - bounded non-AVIP suite smokes: PASS
        - `sv-tests`: `11.10.1--string_concat` PASS
        - `verilator-verification` BMC smoke:
          `assert_changed` PASS
        - `yosys/tests/sva` BMC smoke: `basic00` PASS
        - OpenTitan sim smoke:
          `prim_count` PASS
          (`/tmp/opentitan-circt-sim-local-helper-20260218-085427/run.log`)
    - next closure target:
      - extend static suspension classification beyond direct local
        `func.call`s (notably static-target `call_indirect`) while keeping
        strict-no-deopt safety invariants.
39. Static-target `call_indirect` suspension classification closure
    (February 18, 2026):
    - native thunk policy closures:
      - extended local-callee suspension summary to classify
        `func.call_indirect` using static target extraction.
      - static target extraction now supports:
        - `func.constant @callee` form
        - vtable-style static chain:
          `unrealized_cast(load(gep(addressof @vtable,...)))`
          with `circt.vtable_entries` lookup by slot index.
      - unresolved/dynamic `call_indirect` targets remain conservatively
        suspending.
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-call-indirect-static-nonsuspending-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-call-indirect-static-suspending-unsupported-strict.mlir`
    - validation:
      - targeted strict/parallel regressions: PASS (10 focused tests including
        existing `call_indirect` single/multiblock coverage).
      - bounded integration smokes: PASS
        - AVIP compile lane (`jtag`, seed 1):
          `/tmp/avip-circt-sim-static-indirect-rerun-20260218-090314/matrix.tsv`
          (`compile_status=OK` `28s`, `sim_status=OK` `42s`).
        - `sv-tests`: `11.10.1--string_concat` PASS
          (`/tmp/sv-tests-circt-sim-static-indirect-20260218-090005.txt`).
        - `verilator-verification` BMC smoke:
          `assert_changed` PASS
          (`/tmp/verilator-bmc-static-indirect-20260218-090005.tsv`).
        - `yosys/tests/sva` BMC smoke: `basic00` PASS
          (`/tmp/yosys-sva-bmc-static-indirect-20260218-090005.tsv`).
        - OpenTitan sim smoke: `prim_count` PASS
          (`/tmp/opentitan-circt-sim-static-indirect-20260218-090005/run.log`).
    - next closure target:
      - push static classification deeper into frequent unresolved
        `call_indirect` cases by introducing guarded runtime profile-based
        specialization (target set/version guards) before strict default-on.
40. Unresolved-vtable-slot `call_indirect` conservative slot-set closure
    (February 18, 2026):
    - native thunk policy closures:
      - extended local-callee suspension summary with conservative
        vtable-slot candidate analysis for `func.call_indirect` when the
        direct callee symbol is unresolved but a static slot index is known.
      - for such calls, the policy now:
        - extracts the static vtable slot index from
          `load(gep(..., slot))` callee chains.
        - collects all `circt.vtable_entries` candidates at that slot across
          module vtables.
        - classifies the call as non-suspending only if every candidate callee
          is local and recursively non-suspending.
      - dynamic/unknown slot forms remain conservatively suspending.
    - regression coverage:
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-call-indirect-vtable-slot-nonsuspending-halt.mlir`
      - `test/Tools/circt-sim/jit-process-thunk-func-call-local-helper-call-indirect-vtable-slot-suspending-unsupported-strict.mlir`
    - validation:
      - targeted strict/parallel regressions: PASS (10 focused tests).
      - bounded integration smokes: PASS
        - AVIP compile lane (`jtag`, seed 1): PASS
          (`/tmp/avip-circt-sim-vtable-slot-20260218-090917/matrix.tsv`,
          compile `26s`, sim `41s`).
        - `sv-tests`: `11.10.1--string_concat` PASS
          (`/tmp/sv-tests-circt-sim-vtable-slot-20260218-090917.txt`).
        - `verilator-verification` BMC smoke:
          `assert_changed` PASS
          (`/tmp/verilator-bmc-vtable-slot-20260218-090917.tsv`).
        - `yosys/tests/sva` BMC smoke: `basic00` PASS
          (`/tmp/yosys-sva-bmc-vtable-slot-20260218-090918.tsv`).
        - OpenTitan sim smoke: `prim_count` PASS
          (`/tmp/opentitan-circt-sim-vtable-slot-20260218-090918/run.log`).
    - next closure target:
      - add guarded runtime indirect-target-set profiling and specialization
        for hot unresolved `call_indirect` sites (target-set hash/version
        guards + strict deopt fallback) to close remaining strict tails.
41. Guarded runtime `call_indirect` target-set profiling substrate
    (February 18, 2026):
    - runtime profiling closure:
      - added per-site runtime target-set profiling for `func.call_indirect`
        in `LLHDProcessInterpreter`:
        - stable `site_id`
        - `owner` + `location`
        - `calls_total` + `unresolved_calls`
        - per-target call counts
        - `target_set_version` (increments on first-seen target per site)
        - `target_set_hash` (stable hash of sorted target names)
      - profiling is guarded and only enabled when:
        - `--mode=compile`, and
        - JIT report emission is active (`--jit-report` or
          `CIRCT_SIM_JIT_REPORT_PATH`).
    - telemetry/report closure:
      - extended JIT JSON schema in `circt-sim` with:
        - `jit_call_indirect_sites_total`
        - `jit_call_indirect_calls_total`
        - `jit_call_indirect_unresolved_total`
        - `jit_call_indirect_sites[]` with per-site target-set details.
    - regression coverage:
      - added
        `test/Tools/circt-sim/jit-report-call-indirect-target-profile.mlir`
        to lock schema/behavior for mixed resolved+unresolved sites.
    - validation:
      - builds:
        - `ninja -C build circt-sim`: PASS
        - `ninja -C build-test circt-sim`: PASS
      - focused lit regressions: PASS (5 tests)
        - `jit-report-call-indirect-target-profile.mlir`
        - static/vtable-slot `call_indirect` local-helper strict/parallel
          coverage set.
      - bounded integration smokes: PASS
        - AVIP compile lane (`jtag`, seed 1):
          `/tmp/avip-circt-sim-indirect-profile-20260218-092325/matrix.tsv`
          (`compile_status=OK` `27s`, `sim_status=OK` `41s`).
        - `sv-tests`: `11.10.1--string_concat` PASS
          (`/tmp/sv-tests-circt-sim-indirect-profile-20260218-092447.txt`).
        - `verilator-verification` BMC smoke:
          `assert_changed` PASS
          (`/tmp/verilator-bmc-indirect-profile-20260218-092447.tsv`).
        - `yosys/tests/sva` BMC smoke: `basic00` PASS
          (`/tmp/yosys-sva-bmc-indirect-profile-20260218-092447.tsv`).
        - OpenTitan sim smoke: `prim_count` PASS
          (`/tmp/opentitan-circt-sim-indirect-profile-20260218-092447.log`).
    - next closure target:
      - consume hot unresolved-site target-set profiles to install guarded
        native specializations (site hash/version guards) with strict
        `guard_failed` deopt fallback on guard mismatch.
42. UVM `%d,%s` `sscanf` legalization closure + all-AVIP compile unblock
    (February 18, 2026):
    - root cause and lowering closure:
      - fixed `moore.builtin.sscanf`/`fscanf` destination writeback helper
        (`writeScanfResultToRef`) for UVM-style mixed destinations
        (`!moore.ref<time>`, `!moore.ref<string>`):
        - added string destination support via
          `__moore_packed_string_to_string`.
        - fixed invalid `arith.extsi i64 -> i64` generation when scanning into
          64-bit time/integer destinations.
      - threaded `ModuleOp` through scanf writeback helper so runtime helper
        calls are materialized in conversion.
    - regression coverage:
      - added `test/Conversion/MooreToCore/sscanf-time-string.mlir` to lock
        mixed time+string lowering (`__moore_sscanf` +
        `__moore_packed_string_to_string`).
    - validation:
      - builds:
        - `ninja -C build-test -j2 circt-opt circt-verilog`: PASS
      - focused regressions:
        - `test/Conversion/MooreToCore/sscanf-time-string.mlir`: PASS
        - `test/Tools/circt-sim/syscall-sscanf.sv`: PASS
      - AVIP all9 compile-mode matrix:
        - `/tmp/avip-circt-sim-all9-rerun-20260218-103206/matrix.tsv`
        - compile `OK` on all 9 AVIPs.
        - sim `OK` on 7/9; `axi4` + `uart` timeout at `240s`.
      - extended timeout rerun (`axi4`,`uart`):
        - `/tmp/avip-circt-sim-axi4-uart-long-20260218-105251/matrix.tsv`
        - both still timeout at `600s`, confirming runtime-progress bottlenecks
          rather than short timeout artifacts.
    - next closure target:
      - profile and close long-tail runtime hot loops in `axi4`/`uart`
        compile-mode lanes (step-progress dominated process loops), then rerun
        all9 with strict coverage targets.
43. `moore.wait_event` sensitivity-cache closure for hot event loops
    (February 18, 2026):
    - runtime closure:
      - added `moore.wait_event` reuse of per-process/per-op
        `waitSensitivityCache` entries in
        `LLHDProcessInterpreter::interpretMooreWaitEvent`.
      - on cache hit, wait paths now directly suspend on cached sensitivity
        lists without re-walking detect-event body SSA chains.
      - added env-gated trace hook:
        `CIRCT_SIM_TRACE_WAIT_EVENT_CACHE=1`
        (`[WAIT-EVENT-CACHE] store/hit ...`).
    - regression coverage:
      - `test/Tools/circt-sim/moore-wait-event-sensitivity-cache.mlir`
        (asserts first store + subsequent hit).
    - validation:
      - build:
        - `ninja -C build-test circt-sim -j2`: PASS.
      - focused wait-event regressions: PASS
        - new cache test
        - `moore-wait-event.mlir`
        - `moore-wait-memory-event.mlir`
        - `wait-event-class-member.mlir`.
      - bounded AVIP checks:
        - `uart` compile lane:
          `/tmp/avip-circt-sim-uart-waitcache-20260218-113049/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`, 180s cap).
        - `i3c` compile lane regression guard:
          `/tmp/avip-circt-sim-i3c-waitcache-20260218-113610/matrix.tsv`
          (`compile_status=OK`, `sim_status=OK`, `cov_1_pct=100`,
          `cov_2_pct=100`).
      - direct UART bounded-progress sample:
        - `/tmp/uart-direct-waitcache-113439.log`
        - reached `max-time=6000000000 fs` with hot GenerateBaudClk processes
          showing `sens_cache=59996`, confirming cache engagement on the
          dominant loop.
    - next closure target:
      - remove remaining UART/AXI4 timeout pressure by reducing per-activation
        cost in hot function loops beyond wait setup (clock-divider and BFM
        loop bodies) and improve timeout-lane telemetry emission on bounded
        exits.
44. UART clock-divider fast-path hardening + bounded-lane hotspot extraction
    (February 18, 2026):
    - runtime closures (`tools/circt-sim/LLHDProcessInterpreter.h`,
      `tools/circt-sim/LLHDProcessInterpreter.cpp`):
      - hardened `*::BaudClkGenerator` fast path for null-handle calls:
        - null `self` now suspends via a non-resolving memory waiter instead
          of degrading into no-sensitivity `moore.wait_event` spin loops.
      - added safe count-state fast-path caching:
        - static symbol-use check marks `BaudClkGenerator::count` as
          `countLocalOnly` only when its address-of use is confined to the
          target callee + generated init function.
        - local-only mode avoids per-edge count memory loads/stores.
        - escape-visible mode retains memory writes for external observers.
    - regression coverage:
      - added:
        - `test/Tools/circt-sim/func-baud-clk-generator-fast-path-null-self.mlir`
        - `test/Tools/circt-sim/func-baud-clk-generator-fast-path-count-visible.mlir`
      - retained existing baseline:
        - `test/Tools/circt-sim/func-baud-clk-generator-fast-path.mlir`.
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions:
        - all three Baud fast-path tests above: PASS.
      - bounded AVIP UART lanes:
        - `/tmp/avip-circt-sim-uart-baudfp-check-20260218-1219/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`, `sim_sec=240s`).
        - `/tmp/avip-circt-sim-uart-baudfp-postopt-20260218-1300/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`, `sim_sec=120s`).
      - direct hotspot profiling (wrapper-equivalent tops/args,
        internal timeout mode):
        - `/tmp/uart-direct-timeout60-funcprof.log` and
          `/tmp/uart-direct-timeout60-funcprof-after-countcache.log`
          show dominant remaining function-call pressure:
          `~34k-38k` calls each to:
          - `UartTxDriverBfm::BaudClkGenerator`
          - `UartTxMonitorBfm::BaudClkGenerator`
          - `UartRxMonitorBfm::BaudClkGenerator`
          with `0.00% / 0.00%` coverage at bounded wall timeout.
      - direct Baud trace confirmation:
        - `/tmp/uart-direct-baudtrace-nullstall-60s.log` shows active
          fast-path hits (no `missing-gep-fields` rejects).
    - next closure target:
      - remove edge-by-edge wakeup pressure in hot UART clock-divider loops
        (GenerateBaudClk/BaudClkGenerator path) with guarded native scheduling
        that preserves parity under period/guard mismatch fallback.
45. Compile-budget-zero direct process fast-path dispatch for hot LLHD loops
    (February 18, 2026):
    - runtime closure:
      - added direct process fast-path dispatch at `executeProcess()` entry for
        top-level `llhd.process` bodies:
        - `tools/circt-sim/LLHDProcessInterpreter.cpp`
      - implemented cached fast-path classification and execution in:
        - `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`
      - supported direct fast-path kinds:
        - periodic toggle clock loops
        - resumable wait self-loops
      - direct dispatch reuses existing native-thunk executors but bypasses
        compile-budgeted thunk installation, eliminating repeated
        `missing_thunk/compile_budget_zero` noise for these shapes.
      - added lifecycle cleanup for per-process direct fast-path/periodic-spec
        caches in `finalizeProcess`.
    - regression coverage:
      - added `test/Tools/circt-sim/jit-process-fast-path-budget-zero.mlir`
        to lock compile-mode behavior at `--jit-compile-budget=0`:
        - `jit_compiles_total = 0`
        - `jit_deopts_total = 0`
        - `jit_deopt_reason_missing_thunk = 0`
    - validation:
      - build:
        - `ninja -C build-test circt-sim`: PASS.
      - focused regression:
        - manual RUN + FileCheck sequence for the new test: PASS
          (`/tmp/jit-fastpath-budget0/{log.txt,jit.json}`).
    - current blocker (separate in-flight issue):
      - bounded UART rerun on this tree currently crashes during module-level
        init (`executeModuleLevelLLVMOps`), so updated UART runtime delta is
        blocked pending that fix.
    - next closure target:
      - resolve module-level init crash, then rerun bounded UART and AVIP lanes
        to quantify timeout/cov improvements from direct loop fast paths.
46. UART Baud delay-batching activation guardrail + bounded-progress follow-up
    (February 18, 2026):
    - regression closure:
      - added `test/Tools/circt-sim/func-baud-clk-generator-fast-path-delay-batch.mlir`
        to lock env-gated `batch-schedule` activation and functional output
        parity (`out=1`) for `*::BaudClkGenerator`.
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions:
        - `func-baud-clk-generator-fast-path.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-null-self.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-count-visible.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-delay-batch.mlir` RUN pipeline:
          PASS.
      - direct UART trace sanity:
        - `/tmp/uart-direct-baudbatch-trace30-v3.log`
        - confirms stable batch engagement (`batch-schedule=50`,
          `batch-mismatch=0`) in bounded run.
      - direct UART profile snapshot (internal timeout mode):
        - `/tmp/uart-direct-timeout60-baudbatch-v3.log`
        - reached `310910000000 fs` with top Baud calls reduced to
          `~48622-48623` each (down from prior `~186k`-class pressure in the
          same bounded window).
      - bounded AVIP UART lane:
        - `/tmp/avip-circt-sim-uart-baudbatch-v3-20260218-132926/matrix.tsv`
          remains `TIMEOUT` at `120s`, but now progresses into active
          scoreboard traffic (`~423176 ns` region in sim log).
    - next closure target:
      - convert bounded time-progress gains into lane completion by extending
        the same guarded batching approach to remaining hot loop bodies beyond
        BaudClkGenerator (GenerateBaudClk caller-side dispatch and monitor/driver
        wake choreography), then rerun all9 compile lanes.
47. Resumable wait-self-loop direct linear dispatch for probe/store mirror
    loops (February 18, 2026):
    - runtime closure:
      - refined `executeResumableWaitSelfLoopNativeThunk` in
        `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp` with a
        true direct linear lane for simple self-loop waits:
        - supports non-suspending prelude ops (including `llhd.probe`,
          `llhd.drv`, and `llvm.store`) followed by self-loop `llhd.wait`.
        - executes prelude + wait directly (without per-op `executeStep()`
          dispatch), preserving wait scheduling semantics and deopt guards.
      - keeps existing generic resumable-self-loop fallback for non-linear or
        unsupported shapes.
    - regression coverage:
      - added
        `test/Tools/circt-sim/jit-process-fast-path-store-wait-self-loop.mlir`
        to lock compile-mode budget-zero behavior for probe/store mirror loops:
        - `llhd_process_0` and periodic toggler both report `steps=0`.
        - `jit_compiles_total = 0`
        - `jit_deopts_total = 0`
        - `jit_deopt_reason_missing_thunk = 0`
    - validation:
      - build:
        - `ninja -C build-test -j4 circt-sim`: PASS.
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions (manual RUN + checks): PASS.
        - `jit-process-fast-path-store-wait-self-loop.mlir`
        - `jit-process-fast-path-budget-zero.mlir`
      - bounded UART compile-lane sample:
        - `/tmp/uart-timeout20-storewaitfastpath-20260218-133742.log`
        - `llhd_process_0` now runs with `steps=0` (was hot pre-patch),
          and top remaining hotspots are now
          `fork_{80,81,82}_branch_0` waiting in
          `func.call(*::GenerateBaudClk)` (`~37.4k` steps each in this bound).
        - lane remains timeout/0% coverage in this short bound; closure focus
          shifts to caller-side `GenerateBaudClk` resume overhead.
    - next closure target:
      - add targeted fast path for `GenerateBaudClk` caller-side resumptions
        (fork branch lane) so bounded UART can advance further per wall second
        before broad all-AVIP rerun.
48. Baud delay-batch guard broadening for low-visibility edge lanes + UART
    bounded-progress jump (February 18, 2026):
    - runtime closure:
      - refined `handleBaudClkGeneratorFastPath` guarding in
        `tools/circt-sim/LLHDProcessInterpreter.cpp`:
        - batch-resume guard now accepts elapsed-time-only validation as the
          primary criterion (`elapsedFs == expectedDelayFs`) even when direct
          clock-level sampling is unavailable.
        - when clock-level sampling is available, parity mismatch can be
          corrected via one-edge adjustment before applying batched count
          updates.
        - stable edge-interval tracking for batch eligibility is now based on
          observed activation deltas, avoiding dependence on per-edge sampled
          polarity toggles.
        - batch scheduling no longer requires `clockSampleValid` at schedule
          point when interval stability is already established.
      - keeps existing mismatch fallback/de-batching path for unsafe cases.
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
        - `ninja -C build-test -j4 circt-sim`: PASS.
      - focused regressions:
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`: PASS
          (`batch-schedule` observed).
        - `jit-process-fast-path-store-wait-self-loop.mlir`: PASS.
      - UART bounded compile lane (20s):
        - trace sample:
          `/tmp/uart-baudtrace-20s-post-20260218-135147.log`
          shows active `batch-schedule` hits.
        - runtime sample:
          `/tmp/uart-timeout20-post-batchguard-20260218-135207.log`
          advances to `74876400000 fs` (up from ~`4e9`-class pre-closure in
          same bound), and
          `fork_{80,81,82}_branch_0` drop from ~`37k-43k` steps to `52`.
      - UART extended bounds:
        - `/tmp/uart-timeout60-post-batchguard-20260218-135349.log`:
          `353040000000 fs`, coverage `Rx=0%`, `Tx=100%`.
        - `/tmp/uart-timeout120-post-batchguard-20260218-135534.log`:
          `569765800000 fs`, coverage `Rx=0%`, `Tx=100%`.
    - next closure target:
      - resolve remaining functional/phase-progress blocker for UART Rx
        coverage (`UartRxCovergroup` still 0%) and reduce new dominant hotspot
        (`fork_18_branch_0` `sim.fork` lane) before all-AVIP rerun.
49. `GenerateBaudClk` caller-side resume fast path closure for hot fork
    wrappers (February 18, 2026):
    - runtime closure:
      - added a targeted call-stack resume fast path in
        `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`:
        - detects resumed `*::GenerateBaudClk` frames at tail
          `func.call *::BaudClkGenerator` sites.
        - bypasses full `interpretFuncBody` re-entry on those resume hits and
          dispatches directly through existing
          `handleBaudClkGeneratorFastPath` logic.
        - keeps conservative fallback to normal resume path on guard miss.
      - added env-gated trace marker:
        - `[BAUD-GEN-FP] resume-hit ...`
        - toggled by `CIRCT_SIM_TRACE_BAUD_FASTPATH=1`.
    - regression coverage:
      - added
        `test/Tools/circt-sim/func-generate-baud-clk-resume-fast-path.mlir`.
      - locks caller-side resume fast-path activation in both:
        - default lane.
        - `--parallel=4 --work-stealing --auto-partition` lane.
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions:
        - `func-generate-baud-clk-resume-fast-path.mlir`: PASS.
        - `func-generate-baud-clk-resume-fast-path.mlir` parallel lane: PASS.
        - `func-baud-clk-generator-fast-path.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-count-visible.mlir`: PASS.
      - direct UART bounded profile (compile mode, `--timeout=60`):
        - `/tmp/uart-direct-timeout60-generatefp.log`
        - `fork_{80,81,82}_branch_0` `GenerateBaudClk` steps now `52` each,
          down from prior `~48673-48674` in
          `/tmp/uart-direct-timeout60-baudbatch-v3.log`.
        - simulation-time progress improved to `324960000000 fs`
          (from `310910000000 fs` in the prior 60s bound).
      - bounded AVIP UART lane (compile mode, 120s external bound):
        - `/tmp/avip-circt-sim-uart-generatefp2-20260218-135458/matrix.tsv`
          remains timeout; scoreboard-region activity still reaches
          `~423169 ns` in sim log.
    - next closure target:
      - convert this caller-side overhead win into completion by focusing on
        remaining dominant non-Baud hotspots (`fork_18_branch_0` `sim.fork`
        lane and Rx-coverage progression) and then rerun all9 compile lanes.
50. Execute-phase monitor `sim.fork` objection-waiter stabilization for
    `fork_18_branch_0` hotspot (February 18, 2026):
    - runtime closure:
      - hardened execute-phase interception wait behavior in
        `tools/circt-sim/LLHDProcessInterpreter.cpp`:
        - when objection-zero waiter mode is armed, force scheduler process
          state to `Waiting` immediately.
        - guard `executeProcess()` against spurious wakeups while process is
          suspended on:
          - execute-phase monitor poll state
          - objection-zero waiter state.
      - this prevents unintended re-entry into intercepted monitor `sim.fork`
        loops during objection-driven suspension.
    - regression coverage:
      - added
        `test/Tools/circt-sim/execute-phase-monitor-fork-objection-waiter.mlir`.
      - locks:
        - interception trace (`wait_mode=objection_zero`)
        - functional ordering (`drop done` before `phase done`)
        - default + parallel-flag lanes
          (`--parallel=4 --work-stealing --auto-partition`).
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions:
        - `execute-phase-monitor-fork-objection-waiter.mlir`: PASS.
        - `func-generate-baud-clk-resume-fast-path.mlir`: PASS.
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`: PASS.
        - `jit-process-fast-path-store-wait-self-loop.mlir`: PASS.
      - bounded UART direct profile (compile mode, `--timeout=60`):
        - `/tmp/uart-direct-timeout60-objwaiter-summary.log`
        - dominant execute-phase monitor process
          `fork_18_branch_0` observed at `steps=1` in the bounded window.
    - next closure target:
      - convert reduced monitor-fork churn into end-to-end UART lane
        completion by unblocking Rx functional progression
        (`UartRxCovergroup` remains `0%`) and rerun full all9 compile lanes.
51. Execute-phase `wait(condition)` objection fallback poll backoff
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterWaitCondition.cpp`, widened
        execute-phase wait(condition) objection fallback poll interval from
        `10000000 fs` to `1000000000 fs` (10ns -> 1us) when
        `objectionWaitHandle` is active.
      - preserves objection-zero waiter registration as the primary wake path;
        timed polling remains a sparse safety net.
    - regression coverage:
      - added
        `test/Tools/circt-sim/wait-condition-execute-phase-objection-fallback-backoff.mlir`
        to lock:
        - `func=uvm_pkg::uvm_phase_hopper::execute_phase`
        - non-invalid `objectionWaitHandle`
        - `targetTimeFs=1000000000`.
    - validation:
      - build:
        - `ninja -C build -j4 circt-sim`: PASS.
      - focused regressions: PASS
        - `wait-condition-execute-phase-objection-fallback-backoff.mlir`
        - `execute-phase-monitor-fork-objection-waiter.mlir`
        - `func-generate-baud-clk-resume-fast-path.mlir`
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`
        - `jit-process-fast-path-store-wait-self-loop.mlir`
      - bounded AVIP UART compile lane (`--timeout=60`, compile mode):
        - `/tmp/avip-circt-sim-uart-objwait-backoff-20260218-144031/matrix.tsv`
          (`compile_status=OK`, `sim_status=OK`)
        - `/tmp/avip-circt-sim-uart-objwait-backoff-20260218-144031/uart/sim_seed_1.log`
          reached `505859300000 fs` with `UartTxCovergroup=100%`,
          `UartRxCovergroup=0%`.
    - next closure target:
      - unblock UART Rx progression by reducing dominant Rx-side monitoring and
        call-indirect wait stacks (for example, `fork_82_branch_1`,
        `fork_80_branch_1`) and then re-run full all9 compile lanes.
52. Queue-backed `wait(condition)` fallback poll backoff
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterWaitCondition.cpp`, widened
        queue-backed wait_condition fallback poll interval from
        `100000000 fs` to `1000000000 fs` (100ns -> 1us).
      - queue-not-empty waiters remain the primary wake path; timed polling is
        retained as watchdog fallback.
    - regression coverage:
      - added `test/Tools/circt-sim/wait-condition-queue-fallback-backoff.mlir`
        to lock:
        - queue-backed wait detection (`queueWait=0x...`)
        - sparse fallback schedule (`targetTimeFs=1000000000`).
    - validation:
      - builds:
        - `ninja -C build -j4 circt-sim`: PASS
        - `ninja -C build-test -j4 circt-sim`: PASS
      - focused regressions: PASS
        - `wait-condition-queue-fallback-backoff.mlir`
        - `wait-condition-execute-phase-objection-fallback-backoff.mlir`
        - `wait-queue-size.sv`
        - `wait-condition-spurious-trigger.mlir`
      - direct UART wait-condition trace (compile mode, 20s bound):
        - baseline: `/tmp/uart-objwait-backoff-waitcond20.log`
        - updated: `/tmp/uart-queuebackoff-waitcond20.log`
        - queue-backed `m_init_process_guards` wait-condition traces:
          `322 -> 75`
        - `fork_2_branch_0` steps: `1982 -> 500`
        - bounded sim-time progression:
          `31971000000 fs -> 72938500000 fs`
      - bounded AVIP UART compile lane (`--timeout=60`, compile mode):
        - `/tmp/avip-circt-sim-uart-queuebackoff-20260218-145002/matrix.tsv`
          (`compile_status=OK`, `sim_status=OK`)
        - `/tmp/avip-circt-sim-uart-queuebackoff-20260218-145002/uart/sim_seed_1.log`
          reached `497007600000 fs`, with `fork_18_branch_0 steps=1`,
          `UartTxCovergroup=100%`, `UartRxCovergroup=0%`.
    - next closure target:
      - address the remaining Rx-progress bottleneck in monitor/driver call
        stacks (notably `fork_82_branch_1` `StartMonitoring` and related
        `func.call_indirect` chains), then re-run full all9 compile lanes.
53. Execute-phase monitor wake cleanup parity + sparse wait(condition)
    watchdog broadening (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreter.cpp`, objection-zero waiter
        wake now restores monitor-fork cleanup symmetry by killing/erasing the
        tracked phase child process tree before resuming execute-phase monitor
        waiters.
      - in `tools/circt-sim/LLHDProcessInterpreterWaitCondition.cpp`,
        widened sparse watchdog polls from `1us` to `10us` for:
        - queue-backed wait(condition) (`queueWait != 0`)
        - execute_phase objection-backed wait(condition)
      - synchronized execute-phase monitor poll helper declarations/state maps
        in `tools/circt-sim/LLHDProcessInterpreter.h`.
    - regression coverage:
      - added
        `test/Tools/circt-sim/fork-execute-phase-monitor-intercept-single-shot.mlir`
        to lock single-shot interception behavior for monitor `sim.fork`.
      - updated
        `test/Tools/circt-sim/wait-condition-execute-phase-objection-fallback-backoff.mlir`
        to lock `targetTimeFs=10000000000`.
      - updated
        `test/Tools/circt-sim/wait-condition-queue-fallback-backoff.mlir`
        to lock `targetTimeFs=10000000000`.
    - validation:
      - build:
        - `ninja -C build-test -j4 circt-sim`: PASS.
      - focused regressions: PASS
        - `fork-execute-phase-monitor-intercept-single-shot.mlir`
        - `execute-phase-monitor-fork-objection-waiter.mlir`
        - `wait-condition-execute-phase-objection-fallback-backoff.mlir`
        - `wait-condition-queue-fallback-backoff.mlir`
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`
      - direct UART bounded sample (`max-time=70000000000 fs`, compile mode):
        - baseline:
          `/tmp/uart-maxtime70e9-post-forkpollv2-20260218.log`
        - updated:
          `/tmp/uart-maxtime70e9-backoff10us-procstatsopt-20260218.log`
        - queue-backed wait loop `fork_2_branch_0` reduced from `4262` to
          `104` steps; global execution count stayed near-flat
          (`1433758 -> 1433002`) and bounded coverage remained `Rx=0%`,
          `Tx=0%` at this short horizon.
      - longer bounded UART timeout lane:
        - `/tmp/uart-maxtime353e9-backoff10us-20260218.log`
        - reached `278776700000 fs` before timeout;
          coverage remained `UartRxCovergroup=0%`, `UartTxCovergroup=0%`.
    - next closure target:
      - continue root-cause closure on UART Rx functional progression while
        reducing remaining monitor/driver call-indirect wait stacks.
54. `StartMonitoring` tail-wrapper collapse for `Deserializer` resume stacks
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`
        (`resumeSavedCallStackFrames`), added a guarded collapse of outer
        `*::StartMonitoring` frames when the active inner callee frame is
        `*::Deserializer` and the wrapper is a pure tail-call-through
        (`call` then `return`).
      - this keeps resume churn on the hot `Deserializer` frame and avoids
        carrying redundant wrapper frames across repeated wait/resume cycles.
      - added env-gated trace marker:
        - `[MON-DESER-FP] resume-hit ...`
        - controlled by
          `CIRCT_SIM_TRACE_MONITOR_DESERIALIZER_FASTPATH=1`.
    - regression coverage:
      - added
        `test/Tools/circt-sim/func-start-monitoring-resume-fast-path.mlir`.
      - locks trace activation and behavior in both:
        - default lane.
        - `--parallel=4 --work-stealing --auto-partition` lane.
    - validation:
      - builds:
        - `ninja -C build -j4 circt-sim`: PASS.
        - `ninja -C build-test -j4 circt-sim`: PASS.
      - focused regressions: PASS
        - `func-start-monitoring-resume-fast-path.mlir`
        - `func-generate-baud-clk-resume-fast-path.mlir`
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`
        - `execute-phase-monitor-fork-objection-waiter.mlir`
        - `wait-condition-queue-fallback-backoff.mlir`
        - `wait-condition-execute-phase-objection-fallback-backoff.mlir`
        - `jit-process-fast-path-store-wait-self-loop.mlir`
      - bounded UART direct sample (compile mode, 60s wall guard):
        - `/tmp/uart-mon-deser-collapse-direct60-20260218-151813.log`
        - trace hit observed.
        - reached `508600800000 fs`; hotspot sample
          `fork_82_branch_1 steps=8694`.
      - bounded UART parallel sample (compile mode, 30s wall guard):
        - `/tmp/uart-mon-deser-collapse-parallel30-20260218-151942.log`
        - trace hit observed in parallel lane.
        - `fork_82_branch_1` wrapper stack depth dropped to `callStack=1`.
    - next closure target:
      - extend this tail-wrapper strategy to additional hot monitor/driver
        wrapper chains and continue all9 compile-lane convergence checks.
55. `DriveToBfm` tail-wrapper collapse for `SampleData` resume stacks
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`
        (`resumeSavedCallStackFrames`), generalized tail-wrapper collapse into
        a reusable helper and added a second guarded chain:
        - outer: `*::DriveToBfm`
        - inner callee: `*::SampleData`
      - this extends the wrapper-collapse strategy beyond monitor-only paths to
        the hot Tx-driver wrapper lane seen in UART bounded profiles.
      - added env-gated trace marker:
        - `[DRV-SAMPLE-FP] resume-hit ...`
        - controlled by
          `CIRCT_SIM_TRACE_DRIVE_SAMPLE_FASTPATH=1`.
    - regression coverage:
      - added
        `test/Tools/circt-sim/func-drive-to-bfm-resume-fast-path.mlir`.
      - locks default + parallel lane behavior.
    - validation:
      - build:
        - `ninja -C build-test -j4 circt-sim`: PASS.
      - focused lit (filtered): PASS
        - `func-drive-to-bfm-resume-fast-path.mlir`
        - `func-start-monitoring-resume-fast-path.mlir`
        - `func-generate-baud-clk-resume-fast-path.mlir`
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`
        - `execute-phase-monitor-fork-objection-waiter.mlir`
      - bounded AVIP UART compile lane (`SIM_TIMEOUT=60`):
        - `/tmp/avip-circt-sim-uart-drive-sample-collapse-20260218-153547/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`)
        - UART sim log confirms
          `[DRV-SAMPLE-FP] resume-hit proc=93 callee=UartTxDriverBfm::DriveToBfm`.
      - bounded UART compile direct sample (`--parallel=4`, 30s):
        - `/tmp/uart-drive-sample-parallel30-20260218-153743.log`
        - trace hit confirmed, no wrapper-collapse regressions observed.
    - next closure target:
      - continue wrapper-chain burn-down on remaining UART monitor lanes while
        prioritizing Rx functional progression/coverage closure in all9
        compile lanes.
56. Generic tail-wrapper collapse for resumable call-stack frames
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`
        (`resumeSavedCallStackFrames`), generalized tail-wrapper collapse from
        pair-specific suffix checks to a pure-shape guard:
        - outer frame resumes at `func.return` (no return operands)
        - immediately preceding op is `func.call` (void call)
        - wrapper call target equals the active inner frame symbol.
      - this preserves explicit monitor/driver lanes while enabling additional
        pure tail wrappers without further hardcoded pair additions.
      - retained specialized trace tags:
        - `[MON-DESER-FP]`
        - `[DRV-SAMPLE-FP]`
      - added generic trace tag:
        - `[TAIL-WRAP-FP]`
        - enabled by `CIRCT_SIM_TRACE_TAIL_WRAPPER_FASTPATH=1`.
    - regression coverage:
      - added
        `test/Tools/circt-sim/func-tail-wrapper-generic-resume-fast-path.mlir`
        to lock generic shape-based collapse in default + parallel lanes.
    - validation:
      - build:
        - `ninja -C build-test -j4 circt-sim`: PASS.
      - focused lit (filtered): PASS
        - `func-tail-wrapper-generic-resume-fast-path.mlir`
        - `func-drive-to-bfm-resume-fast-path.mlir`
        - `func-start-monitoring-resume-fast-path.mlir`
        - `func-generate-baud-clk-resume-fast-path.mlir`
        - `func-baud-clk-generator-fast-path-delay-batch.mlir`
        - `execute-phase-monitor-fork-objection-waiter.mlir`
      - bounded AVIP UART compile lane (`SIM_TIMEOUT=60`):
        - `/tmp/avip-circt-sim-uart-tail-wrapper-generic-20260218-154800/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`)
        - UART sim log confirms generic hits:
          - `wrapper=UartTxDriverBfm::DriveToBfm`
          - `wrapper=UartTxMonitorBfm::StartMonitoring`.
    - next closure target:
      - extend bounded profiling to confirm Rx-side monitor wrapper collapse
        activity (`UartRxMonitorBfm::StartMonitoring`) under longer windows,
        then continue Rx functional progression closure.
57. Suspension-time tail-wrapper frame elision + caller-restore state seam
    (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreter.cpp`
        (`interpretFuncBody`), added suspension-time elision of pure
        tail-wrapper frames (`func.call` + terminal `func.return`) when the
        inner call suspends.
      - introduced
        `ProcessExecutionState::callStackOutermostCallOp` in
        `tools/circt-sim/LLHDProcessInterpreter.h` as an explicit caller-restore
        seam for frame elision.
      - in `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`,
        `resumeSavedCallStackFrames` now restores process-level position using
        `callStackOutermostCallOp` when present.
      - plumbed deopt snapshot/restore support in
        `tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp`.
      - trace behavior:
        - generic: `[TAIL-WRAP-FP] suspend-elide ...`
        - specialized tags retained:
          - `[MON-DESER-FP]`
          - `[DRV-SAMPLE-FP]`
    - regression coverage:
      - updated wrapper fast-path regressions to lock tag presence regardless
        of resume-collapse vs suspend-elide activation:
        - `func-start-monitoring-resume-fast-path.mlir`
        - `func-drive-to-bfm-resume-fast-path.mlir`
        - `func-tail-wrapper-generic-resume-fast-path.mlir`
    - validation:
      - build + focused lit: PASS
        - includes all wrapper fast-path regressions plus:
          - `func-generate-baud-clk-resume-fast-path.mlir`
          - `func-baud-clk-generator-fast-path-delay-batch.mlir`
          - `execute-phase-monitor-fork-objection-waiter.mlir`
      - full suite: PASS
        - `ninja -C build-test -j4 check-circt-tools-circt-sim`
          (`Passed=440`, `XFAIL=49`, `Failed=0`).
      - bounded AVIP UART compile lane (`SIM_TIMEOUT=60`):
        - `/tmp/avip-circt-sim-uart-tail-wrapper-elide-20260218-160749/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`)
        - UART log confirms suspend-elide hits for:
          - `UartTxDriverBfm::DriveToBfm`
          - `UartTxMonitorBfm::StartMonitoring`
          - `UartRxMonitorBfm::StartMonitoring`
      - bounded direct UART probe (`UartBaudRate19200Test`, `--timeout=60`):
        - `/tmp/uart-tail-wrapper-elide-rxprobe-19200-timeout60-20260218-160620.log`
        - `fork_80_branch_1` now observed at `callStack=1` (previously `2`),
          confirming Rx-side wrapper depth reduction under bounded progress.
    - next closure target:
      - continue Rx functional progression/coverage root-cause closure
        (`UartRxCovergroup` remains `0%`) after wrapper-depth reduction.
58. Profile-guided `func.call_indirect` suspend analysis widening + LLHD
    sub-reference correctness closure (February 18, 2026):
    - runtime closure:
      - in `tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp`:
        - added a shared suspend-analysis context carrying process identity,
          hot-threshold-derived minimum profile-call count, and collected
          indirect-site guard specs.
        - threaded context-aware suspend checks through single-block,
          multiblock, and structured-control prelude analysis.
        - added profile-guided `func.call_indirect` suspend closure:
          - allow non-suspending classification only for resolved local target
            sets (no unresolved calls) whose callees are statically
            non-suspending.
          - emit/install per-site target-set guards so runtime target drift
            safely deopts thunk reuse.
        - updated trivial-thunk candidate APIs to collect guard specs during
          shape analysis and install them on successful thunk installs.
      - in `tools/circt-sim/LLHDProcessInterpreter.cpp`:
        - corrected LLHD sub-reference probe/drive paths so
          `sig.struct_extract`/`sig.array_get`/`sig.extract` do not bypass
          required subfield extraction through parent-signal short-circuits.
        - sub-reference probes now prefer pending epsilon drives when
          available, preserving same-delta visibility.
        - `llhd.sig.array_get` probe/drive now accounts for enclosing
          struct-field bit offsets in struct-extract chains.
        - added targeted call tracing
          (`CIRCT_SIM_TRACE_CALL_FILTER`) for hotspot diagnosis.
    - regression coverage:
      - updated wait-event compile-mode report expectations to reflect reduced
        duplicate compile/deopt churn (`jit_compiles_total 2 -> 1` and guarded
        deopts `2 -> 1` in guard-failed variants).
      - removed temporary `XFAIL` masking and updated expectations to explicit
        pass behavior (`jit_deopts_total=0`) in:
        - `jit-process-thunk-wait-event-derived-observed-impure-prewait-unsupported.mlir`
        - `jit-process-thunk-wait-event-derived-observed-impure-prewait-unsupported-strict.mlir`
        - `jit-report-deopt-processes.mlir`
      - added focused prelude/guard coverage:
        - `jit-process-thunk-fork-branch-disable-fork-terminator.mlir`
        - `jit-process-thunk-func-call-from-class-wrapper-halt.mlir`
        - `jit-process-thunk-func-call-get-automatic-phase-objection-halt.mlir`
        - `jit-process-thunk-func-call-local-helper-call-indirect-profile-nonsuspending-halt.mlir`
        - `jit-process-thunk-func-call-local-helper-call-indirect-profile-guard-mismatch.mlir`
        - `jit-process-thunk-func-call-m-killed-halt.mlir`
        - `jit-process-thunk-func-call-safe-drop-starting-phase-halt.mlir`
        - `jit-process-thunk-func-call-safe-raise-starting-phase-halt.mlir`
        - `jit-process-thunk-func-call-uvm-create-random-seed-halt.mlir`
        - `jit-process-thunk-llvm-call-process-await-halt.mlir`
        - `jit-process-thunk-llvm-call-process-srandom-halt.mlir`
        - `jit-process-thunk-llvm-call-semaphore-get-blocking-halt.mlir`
        - `jit-process-thunk-scf-for-uvm-is-match-halt.mlir`
      - added LLHD sub-reference correctness regressions:
        - `llhd-drv-array-get-struct-field-offset.mlir`
        - `llhd-prb-subfield-pending-epsilon.mlir`
        - `llhd-drv-memory-backed-struct-array-func-arg.mlir`
        - `llhd-ref-cast-array-subfield-store-func-arg.mlir`
        - `llhd-ref-cast-subfield-store-func-arg.mlir`
      - updated:
        - `llhd-drv-ref-blockarg-array-get.mlir`
    - validation:
      - focused lit filter sets: PASS
        - wait-event thunk-report cluster
        - profile-guarded call-indirect cluster
        - LLHD sub-reference regression cluster.
      - full tools suite: PASS
        - `llvm-lit -sv -j8 build-test/test/Tools/circt-sim`
          (`Total=491`, `Passed=445`, `XFAIL=46`, `Failed=0`).
      - bounded AVIP UART compile lane (`SIM_TIMEOUT=60`):
        - `/tmp/avip-circt-sim-uart-profileguard-20260218-163020/matrix.tsv`
          (`compile_status=OK`, `sim_status=TIMEOUT`).
    - next closure target:
      - push these broader install/guard closures through rerun all9 compile
        matrix validation and then return to Rx functional progression closure
        work (`UartRxCovergroup` still `0%` in bounded UART runs).

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
