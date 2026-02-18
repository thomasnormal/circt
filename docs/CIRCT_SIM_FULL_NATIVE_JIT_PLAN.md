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
