# I3C Engineering Log

## Scope
Track all I3C-specific parity and performance debugging in one place, separate
from the global `avip_engineering_log.md`.

## Goal
Match Xcelium for I3C on:
1. Functional behavior (no scoreboard mismatches).
2. UVM health (`UVM_ERROR=0`, `UVM_FATAL=0`).
3. Coverage parity (reference: `35.19% / 35.19%`).
4. Runtime stability in `circt-sim` compile mode (no re-entrant lifecycle
   failures, bounded memory/time).

## Baseline Reference (Xcelium)
- Date: 2026-02-18
- Command path: `utils/run_avip_xcelium_reference.sh` (seed 1)
- Result:
  - `UVM_ERROR=0`
  - `UVM_FATAL=0`
  - Simulation completes around `3970 ns`
  - Coverage `35.19% / 35.19%`

## Current circt-sim Symptoms
Observed repeatedly in recent runs (2026-02-17 to 2026-02-19):
1. Scoreboard mismatch:
   - `../mbit/i3c_avip/src/hvl_top/env/i3c_scoreboard.sv(162)`
   - sometimes also `(128)` transaction-count mismatch.
2. Coverage can print `100% / 100%` while scoreboard still fails.
3. In some compile-mode traces, repeated UVM fatals:
   - `An uvm_test_top already exists via a previous call to run_test`
   - illegal component creation after build phase.
4. Latest deterministic compile lane (`2026-02-19`) has zero JIT deopt rows,
   but still hits:
   - `UVM_ERROR ... i3c_scoreboard.sv(162)`
   - `virtual method call (func.call_indirect) failed ... uvm_pkg::uvm_root::die`

## Investigation Timeline

### 2026-02-19: run_test guard + sampleWriteDataAndACK ordering regression + cross-AVIP parity check
Files:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`
- `test/Tools/circt-sim/uvm-run-test-single-entry-guard.mlir`
- `test/Tools/circt-sim/i3c-samplewrite-disable-fork-ordering.sv`

Key changes:
1. Added env-gated UVM run_test single-entry guard:
   - `CIRCT_SIM_ENFORCE_SINGLE_RUN_TEST=1` emits error on re-entry.
   - `CIRCT_SIM_TRACE_UVM_RUN_TEST=1` traces run_test entry points.
2. Added `sampleWriteDataAndACK`-style ordering regression for
   `join_any + disable_fork` with `CIRCT_SIM_TRACE_I3C_FORK_RUNTIME=1`.
3. Softened `uvm_pkg::uvm_root::die` failure handling in direct and indirect
   call-failure paths to avoid generic internal-failure warning spam.

Validation:
- Focused lit (`4/4`) PASS:
  - `i3c-samplewrite-disable-fork-ordering.sv`
  - `fork-disable-ready-wakeup.sv`
  - `fork-disable-defer-poll.sv`
  - `uvm-run-test-single-entry-guard.mlir`
- Bounded compile-mode AVIP matrix:
  - `/tmp/avip-circt-sim-i3c-plus-5-20260219-004240/matrix.tsv`
  - AVIPs: `i3c,apb,ahb,uart,spi,jtag` (seed 1).
  - compile/sim: all six lanes `OK`.
  - JIT: `deopt_process_rows=0` on all six lanes.
  - parity failures are cross-AVIP, not I3C-only:
    - `i3c_scoreboard.sv(162)`
    - `apb_scoreboard.sv(272/283/294/305/318)`
    - `AhbScoreboard.sv(243/267/278)`
    - `SpiScoreboard.sv(204)`
  - `uart`/`jtag`: no `UVM_ERROR`/`UVM_FATAL` lines in this run.
  - no `uvm_pkg::uvm_root::die` call-indirect warning seen in this matrix.

### 2026-02-18: Trivial thunk fallback safety hardening
Files:
- `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`

Key change:
- On trivial-thunk fallback with active saved call stack, deopt instead of
  finalizing the process.
- Added tighter fallback side-effect gating for process/initial shapes.

Result:
- Closed one unsafe finalize path.
- I3C still mismatched at scoreboard line 162.

### 2026-02-18: `sim.disable_fork` deferred-kill guard
Files:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `test/Tools/circt-sim/fork-disable-ready-wakeup.sv`
- `test/Tools/circt-sim/fork-disable-defer-poll.sv`

Key change:
- Added deferred `disable_fork` with tokenized scheduling and bounded poll.
- Added trace hooks:
  - `[DISABLE-FORK-DEFER]`
  - `[DISABLE-FORK-DEFER-POLL]`
  - `[DISABLE-FORK-DEFER-FIRE]`

Result:
- Focused regressions pass.
- I3C still fails; children can still be killed while `waiting=1`.

### 2026-02-18: Execute-phase objection lifecycle stabilization
Files:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreter.h`

Key change:
- Replaced descendant-progress completion with objection startup/drop grace
  tracking in execute-phase monitor interception.

Result:
- Timeout regressions improved for I3C (simulation completes more often).
- Core mismatch remains.

### 2026-02-19: Multiblock thunk waiting-resume normalization
Files:
- `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp`
- `unittests/Tools/circt-sim/LLHDProcessInterpreterTest.cpp`

Key change:
- Fixed non-exclusive waiting-state handling in
  `executeMultiBlockTerminatingNativeThunk` so saved-call-stack and sequencer
  retry resumes do not fall through to `unexpected_resume_state`.

Result:
- New unit path (`MultiBlockThunkResumesSavedCallStackWhenWaiting`) now passes.
- I3C compile lane deopt summary moved from repeated
  `guard_failed:post_exec_not_halted` rows to `deopt_process_rows=0`.
- Functional parity mismatch remains (scoreboard line 162).

### 2026-02-19: I3C fork-runtime trace mode
Files:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `test/Tools/circt-sim/fork-disable-ready-wakeup.sv`

Key change:
- Added `CIRCT_SIM_TRACE_I3C_FORK_RUNTIME=1`.
- New trace points at:
  - `join_none` child readiness check/resume boundaries.
  - `sim.disable_fork` enter/defer/kill/resume boundaries.
- Each trace line includes:
  - parent/child process IDs
  - scheduler state
  - function name
  - call-stack depth
  - current/destination block identity (+ op summary).

Result:
- Deterministic I3C compile run with trace emitted 171
  `[I3C-FORK-RUNTIME]` lines:
  - `join_none_check=34`, `join_none_resume=34`
  - `disable_fork_enter=35`, `disable_fork_defer=17`
  - `disable_fork_kill=34`, `disable_fork_resume_parent=17`.
- Confirms repeated monitor-path pattern in
  `i3c_*_monitor_bfm::sampleWriteDataAndACK` around deferred/immediate child
  kills.

## Critical Evidence

### A) `disable_fork` still kills waiting monitor children
Trace logs show patterns like:
- child scheduler state `Waiting`
- interpreter state `waiting=1`
- then killed in deferred fire path.

This indicates wake-consumption/order semantics are still not equivalent to
expected runtime behavior for the monitor fork pattern.

### B) Re-entrant `run_test` appears in some compile-mode runs
From `/tmp/i3c-disabledeferpoll-trace.log`:
- Sequence-related processes later create forks with function context
  `uvm_pkg::uvm_root::run_test`.
- This is followed by repeated `uvm_test_top already exists` fatals.

This strongly suggests a deep process state/call-stack resume drift in a subset
of JIT/thunk paths (or a related region-selection/resume-state bug).

### C) Latest trace confirms monitor `disable_fork` churn despite zero deopts
From
`/tmp/avip-circt-sim-i3c-runtime-trace-20260219-001247/compile/i3c/sim_seed_1.log`:
- No JIT deopt rows in compile-mode report.
- Repeated `join_none` + `disable_fork` cycles in monitor functions, including
  deferred and immediate kill paths.
- Scoreboard mismatch at line 162 still occurs.

## Working Hypotheses
1. Primary parity mismatch is still tied to fork child wake/kill ordering in
   monitor paths (`sampleWriteDataAndACK` subtree).
2. A secondary but severe compile-mode bug can re-enter UVM lifecycle
   (`run_test`) due to incorrect resume/region state under native thunk paths.
3. Both issues can coexist in the same lane and compound failures.

## What Is Verified
1. Build passes for the current circt-sim changes.
2. New fork defer regressions pass:
   - `test/Tools/circt-sim/fork-disable-ready-wakeup.sv`
   - `test/Tools/circt-sim/fork-disable-defer-poll.sv`
3. Saved-call-stack multiblock thunk regression/unit checks pass.
4. Deterministic I3C compile lane now has `deopt_process_rows=0`.
5. I3C AVIP still not at parity despite those fixes.

## Immediate Next Steps
1. Add a regression that fails if `uvm_pkg::uvm_root::run_test` is entered more
   than once in a single simulation.
2. Add focused regression(s) for monitor `disable_fork` ordering using the new
   trace mode to lock expected parent/child resume-kill ordering.
3. Investigate/triage the `uvm_pkg::uvm_root::die` call_indirect failure path
   observed in compile-mode I3C logs.
4. Re-run I3C compile lane with deterministic seed after each change and record:
   - `UVM_FATAL`, `UVM_ERROR`
   - scoreboard line hits
   - simulated time at first mismatch/fatal
   - coverage pair.

## Success Criteria to Close This Log
1. `UVM_FATAL=0`, `UVM_ERROR=0` on I3C seed 1 compile mode.
2. No `uvm_test_top already exists` in logs.
3. No scoreboard mismatch at lines 128/162.
4. Coverage and completion behavior align with Xcelium baseline.

### 2026-02-18: `disable_fork` defer-scope cleanup (non-JIT overlap)
Files:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `test/Tools/circt-sim/fork-disable-defer-poll.sv`

Why:
- Trace showed repeated defer/fire on monitor children in `Waiting` state,
  while still observing scoreboard mismatch (`check_phase` writeData compare).
- To avoid overlap with ongoing broad JIT work, this pass narrowed to runtime
  fork semantics only.

Changes:
1. `shouldDeferDisableFork` now defers only for child scheduler state:
   - `Ready` or `Suspended`
   - excludes pure `Waiting`.
2. Immediate/deferred disable loops now skip already-halted children.
3. Waiting-child regression expectation updated:
   - `fork-disable-defer-poll.sv` expects immediate kill path.

Observed validation state:
- Intermediate run caught a regression in `fork-disable-ready-wakeup.sv`
  (child wake lost when defer was too strict).
- Gate was corrected to include `Suspended`.
- Final focused rerun is pending due concurrent build lock contention in
  `build-test`.

Current status:
- This is a cleanup/stabilization step; I3C parity is **not** yet closed.
- No new claim of scoreboard parity is made in this entry.

### 2026-02-18 Verification Addendum
Focused fork-disable regressions rechecked with direct commands (to avoid shared
build lock contention):
- `fork-disable-ready-wakeup.sv`: PASS
- `fork-disable-defer-poll.sv`: PASS
- `disable-fork-halt.mlir`: PASS

Note:
- This validates the cleanup regression set only; it does not yet close the
  I3C scoreboard parity gap.

### 2026-02-19: Refresh lane after recent runtime/import changes
Command:
- `AVIPS=i3c SEEDS=1 CIRCT_SIM_MODE=compile CIRCT_SIM_WRITE_JIT_REPORT=1 COMPILE_TIMEOUT=300 SIM_TIMEOUT=240 SIM_TIMEOUT_GRACE=30 utils/run_avip_circt_sim.sh /tmp/avip-circt-sim-i3c-refresh-20260219-004123`

Primary artifacts:
- Matrix: `/tmp/avip-circt-sim-i3c-refresh-20260219-004123/matrix.tsv`
- Sim log: `/tmp/avip-circt-sim-i3c-refresh-20260219-004123/i3c/sim_seed_1.log`

Observed result:
- Compile: `OK` (`32s`)
- Sim: `OK` exit (`75s` wall)
- Main-loop termination: `maxTime reached (7940000000000 fs)`
- Scoreboard mismatch persists:
  - `UVM_ERROR ... i3c_scoreboard.sv(162)` occurred once
- No re-entrant lifecycle fatal in this run:
  - `uvm_test_top already exists` occurrences: `0`
- Coverage printout from covergroups in this run:
  - `i3c_controller_covergroup`: `100.00%`
  - `target_covergroup`: `100.00%`

Interpretation:
- Current I3C blocker remains functional parity (scoreboard data mismatch), not compile viability.
- The severe `run_test` re-entry fatal is not deterministic in every lane; keep separate tracking, but current failure is still line-162 mismatch.

Additional focused check from this session:
- `test/Tools/circt-sim/task-output-struct-default.sv` passes and prints:
  - `count=0 data=42`
- This confirms baseline `output` argument defaulting behavior in a minimal task case.
- However, that minimal pass does not close the I3C scoreboard gap.

Next debugging step (root-cause oriented):
1. Build a narrow reproducer around I3C target-side write-data conversion path (`to_class`/`from_class`) and compare controller-vs-target payload at first divergence.
2. Add a dedicated regression that fails on first mismatched writeData sample (without full AVIP runtime).
3. Re-run deterministic lane and require: no line-162 mismatch and no lifecycle re-entry fatal.
