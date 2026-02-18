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
Observed repeatedly in recent runs (2026-02-17 to 2026-02-18):
1. Scoreboard mismatch:
   - `../mbit/i3c_avip/src/hvl_top/env/i3c_scoreboard.sv(162)`
   - sometimes also `(128)` transaction-count mismatch.
2. Coverage can print `100% / 100%` while scoreboard still fails.
3. In some compile-mode traces, repeated UVM fatals:
   - `An uvm_test_top already exists via a previous call to run_test`
   - illegal component creation after build phase.

## Investigation Timeline

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
3. I3C AVIP still not at parity despite those fixes.

## Immediate Next Steps
1. Add a dedicated I3C runtime trace mode that tags:
   - parent/child process IDs
   - function name
   - call stack depth
   - current/destination block identity
   at every `sim.disable_fork` and fork resume/kill boundary.
2. Add a regression that fails if `uvm_pkg::uvm_root::run_test` is entered more
   than once in a single simulation.
3. Tighten JIT thunk guardrails:
   - deopt on any region mismatch for active call-stack resumes.
   - block process-body thunk execution when function-body resume is active.
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
