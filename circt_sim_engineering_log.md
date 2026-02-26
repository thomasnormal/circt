# CIRCT Sim Engineering Log

## 2026-02-26
- Added `utils/run_sv_tests_waveform_diff.sh` to run `sv-tests` simulation cases
  across four lanes with waveform artifacts:
  - `interpret_off` (`--mode=interpret`, `CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=0`)
  - `interpret_on` (`--mode=interpret`, `CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=1`)
  - `compile_aot` (`--mode=compile`)
  - `xcelium` (`xrun -input wave.tcl` with VCD database/probe)
- Runner emits lane matrices in AVIP-compatible schema and auto-generates
  pairwise waveform parity TSVs via `utils/check_avip_waveform_matrix_parity.py`:
  - interpret_off vs interpret_on
  - interpret_off vs compile_aot
  - interpret_off vs xcelium
- Hardened against concurrent tool rebuild races by snapshotting
  `build_test/bin/circt-verilog` and `build_test/bin/circt-sim` into the run
  output directory before test execution.
- Fixed arg forwarding bug in `utils/run_avip_circt_fastpath_ab.sh` where
  `--compare-arg` values starting with `--` were parsed as missing arguments;
  now passed as `--compare-arg=<value>`.
- Smoke validation:
  - `sv_tests_sim_smoke`-style 3-case run produced:
    - `interpret_off` vs `interpret_on`: 3/3 waveform matches
    - `interpret_off` vs `compile_aot`: 3/3 waveform matches
    - `interpret_off` vs `xcelium`: 3/3 mismatches
- Realization: cross-engine mismatch reports were initially dominated by
  timescale formatting (`1fs` vs `1ns`) rather than behavior; default compare
  args now include `--ignore-timescale` to focus on signal/event differences.

## 2026-02-26
- Reproduced two deterministic `circt-sim` regressions before code changes:
  - `llhd-process-wait-no-delay-no-signals.mlir` executed `2` processes instead of expected `3`.
  - `llhd-process-wait-condition-func.mlir` completed at `1000000000 fs` instead of `1000000 fs`.
- Realization: empty-sensitivity wait fallback in `interpretWait` was effectively one-shot; subsequent visits cached wait state but did not re-schedule process wakeup.
- Surprise: wait-condition memory path *did* wake on the 5 fs store, but a previously scheduled sparse timed poll at 1e9 fs remained in the queue and dragged final simulation time forward.
- Fix 1: changed empty-sensitivity fallback to re-arm next-delta every encounter (correctness-first for interpreted mode).
- Fix 2: changed single-load memory-backed wait-condition polling from delta-churn + 1 us sparse fallback to fixed 1 ps cadence, while keeping memory waiters enabled.
- Verification:
  - Direct runs for both reproducer tests now match expected process counts/timestamps.
  - FileCheck invocations for both tests pass.
  - Focused lit sweep `--filter='llhd-process-.*wait'` passed 7/7.

- Additional similarity sweep found another stale-timed-poll bug in queue-backed
  `wait(condition)`:
  - Reproducer: queue wakeup at `5 fs` completed process logic, but final sim
    time still advanced to `10000000000 fs` due an old fallback poll event.
- Fix: queue-backed wait_condition now relies on event-style queue wakeups
  (`timedPoll=0`) instead of scheduling a timed watchdog poll that can become
  stale.
- Added regression: `test/Tools/circt-sim/wait-condition-queue-no-stale-time.mlir`
  asserting completion remains at the real wakeup time (`5 fs`).
- Updated queue fallback trace test:
  `test/Tools/circt-sim/wait-condition-queue-fallback-backoff.mlir` now checks
  `timedPoll=0`.
- Verification after this fix:
  - Targeted wait-condition lit subset passed.
  - Broader wait/wait-condition lit filter passed (20/20).
- Additional similarity sweep found stale timed-poll final-time drift in
  execute-phase objection waits:
  - Reproducer: `uvm_phase_hopper::execute_phase` waiter woke at `5 fs`, but
    final time advanced to `10000000000 fs` from a stale fallback poll event.
- Fix: for execute-phase wait_condition with active objections (`count > 0`),
  use objection event wake only (`timedPoll=0`); keep sparse timed fallback
  when count is zero.
- Added regression:
  `test/Tools/circt-sim/wait-condition-execute-phase-objection-no-stale-time.mlir`
  checking completion at `5 fs`.

- Found another similar stale-time case for non-UVM single-load memory waits:
  - Reproducer completed logic at `5 fs` but ended simulation at `1000000 fs`
    due stale memory poll callback.
- Fix: single-load memory waits now run event-only wakeups by default;
  `uvm_phase::wait_for_state` remains on timed polling fallback for unchanged
  memory polling semantics.
- Added regression:
  `test/Tools/circt-sim/wait-condition-memory-no-stale-time.mlir`
  checking completion at `5 fs`.
- Verification after both fixes:
  - `ninja -C build_test circt-sim`
  - `llvm/build/bin/llvm-lit -a --filter='wait-condition-memory-no-stale-time|wait-condition-memory-poll-recheck|wait-condition-execute-phase-objection-no-stale-time|wait-condition-execute-phase-objection-fallback-backoff|wait-condition-queue-no-stale-time|wait-condition-queue-fallback-backoff' build_test/test/Tools/circt-sim`
  - Result: 6/6 targeted tests passed.
- Follow-up similarity sweep (after waiting for concurrent workspace activity)
  on wait/poll paths:
  - Ran: `llvm/build/bin/llvm-lit -s --filter='wait_for_waiters|wait_for_self_and_siblings_to_drop|wait-condition|wait-event|wait-queue|objection|seq-get-next-item|phase-hopper' build_test/test/Tools/circt-sim`
  - Result: 30/30 passed.
- Transient surprise observed before the settled rerun: many lit failures with
  `Permission denied` launching `build_test/bin/circt-sim`; this disappeared
  after waiting and rerunning, consistent with concurrent binary replacement.
- No additional deterministic stale-timed-poll/final-time-drift bug was
  reproduced beyond the queue, execute-phase objection, and non-UVM memory
  wait-condition cases already fixed.
- Audit finding (new): mixed wait-condition modes can still leave stale timed
  callbacks that advance final simulation time.
  - Reproducer shape: process first executes `uvm_phase::wait_for_state`
    (timed memory poll armed), then later executes queue-backed
    `wait(__moore_queue_size(...) > 0)` (event-only path).
  - Observed with `--skip-passes`: process logic completes and prints `done`,
    but simulation finishes at `1000000 fs` from the stale earlier timed poll.
  - Root cause: prior timed callback remains in EventScheduler with no
    cancellation path; switching to event-only waiting does not remove it.
- Fix for mixed-mode stale timed polls:
  - Added event tagging + cancellation APIs to scheduler internals:
    `DeltaCycleQueue::cancelByTag`, `TimeWheel::cancelByTag`,
    `EventScheduler::cancelEventsByTag`.
  - Tagged `wait_condition` timed poll callbacks by process and cancel tagged
    callbacks when:
    - `__moore_wait_condition` observes `condition=true`
    - wait strategy chooses event-only wakeup (`timedPoll=0`)
  - This physically removes stale callbacks from the queue (not just logical
    token invalidation), so final simulation time cannot drift to stale
    watchdog timestamps.
- Added regression:
  - `test/Tools/circt-sim/wait-condition-mixed-mode-no-stale-time.mlir`
    (`wait_for_state` timed poll followed by queue-backed event wake),
    asserting completion at `7 fs`.
- TDD verification:
  - Baseline before fix:
    `build_test/bin/circt-sim --skip-passes --top=top --sim-stats /tmp/waitcond_mixed_stale_time.mlir`
    completed logic but ended at `1000000 fs`.
  - After fix, same reproducer ends at `7 fs`.
  - Targeted lit run:
    `llvm/build/bin/llvm-lit -sv --filter='wait-condition-mixed-mode-no-stale-time|wait-condition-memory-no-stale-time|wait-condition-memory-poll-recheck|wait-condition-queue-no-stale-time|wait-condition-queue-fallback-backoff|wait-condition-execute-phase-objection-no-stale-time|wait-condition-execute-phase-objection-fallback-backoff' build_test/test/Tools/circt-sim`
    passed 7/7.
- Similarity audit follow-up:
  - Reviewed other wait/poll scheduling sites (`LLHDProcessInterpreter.cpp`,
    `LLHDProcessInterpreterNativeThunkExec.cpp`) for mixed-mode
    wait-condition-like state transitions.
  - No additional deterministic stale-final-time regression reproduced in this
    pass beyond the mixed wait_condition mode fixed above.
- Post-fix test updates:
  - Focused regression subset passed:
    `llvm/build/bin/llvm-lit -sv --filter='wait_for_waiters|wait_for_self_and_siblings_to_drop|seq-get-next-item|wait-condition-mixed-mode-no-stale-time' build_test/test/Tools/circt-sim`
    -> 4/4 passed.
  - A broader 31-test wait/objection sweep was started but did not produce
    timely output in this workspace state; switched to focused subsets to keep
    feedback deterministic.

## 2026-02-26
- Audit finding: `EventScheduler::runUntil(maxTime)` advanced simulation time
  to the next event even when that event was beyond `maxTime`.
  - Repro via new unit test:
    `EventScheduler.RunUntilDoesNotAdvanceBeyondLimit`.
  - Before fix: scheduling an event at `100 fs` and calling `runUntil(50)`
    returned `end.realTime == 100` (unexpected overshoot), with event still
    unprocessed.
- Fix: gate `advanceToNextEvent()` in `EventScheduler::runUntil` using
  `findNextEventTime()` and stop when `nextTime.realTime > maxTime`.
  Also only increment `realTimeAdvances` when real time actually increases.
- Added/updated unit coverage in
  `unittests/Dialect/Sim/EventQueueTest.cpp`:
  - `EventScheduler.RunUntilDoesNotAdvanceBeyondLimit` (new regression)
  - `EventScheduler.CancelEventsByTagRemovesTaggedOnly`
  - `EventScheduler.CancelByTagInsideExtraDeltaCallback`
  - `EventScheduler.CancelByTagInsideExtraDeltaSameDeltaReschedule`
- Verification:
  - `ninja -C build_test CIRCTSimTests`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='EventScheduler.RunUntilDoesNotAdvanceBeyondLimit:EventScheduler.DelayedScheduling:EventScheduler.Statistics:EventScheduler.Reset:EventScheduler.CancelEventsByTagRemovesTaggedOnly:EventScheduler.CancelByTagInsideExtraDeltaCallback:EventScheduler.CancelByTagInsideExtraDeltaSameDeltaReschedule'`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='EventScheduler.*:TimeWheel.*:DeltaCycleQueue.*:Event.*:SimTime.*:SchedulingRegion.*'`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='ProcessScheduler.SuspendUntilTime:ProcessSchedulerIntegration.DelayedEventAdvancesTime:ProcessSchedulerIntegration.MultipleDelayedEvents:ProcessSchedulerIntegration.DelayedWaitPattern:ProcessSchedulerIntegration.EventSchedulerIntegrationCheck'`
  - All selected tests passed.

## 2026-02-26
- Additional audit finding: `ProcessScheduler::runUntil(maxTime)` also advanced
  to future wakeups beyond `maxTime`, mirroring the EventScheduler horizon bug.
  - Reproducer unit test:
    `ProcessSchedulerIntegration.RunUntilDoesNotAdvanceBeyondLimit`.
  - Before fix: scheduling an event at `100 fs` and calling `runUntil(50)`
    returned `end.realTime == 100` and executed the callback.
- Fix: in `ProcessScheduler::runUntil`, compute the next wake source across:
  - event scheduler (`peekNextRealTime`)
  - active clock domains (`nextWakeFs`)
  - active minnows (`nextWakeFs`)
  and stop if that minimum wake is beyond `maxTime`.
- Added regression:
  - `unittests/Dialect/Sim/ProcessSchedulerTest.cpp`
    `ProcessSchedulerIntegration.RunUntilDoesNotAdvanceBeyondLimit`.
- Verification:
  - `ninja -C build_test CIRCTSimTests`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='ProcessSchedulerIntegration.RunUntilDoesNotAdvanceBeyondLimit:ProcessScheduler.SuspendUntilTime:ProcessSchedulerIntegration.DelayedEventAdvancesTime:ProcessSchedulerIntegration.MultipleDelayedEvents:ProcessSchedulerIntegration.DelayedWaitPattern:ProcessSchedulerIntegration.EventSchedulerIntegrationCheck'`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='EventScheduler.*:ProcessScheduler.*:ProcessSchedulerIntegration.*'`
  - All selected tests passed.

## 2026-02-26
- Additional audit finding in parallel mode:
  `ParallelScheduler::runParallel(maxTime)` advanced beyond `maxTime`.
  - Reproducer unit test:
    `ParallelSchedulerTest.RunParallelDoesNotAdvanceBeyondLimit`.
  - Before fix: event at `100 fs` with `runParallel(50)` returned
    `end.realTime == 100`.
- Follow-up audit finding:
  `runParallel(0)` failed to process events already scheduled at time 0 due
  strict `< maxTime` loop guard.
  - Reproducer unit test:
    `ParallelSchedulerTest.RunParallelProcessesCurrentTimeAtLimit`.
  - Before fix: time-0 callback did not fire.
- Fixes:
  - Added `ProcessScheduler::peekNextWakeTime()` to expose the earliest pending
    wake across EventScheduler, clock domains, and minnows.
  - `ParallelScheduler::runParallel` now:
    - checks `peekNextWakeTime()` before advancing, and
    - runs with `currentTime <= maxTime` so time-0 work is included.
- Added regression tests in
  `unittests/Dialect/Sim/ParallelSchedulerTest.cpp`:
  - `RunParallelDoesNotAdvanceBeyondLimit`
  - `RunParallelProcessesCurrentTimeAtLimit`
- Verification:
  - `ninja -C build_test CIRCTSimTests`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='ParallelSchedulerTest.*'`
  - `build_test/unittests/Dialect/Sim/CIRCTSimTests --gtest_filter='EventScheduler.*:ProcessScheduler.*:ProcessSchedulerIntegration.*:ParallelSchedulerTest.*'`
  - All selected tests passed.

## 2026-02-26
- Additional audit finding (runtime UB risk in diagnostics/tracing):
  several `llvm::StringRef` variables were initialized from ternaries that mixed
  `std::string` and string literals, e.g. `cond ? mapIt->second : "?"`.
  - In this shape the conditional expression materializes a temporary
    `std::string` in the literal branch; binding `StringRef` to it can dangle.
  - Clang warning seen during rebuild before fix:
    `object backing the pointer will be destroyed at the end of the full-expression`.
- Fix: replaced those ternary initializers with stable two-step initialization:
  initialize `StringRef` to a literal, then overwrite from map lookup when found.
- Files touched:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
  - `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
  - `tools/circt-sim/LLHDProcessInterpreterMemory.cpp`
  - `tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp`
- Verification:
  - `ninja -C build_test circt-sim`
  - `rg -n "dangling-gsl|object backing the pointer will be destroyed" /tmp/circt_sim_rebuild_after_stringref_fix.log`
    -> no matches
  - `build_clang_test/bin/llvm-lit -sv build_test/test/Tools/circt-sim/max-time-no-false-hit-on-idle.mlir build_test/test/Tools/circt-sim/max-time-inclusive-current-time.mlir build_test/test/Tools/circt-sim/interface-pullup-distinct-driver-sensitivity.sv`
    -> 3/3 passed

## 2026-02-26
- Additional audit finding in `func.call_indirect` site-cache interpreted path:
  `interpretFuncBody(...)` return value was ignored in `ci_cache_interpreted`.
  - Behavioral consequence: on callee failure, no result overwrite occurred,
    so the call op could reuse a stale result value from a previous iteration.
  - Deterministic reproducer built: first iteration succeeds, second iteration
    fails in callee (`llhd.drv` to unknown ref), same call site executes twice;
    before fix observed `sum=2` (stale), expected `sum=1`.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`
  - In `ci_cache_interpreted`, capture `LogicalResult` from
    `interpretFuncBody`, handle `failed()` by propagating suspension
    (`waiting`) or writing zero results, and skip result assignment when
    suspended.
- Added regression:
  - `test/Tools/circt-sim/call-indirect-direct-dispatch-cache-failure-result.mlir`
  - Checks `sum=1` and `CHECK-NOT: sum=2`.
- Verification:
  - `ninja -C build_test circt-sim`
  - `build_test/bin/circt-sim test/Tools/circt-sim/call-indirect-direct-dispatch-cache-failure-result.mlir`
    -> `sum=1`
  - `build_test/bin/circt-sim test/Tools/circt-sim/call-indirect-direct-dispatch-cache.mlir`
    -> `sum=85`
  - `build_test/bin/circt-sim test/Tools/circt-sim/vtable-dispatch-internal-failure.mlir`
    -> expected warnings + prints observed
  - `ninja -C build_test check-circt-tools-circt-sim`
    -> failed for unrelated pre-existing workspace issues in unittests
       (`unittests/Support/CIRCTSupportTests` link error and
       `unittests/Tools/circt-sim/LLHDProcessInterpreterTest.cpp` compile error).
- Follow-up audit finding in `ci_main_interpreted` recursion bookkeeping:
  failure paths decremented recursion depth twice (once unconditionally after
  return from `interpretFuncBody`, and again in failure/suspension branches).
  - Risk: undercounted recursion depth for failing virtual calls.
- Fix:
  - removed duplicate `decrementRecursionDepthEntry(...)` calls in the
    `failed(funcResult)` branches of
    `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`.
- Spot verification after fix:
  - `ninja -C build_test circt-sim`
  - reran:
    - `test/Tools/circt-sim/call-indirect-direct-dispatch-cache-failure-result.mlir`
    - `test/Tools/circt-sim/call-indirect-direct-dispatch-cache.mlir`
    - `test/Tools/circt-sim/vtable-dispatch-internal-failure.mlir`
  - all produced expected outputs.

## 2026-02-26
- Additional audit finding in compiled->interpreter trampoline dispatch:
  `dispatchTrampoline` ignored `interpretFuncBody` failure and still unpacked
  return slots from the trampoline stack buffer.
  - Root cause: `rets` buffer is stack-allocated in generated trampoline code
    and uninitialized; on failure, no values are written, so compiled callers
    receive garbage/stale data.
  - Deterministic reproducer (AOT + demoted UVM function):
    first call succeeds, second call fails via `llhd.drv` to unknown ref.
    Before fix observed `r0=1 r1=20437424 sum=20437425`.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp` in `dispatchTrampoline`:
    - zero-initialize `rets` upfront (`std::fill_n`),
    - capture `LogicalResult` from `interpretFuncBody`,
    - return early on failure/suspension (keep zeroed returns).
- Added regression:
  - `test/Tools/circt-sim/aot-trampoline-failure-zero-result.mlir`
  - validates compiled run prints `r0=1 r1=0 sum=1`.
- Verification:
  - `ninja -C build_test circt-sim`
  - `env CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-trampoline-failure-zero-result.mlir -o /tmp/aot-trampoline-failure-zero-result.so`
  - `env CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim test/Tools/circt-sim/aot-trampoline-failure-zero-result.mlir --compiled=/tmp/aot-trampoline-failure-zero-result.so`
    -> `r0=1 r1=0 sum=1`
  - Spot non-regression:
    - `env CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-entry-table-trampoline-counter.mlir -o /tmp/aot-entry-table-trampoline-counter.so`
    - `env CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim test/Tools/circt-sim/aot-entry-table-trampoline-counter.mlir --compiled=/tmp/aot-entry-table-trampoline-counter.so --aot-stats`
      -> expected trampoline counters + output observed.

- Additional audit finding in trampoline symbol mapping:
  compiled process callback dispatch could crash on `llvm.func` trampoline
  targets that have no native fallback pointer.
  - Reproducer: `test/Tools/circt-sim/aot-process-indirect-cast-dispatch.mlir`
    in compiled mode aborted with:
    `FATAL: trampoline dispatch for func_id=0 (set_true) — FuncOp not found in module`.
  - Root cause: trampoline metadata kept only `func.func` mappings plus native
    fallbacks. `llvm.func` symbols with bodies (but no native pointer) were not
    tracked, and `dispatchTrampoline` treated them as fatal missing symbols.
- Fix:
  - Added `trampolineLLVMFuncOps` mapping in
    `tools/circt-sim/LLHDProcessInterpreter.h/.cpp`.
  - `loadCompiledFunctions` now records `llvm.func` trampoline symbols for
    interpreter fallback even when no native pointer exists.
  - `dispatchTrampoline` now dispatches to `interpretLLVMFuncBody` for those
    `llvm.func` trampolines instead of aborting.
- Regression update:
  - Extended `test/Tools/circt-sim/aot-process-indirect-cast-dispatch.mlir`
    with runtime check:
    - no fatal trampoline-dispatch crash
    - compiled run prints `a=1`.
- Verification:
  - `ninja -C build_test circt-sim`
  - `build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-process-indirect-cast-dispatch.mlir -o /tmp/aot-process-indirect-cast-dispatch.so`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-process-indirect-cast-dispatch.mlir --compiled=/tmp/aot-process-indirect-cast-dispatch.so`
  - FileCheck validation:
    - `--check-prefix=COMPILE` and `--check-prefix=RUNTIME` for the updated test.
  - Spot non-regression:
    - `aot-trampoline-failure-zero-result.mlir` (COMPILE+COMPILED checks)
    - `aot-entry-table-trampoline-counter.mlir` (COMPILE+COMPILED checks)
    - `call-indirect-direct-dispatch-cache-failure-result.mlir` (CHECK).

- Additional audit finding in call_indirect compiled/native result mapping:
  result assignment used call-site result count to index the produced-result
  vector in several paths. When a call site requested more results than the
  resolved callee produced (possible via unrealized function-pointer casts),
  this caused out-of-bounds access.
  - Deterministic reproducer (before fix):
    - compiled run of `func.call_indirect` cast to `(i32) -> (i32, i32)` with
      resolved callee `@ret1 : (i32) -> i32` aborts with:
      `SmallVector::operator[] Assertion 'idx < size()' failed`.
- Fix:
  - Added a single bounded helper in
    `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp` to assign
    call_indirect results with:
    - min-size copy of produced values
    - zero-fill of any extra call-site results.
  - Replaced all call_indirect result assignment sites (native and interpreted
    paths) to use the helper.
- Added regression:
  - `test/Tools/circt-sim/aot-call-indirect-result-arity-mismatch.mlir`
  - checks:
    - no assertion in compiled mode
    - output `r0=5 r1=0`.
- Verification:
  - `ninja -C build_test circt-sim`
  - `circt-sim-compile` + `circt-sim --compiled` with FileCheck for:
    - `aot-call-indirect-result-arity-mismatch.mlir` (COMPILE+RUNTIME)
  - Spot non-regression:
    - `call-indirect-direct-dispatch-cache-failure-result.mlir` (CHECK)
    - `aot-process-indirect-cast-dispatch.mlir` (RUNTIME)
    - `aot-entry-table-trampoline-counter.mlir` (COMPILE+COMPILED).
