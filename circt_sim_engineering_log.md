# CIRCT Sim Engineering Log

## 2026-02-27
- Continued coverage audit: found and fixed cross/`iff` sampling over-gating.
  - Reproducer shape:
    - `cp1` has `iff (en1)`, `x12` crosses `cp1`, and `x23` crosses unrelated
      `cp2/cp3`.
    - With `en1=0`, `x23` should still sample.
  - Bug:
    - MooreToCore lowered cross sampling behind a single conjunction of *all*
      coverpoint `iff` conditions, so one unrelated false `iff` suppressed all
      crosses.
  - Fix:
    - Added runtime API `__moore_cross_sample_masked(cg, cp_values, cp_valid_mask, num_values)`.
    - MooreToCore now lowers `CovergroupSampleOp` to pass per-coverpoint
      validity mask when any `iff` guards are present.
    - Runtime cross sampling now skips only crosses that reference invalid
      coverpoints; unrelated crosses still sample.
    - Interpreter gained dispatch for `__moore_cross_sample_masked`.
  - Regression added:
    - `test/Tools/circt-sim/coverage-cross-iff-per-cross-mask.sv`
      checks:
      - `x12_nonzero=0` (guarded coverpoint in cross)
      - `x23_nonzero=1` (unrelated cross still sampled)
- Audit tooling gap found in the new parity test:
  - `coverage-dispatch-parity.test` used `printf "%s"` in a `RUN:` line, which
    conflicts with lit `%s` substitution and can false-pass.
  - Fixed by removing `%`-format usage in favor of `cat <<<`.
- Post-fix validation:
  - Rebuilt `circt-verilog` and `circt-sim` with the new runtime/lowering path.
  - Focused coverage regression set passed, including the new per-cross-mask
    test, parity test, and unhandled-dispatch diagnostic test.
  - Full `test/Tools/circt-sim/{coverage*,coverpoint*,cross*}.sv` sweep passed
    (excluding explicit `XFAIL` tests).

## 2026-02-27
- Reproduced and fixed interpreted-mode coverage regressions where covergroup
  option setters were emitted by MooreToCore but silently ignored in
  `LLHDProcessInterpreter` (external-call fallback returns success+X for
  unhandled symbols).
- Added missing interpreter dispatch handlers for:
  - `__moore_covergroup_set_weight`
  - `__moore_covergroup_set_goal`
  - `__moore_covergroup_set_per_instance`
  - `__moore_covergroup_set_at_least`
  - `__moore_covergroup_set_comment`
  - `__moore_coverpoint_set_weight`
  - `__moore_coverpoint_set_goal`
  - `__moore_coverpoint_set_at_least`
  - `__moore_coverpoint_set_comment`
  - `__moore_coverpoint_set_auto_bin_max`
- Root-cause surprise: `__moore_cross_add_named_bin` filter decoding used host
  `sizeof/offsetof(MooreCrossBinsofFilter)` while interpreter GEP address math
  is packed (no ABI padding). This mis-decoded filter arrays and could produce
  empty ignore-bin filters that matched everything.
- Fix: decode cross filters using packed struct layout offsets/stride for
  `{i32, ptr, i32, ptr, i32, ptr, i32, i1}`.
- Added/updated regression coverage tests and removed stale `XFAIL` markers
  for now-passing coverpoint/cross method tests.
- Added source-level parity regression
  `test/Tools/circt-sim/coverage-dispatch-parity.test` to guard against future
  missing coverage/cross dispatch handlers:
  - mechanically compares emitted `__moore_cover*`/`__moore_cross*` symbols in
    MooreToCore against `calleeName == ...` handlers in
    `LLHDProcessInterpreter`.
  - currently reports no missing handlers.

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

- Additional audit finding in call_indirect compiled/native argument mapping:
  native dispatch paths read arguments using resolved callee arity (`numArgs`)
  without checking provided call-site argument count.
  - Deterministic reproducer (before fix):
    - resolved callee `@add2 : (i32, i32) -> i32`
    - call site cast to `(i32) -> i32`
    - compiled run aborted with:
      `SmallVector::operator[] Assertion 'idx < size()' failed`.
- Fix:
  - Added bounded native-arg packing helper in
    `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp` that zero-fills
    missing arguments.
  - Replaced all native `call_indirect` argument packing loops to use helper.
- Added regression:
  - `test/Tools/circt-sim/aot-call-indirect-arg-arity-mismatch.mlir`
  - checks:
    - no assertion in compiled mode
    - output `r=5` (missing second arg is zero-filled).
- Verification:
  - `ninja -C build_test circt-sim`
  - `circt-sim-compile` + `circt-sim --compiled` with FileCheck for:
    - `aot-call-indirect-arg-arity-mismatch.mlir` (COMPILE+RUNTIME)
    - `aot-call-indirect-result-arity-mismatch.mlir` (COMPILE+RUNTIME)
  - Spot non-regression:
    - `call-indirect-direct-dispatch-cache-failure-result.mlir` (CHECK)
    - `aot-process-indirect-cast-dispatch.mlir` (RUNTIME).

- Additional audit finding in interpreted function argument binding:
  missing function arguments were not initialized at function entry, so
  block arguments could retain stale values from prior invocations.
  - Deterministic reproducer (before fix):
    - warm call to `@sum2_i65(%a, %b)` sets `%b=7`
    - later mismatched `call_indirect` cast invokes same callee with only `%a=5`
    - interpreted fallback produced `r=12` (stale `%b=7`) instead of zero-filled
      missing arg behavior (`r=5`).
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
    - `interpretFuncBody`: initialize missing entry block args to zero.
    - `interpretLLVMFuncBody`: initialize missing entry block args to zero.
  - This prevents stale value-map reuse for omitted args in mismatched dynamic
    call shapes.
- Added regression:
  - `test/Tools/circt-sim/aot-call-indirect-interp-missing-arg-stale.mlir`
  - checks:
    - no stale prior-arg result (`RUNTIME-NOT: r=12`)
    - expected output `RUNTIME: r=5`.
- Verification:
  - `ninja -C build_test circt-sim`
  - `circt-sim-compile` + `circt-sim --compiled` with FileCheck for:
    - `aot-call-indirect-interp-missing-arg-stale.mlir` (COMPILE+RUNTIME)
    - `aot-call-indirect-arg-arity-mismatch.mlir` (COMPILE+RUNTIME)
    - `aot-call-indirect-result-arity-mismatch.mlir` (COMPILE+RUNTIME)
  - Spot non-regression:
    - `call-indirect-direct-dispatch-cache-failure-result.mlir` (CHECK).

- Additional audit finding in trampoline dispatch for external `llvm.func` symbols:
  when a compiled trampoline targeted an external `llvm.func` with no native
  fallback symbol, `dispatchTrampoline` attempted `interpretLLVMFuncBody` on an
  external declaration and crashed.
  - Deterministic reproducer (before fix):
    - compiled run of external trampoline target `@missing_fn(i64,i64) -> !llvm.struct<(i64,i64)>`
      aborted with stack dump from `interpretLLVMFuncBody`.
  - Related surprise found during repro:
    - native trampoline fallback path also accepted unsupported ABI shapes
      (`numArgs > 8` or `numRets > 1`), which could leak garbage return slots.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp` (`dispatchTrampoline`):
    - zero-initialize return slots at function entry for all paths,
    - guard native fallback to supported ABI subset (`args <= 8`, `rets <= 1`),
    - for external `llvm.func` trampolines without compatible native fallback,
      return zeroed outputs with a warning instead of trying to interpret a
      declaration.
- Added regression:
  - `test/Tools/circt-sim/aot-trampoline-external-llvm-no-native-fallback.mlir`
  - checks:
    - no crash/stack dump,
    - warning about missing native fallback,
    - deterministic output `r=0`.
- Verification:
  - `build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-trampoline-external-llvm-no-native-fallback.mlir -o /tmp/aot-trampoline-external-llvm-no-native-fallback.so`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-trampoline-external-llvm-no-native-fallback.mlir --compiled=/tmp/aot-trampoline-external-llvm-no-native-fallback.so`
  - FileCheck validation for `COMPILE` + `RUNTIME` prefixes.

- Additional audit finding in trampoline native fallback ABI handling:
  external `llvm.func` trampolines were native-dispatched via hard-coded
  `uint64_t` function-pointer signatures without validating argument/result
  types.
  - Deterministic reproducer (before fix):
    - compiled trampoline target `@sqrt(f64) -> f64` produced garbage bits
      (`bits=140608407856656`) instead of a valid floating-point result path.
  - Root cause:
    - `dispatchTrampoline` only checked slot counts (`<=8 args`, `<=1 ret`) and
      did not reject non-integer/non-pointer signatures.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp` (`dispatchTrampoline`):
    - derive trampoline arg/result types before native dispatch,
    - allow native fallback only for scalar integer/pointer/index types with
      widths <= 64,
    - reject incompatible signatures with a warning and fall back to safe
      zeroed returns (plus external-llvm warning path).
- Added regression:
  - `test/Tools/circt-sim/aot-trampoline-native-fallback-f64-incompatible-abi.mlir`
  - checks:
    - warning for unsupported native trampoline ABI,
    - warning for external llvm trampoline fallback,
    - deterministic output `bits=0` (no garbage).
- Verification:
  - `ninja -C build_test circt-sim`
  - `circt-sim-compile` + `circt-sim --compiled` + FileCheck on the new test.
  - Spot non-regression:
    - `aot-trampoline-external-llvm-no-native-fallback.mlir`
    - `aot-process-indirect-cast-dispatch.mlir`
    - `aot-trampoline-failure-zero-result.mlir`.

- Additional audit finding in `func.call` native fast paths:
  native dispatch accepted non-integer signatures (e.g. `f64 -> f64`) and
  invoked compiled entries via hard-coded `uint64_t` ABI casts.
  - Deterministic reproducer (before fix):
    - `tmp_func_call_f64.mlir` with `func.func @id_f64(%x: f64) -> f64`
      returned garbage in compiled mode (`bits=140288544047360`) while
      interpreted mode returned the expected `bits=4612811918334230528`.
  - Root cause:
    - `interpretFuncCall` and `interpretFuncCallCachedPath` only checked
      width/count (`<=64`, `<=8 args`, `<=1 result`), not type class. Float
      args/results passed eligibility and were called with integer ABI.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
    - In both `func.call` native paths, gate dispatch on signature compatibility
      (integer<=64, `index`, or `!llvm.ptr`; still `<=8` args and `<=1` result).
    - Incompatible signatures now cleanly fall back to interpreter execution.
- Added regression:
  - `test/Tools/circt-sim/aot-func-call-native-f64-incompatible-abi.mlir`
  - checks:
    - interpreted output `bits=4612811918334230528`
    - compiled output matches exactly (no garbage from integer-ABI call).
- Verification:
  - `ninja -C build_test circt-sim`
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/aot-func-call-native-f64-incompatible-abi.mlir \
      build_test/test/Tools/circt-sim/aot-trampoline-native-fallback-f64-incompatible-abi.mlir`
  - Direct manual repro check after fix:
    - `build_test/bin/circt-sim tmp_func_call_f64.mlir`
    - `build_test/bin/circt-sim-compile tmp_func_call_f64.mlir -o tmp_func_call_f64.so`
    - `build_test/bin/circt-sim tmp_func_call_f64.mlir --compiled=tmp_func_call_f64.so`

- Additional audit finding in trampoline dispatch with native module init:
  name-only trampolines (no surviving MLIR symbol op) could still be emitted
  for libc/runtime calls (e.g. `memset`), and `dispatchTrampoline` aborted
  before trying native fallback.
  - Deterministic reproducer:
    - `test/Tools/circt-sim/aot-native-module-init-memset.mlir`
    - native-module-init mode crashed with:
      `FATAL: trampoline dispatch for func_id=0 (memset) — function symbol not found in module`.
- Root cause:
  - `loadCompiledFunctions` only attached `trampolineNativeFallback` when a
    trampoline name resolved through `funcLookupCache` as `LLVM::LLVMFuncOp`.
  - For name-only trampolines, no fallback was recorded.
  - `dispatchTrampoline` required a mapped symbol op up front and aborted
    before native fallback could run.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
    - In trampoline mapping, try `dlsym` fallback for name-only trampolines.
    - In `dispatchTrampoline`, zero-init returns first, permit symbol-less
      native-fallback dispatch, and degrade unknown/no-fallback cases to
      warning + zero return instead of abort.
    - Guard arg/result type extraction with `hasLLVMFuncOp` to avoid null-op
      access in symbol-less fallback paths.
- Verification:
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/aot-native-module-init-memset.mlir`
  - plus targeted non-regression:
    - `aot-func-call-native-f64-incompatible-abi.mlir`
    - `aot-trampoline-native-fallback-f64-incompatible-abi.mlir`

- Additional audit finding in UVM sequencer native-state stats:
  `item_map_peak` and `item_map_live` were overcounted because stats used
  `itemToSequencer.size()`, but that map stores both:
  - real item ownership (`itemAddr -> sequencerAddr`), and
  - synthetic process routing entries (`sequencerProcKey(procId) -> sequencerAddr`).
  This inflated peaks in single-item tests (observed `item_map_peak=2/3` where
  expected is `1`).
- Root cause:
  - `recordUvmSequencerItemOwner` updated `uvmSeqItemOwnerPeak` from total
    `itemToSequencer.size()`.
  - profile summary printed `item_map_live` from `itemToSequencer.size()`.
  - synthetic process keys are not item ownership and must not affect these
    metrics.
- Fix:
  - `tools/circt-sim/LLHDProcessInterpreter.h`: add
    `uvmSeqItemOwnerLive` counter.
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`:
    - increment live only on true item-owner insert,
    - decrement live only on true item-owner erase,
    - compute peak from live counter.
  - `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`:
    - report `item_map_live` from `uvmSeqItemOwnerLive`,
    - gate native-state summary on `uvmSeqItemOwnerLive != 0` instead of raw
      map non-empty.
- Verification:
  - `ninja -C build_test circt-sim`
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/finish-item-blocks-until-item-done.mlir \
      build_test/test/Tools/circt-sim/finish-item-multiple-outstanding-item-done.mlir \
      build_test/test/Tools/circt-sim/uvm-oneway-hash-nonpacked-signature.mlir \
      build_test/test/Tools/circt-sim/aot-uvm-random-native-optin.mlir \
      build_test/test/Tools/circt-sim/aot-native-module-init-memset.mlir \
      build_test/test/Tools/circt-sim/aot-func-call-native-f64-incompatible-abi.mlir \
      build_test/test/Tools/circt-sim/aot-trampoline-native-fallback-f64-incompatible-abi.mlir`
  - Result: 7/7 passed.

- Additional audit finding in callback classification:
  one-shot resume tails after a single `llhd.wait` were misclassified as
  `Coroutine` whenever there was no CFG back-edge to the wait block.
  This incorrectly downgraded finite delay waits (e.g. wait-then-terminate)
  and polluted compile-report ExecModel breakdowns.
- Root cause:
  - `AOTProcessCompiler::classifyProcess` Step C required an explicit
    back-edge to `waitBlock` before any further analysis.
  - That rejected valid finite tails where resume paths terminate without
    re-entering wait (including simple `wait delay` -> `sim.terminate`).
- Fix:
  - `tools/circt-sim/AOTProcessCompiler.cpp`
    - Remove the hard back-edge prerequisite.
    - Keep reverse-reachability from `waitBlock`.
    - For resume-reachable blocks that do not reach `waitBlock`, allow them
      only if the induced tail subgraph is acyclic (finite one-shot tail).
    - Reject only when that non-reentering tail contains a cycle.
- Test update and verification:
  - Updated regression harness to preserve both test processes:
    - `test/Tools/circt-sim/callback-classify-pointer-probe-dynamic.mlir`
      now runs with `--skip-passes`.
  - Commands:
    - `ninja -C build_test circt-sim`
    - `python3 llvm/llvm/utils/lit/lit.py -sv \
        build_test/test/Tools/circt-sim/callback-classify-pointer-probe-dynamic.mlir \
        build_test/test/Tools/circt-sim/finish-item-blocks-until-item-done.mlir \
        build_test/test/Tools/circt-sim/finish-item-multiple-outstanding-item-done.mlir \
        build_test/test/Tools/circt-sim/aot-uvm-random-native-optin.mlir \
        build_test/test/Tools/circt-sim/aot-native-module-init-memset.mlir`
  - Result: 5/5 passed.

- Audit sweep note:
  ran `lit` over `build_test/test/Tools/circt-sim` and observed many unrelated
  failures in this dirty worktree (mostly output/expectation drift in AOT
  compile-report/stat lines and some interface-resume tracing expectations).
  Kept this audit scoped to the reproducible callback-classification bug above.

- Additional audit finding in AOT tagged-indirect lowering:
  `LowerTaggedIndirectCalls` decoded tagged function pointers and indexed
  `@__circt_sim_func_entries[fid]` without a bounds check.
  - Deterministic repro (before fix):
    - `/tmp/tagged_indirect_oob.mlir` with constant tagged pointer `0xF0000082`
      and only one valid tagged vtable entry.
    - Interpreted run: `bad=0` with unresolved-call warning.
    - Compiled run: `bad=47` (out-of-bounds entry-table read misdispatch).
- Root cause:
  - `tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp` treated any pointer
    in `[0xF0000000, 0x100000000)` as a valid tagged FuncId and performed
    unchecked GEP/load from the entry table.
- Fix:
  - Add `fid < num_func_entries` guard in lowered IR.
  - Introduce explicit invalid-tag path for both `call` and `invoke` rewrites
    that returns zero (or no-op for void), matching interpreter fallback
    behavior instead of reading out-of-bounds table slots.
- Added regression:
  - `test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir`
  - checks:
    - lowering pass still runs (`LowerTaggedIndirectCalls: lowered 1 indirect calls`)
    - interpreted output `bad=0`
    - compiled output `bad=0`
- Verification:
  - `ninja -C build_test circt-sim-compile circt-sim`
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir \
      build_test/test/Tools/circt-sim/callback-classify-pointer-probe-dynamic.mlir \
      build_test/test/Tools/circt-sim/finish-item-blocks-until-item-done.mlir \
      build_test/test/Tools/circt-sim/finish-item-multiple-outstanding-item-done.mlir \
      build_test/test/Tools/circt-sim/aot-uvm-random-native-optin.mlir \
      build_test/test/Tools/circt-sim/aot-native-module-init-memset.mlir`
  - Result: 6/6 passed.

- Additional audit finding in AOT tagged-indirect lowering (null entry slots):
  the previous tagged-FuncId bounds check still allowed calling a null entry
  pointer when `fid` was in range but `@__circt_sim_func_entries[fid]` was
  null (e.g. unresolved/missing vtable symbol with no generated trampoline).
  - Deterministic repro (before fix):
    - `/tmp/tagged_indirect_null_dynamic.mlir` with vtable entries
      `[add42, missing]` and runtime select of slot1.
    - Interpreted run completed with `dyn=x`.
    - Compiled run crashed inside `circt-sim` after startup (null call target).
- Root cause:
  - `tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp` checked only
    `fid < num_func_entries` before loading `entry_ptr` and issuing the call.
  - No `entry != null` guard existed on the tagged dispatch path.
- Fix:
  - Add `entry_nonnull` checks in both lowered `call` and `invoke` paths.
  - Route null-entry cases to the same safe invalid-tag fallback path
    (zero return / no-op for void), preventing null indirect calls.
- Added regression:
  - `test/Tools/circt-sim/aot-call-indirect-tagged-null-entry-safe.mlir`
  - checks:
    - lowering pass still runs,
    - interpreted behavior (`dyn=x`) remains,
    - compiled behavior is safe (`dyn=0`) with no crash.
- Verification:
  - `ninja -C build_test circt-sim-compile circt-sim`
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/aot-call-indirect-tagged-null-entry-safe.mlir \
      build_test/test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir \
      build_test/test/Tools/circt-sim/callback-classify-pointer-probe-dynamic.mlir \
      build_test/test/Tools/circt-sim/finish-item-blocks-until-item-done.mlir \
      build_test/test/Tools/circt-sim/finish-item-multiple-outstanding-item-done.mlir \
      build_test/test/Tools/circt-sim/aot-uvm-random-native-optin.mlir \
      build_test/test/Tools/circt-sim/aot-native-module-init-memset.mlir`
  - Result: 7/7 passed.

- Additional audit finding in AOT tagged-indirect `invoke` lowering:
  unwind-destination PHI repair used the wrong predecessor block on the tagged
  exceptional edge.
  - Deterministic repro (before fix):
    - New unit test `LowerTaggedIndirectCallsTest.InvokeUnwindPhiUsesInvokeBlocks`
      built from raw LLVM IR with:
      - an indirect `invoke`, and
      - an unwind block PHI (`%u = phi ... [ %tag, %entry ]`).
    - After `runLowerTaggedIndirectCalls`, LLVM verifier failed with:
      `PHI node entries do not match predecessors`.
    - Observed bad PHI incoming blocks: `%tagged_call` and `%direct_call`
      while unwind predecessors are `%tagged_invoke` and `%direct_call`.
- Root cause:
  - `tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp` updated unwind PHIs
    with `Phi.setIncomingBlock(Idx, taggedBB)`.
  - `taggedBB` is only the pre-invoke dispatch block; the actual exceptional
    edge comes from `taggedInvokeBB`.
- Fix:
  - Update unwind PHI rewrite to use `taggedInvokeBB` for the tagged incoming
    edge.
- Added unit test coverage:
  - `unittests/Tools/circt-sim-compile/LowerTaggedIndirectCallsTest.cpp`
  - plus unit-test target wiring:
    - `unittests/Tools/CMakeLists.txt`
    - `unittests/Tools/circt-sim-compile/CMakeLists.txt`
- Verification:
  - `ninja -C build_test CIRCTSimCompileToolTests`
  - `build_test/unittests/Tools/circt-sim-compile/CIRCTSimCompileToolTests`
  - `build_test/unittests/Tools/circt-sim-compile/CIRCTSimCompileToolTests --gtest_filter=LowerTaggedIndirectCallsTest.InvokeUnwindPhiUsesInvokeBlocks`
  - Result: pass.
- Surprise noted during minimization:
  - A mixed-dialect repro (`func.func` calling `llvm.call @indirect_invoke`)
    triggered a separate `circt-sim-compile --emit-llvm` segfault during
    MLIR->LLVM translation (`llvm::ConstantExpr::getBitCast` from
    `ModuleTranslation::convertOneFunction`).
  - This appears independent of `LowerTaggedIndirectCalls` and should be
    investigated separately.

- Additional audit finding in AOT trampoline generation:
  `circt-sim-compile --emit-llvm` could segfault while translating modules
  where a non-compiled `llvm.func` with EH personality metadata is converted
  into an interpreter trampoline.
  - Deterministic repro (before fix):
    - Minimal mixed-dialect input with:
      - `func.func @entry` calling `llvm.call @invoke_wrap`, and
      - `llvm.func @invoke_wrap(...) attributes { personality = @__gxx_personality_v0 }`
        containing an `llvm.invoke` + `landingpad`.
    - Command:
      - `build_test/bin/circt-sim-compile --emit-llvm /tmp/mixed_invoke_direct.mlir -o /tmp/mixed_invoke_direct.ll`
    - Result: SIGSEGV in MLIR->LLVM translation path
      (`llvm::ConstantExpr::getBitCast` via `ModuleTranslation::convertOneFunction`).
- Root cause:
  - `generateTrampolines()` materialized bodies for selected external
    `llvm.func`s but left a stale `personality` function attribute attached.
  - The generated trampoline body has no EH semantics; keeping that attribute
    could trigger translator crashes while materializing function attributes.
- Fix:
  - `tools/circt-sim-compile/circt-sim-compile.cpp`
    - In `generateTrampolines()`, remove `personality` attribute before
      constructing trampoline blocks/bodies.
- Added regression:
  - `test/Tools/circt-sim/aot-trampoline-personality-no-crash.mlir`
  - checks successful compile with expected stats + LLVM IR emission.
- Verification:
  - `ninja -C build_test circt-sim-compile`
  - `build_test/bin/circt-sim-compile --emit-llvm test/Tools/circt-sim/aot-trampoline-personality-no-crash.mlir -o /tmp/t4.ll`
  - `build_test/unittests/Tools/circt-sim-compile/CIRCTSimCompileToolTests --gtest_filter=LowerTaggedIndirectCallsTest.InvokeUnwindPhiUsesInvokeBlocks`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir`
  - `build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir -o /tmp/t1.so`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-call-indirect-tagged-fid-oob-safe.mlir --compiled=/tmp/t1.so`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-call-indirect-tagged-null-entry-safe.mlir`
  - `build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-call-indirect-tagged-null-entry-safe.mlir -o /tmp/t2.so`
  - `build_test/bin/circt-sim test/Tools/circt-sim/aot-call-indirect-tagged-null-entry-safe.mlir --compiled=/tmp/t2.so`
  - Result: all pass; prior segfault no longer reproduces.

- Additional audit finding in trampoline ABI packing:
  wide integer scalars (`iN` where `N > 64`) were silently truncated to a
  single `uint64_t` slot on compiled->interpreted calls and zero-extended on
  return, losing high bits.
  - Deterministic repro (before fix):
    - `/tmp/tramp_i128.mlir`
      - `func.func @entry(%x: i128) -> i128 { %r = llvm.call @ext_i128(%x) ... }`
      - `llvm.func @ext_i128(i128) -> i128`
    - Command:
      - `build_test/bin/circt-sim-compile --emit-llvm /tmp/tramp_i128.mlir -o /tmp/tramp_i128.ll`
    - Observed bad lowering:
      - `__circt_sim_call_interpreted(..., i32 1, ..., i32 1)`
      - `trunc i128 -> i64` before call
      - `zext i64 -> i128` after call
- Root cause:
  - `countTrampolineSlots()` treated all integers as one slot.
  - `emitPackValue()` truncated integers wider than 64 bits.
  - `emitUnpackValue()` reconstructed wide integers from only one slot.
  - Interpreter trampoline helpers mirrored the same one-slot assumption.
- Fix:
  - `tools/circt-sim-compile/circt-sim-compile.cpp`
    - `countTrampolineSlots(iN)` now returns `ceil(N/64)`.
    - `emitPackValue` splits wide integers into little-endian 64-bit chunks.
    - `emitUnpackValue` reconstructs wide integers by zext+shift+or across all
      consumed chunks.
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
    - `unpackTrampolineArg` now consumes `ceil(N/64)` slots for wide ints.
    - `packTrampolineResult` now writes `ceil(N/64)` slots for wide ints.
- Added regressions:
  - `test/Tools/circt-sim/aot-trampoline-i128-multi-slot-packing.mlir`
    - checks emitted LLVM uses two slots and reconstructs via shift/or.
  - `test/Tools/circt-sim/aot-trampoline-f32-packing-no-invalid-bitcast.mlir`
    - checks non-64-bit float trampoline packing avoids invalid bitcasts.
- Verification:
  - `ninja -C build_test circt-sim-compile circt-sim`
  - `build_test/bin/circt-sim-compile --emit-llvm test/Tools/circt-sim/aot-trampoline-i128-multi-slot-packing.mlir -o /tmp/tramp_i128_after2.ll`
  - `build_test/bin/circt-sim-compile --emit-llvm test/Tools/circt-sim/aot-trampoline-f32-packing-no-invalid-bitcast.mlir -o /tmp/tramp_f32_after2.ll`
  - `build_test/unittests/Tools/circt-sim-compile/CIRCTSimCompileToolTests --gtest_filter=LowerTaggedIndirectCallsTest.InvokeUnwindPhiUsesInvokeBlocks`
  - Spot-checks in emitted IR confirm:
    - `__circt_sim_call_interpreted(..., i32 2, ..., i32 2)` for `i128`
    - high-word `lshr/shl` and `or` reconstruction present
    - no `bitcast float -> i64` / `bitcast i64 -> float`
  - Note:
    - `ninja -C build_test check-circt-tools-circt-sim` currently fails in
      unrelated existing build issues (`CIRCTSupportTests` LLHD TypeID link
      failure and `CIRCTSimToolTests` compile mismatch), before lit reaches the
      new regressions.

- Additional audit finding in trampoline return path:
  `dispatchTrampoline` packed non-struct returns with `getUInt64()` directly,
  truncating wide integer returns (`iN`, `N > 64`) even after multi-slot
  compiler packing was fixed.
  - Deterministic repro (before fix):
    - Added runtime regression
      `test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir`
      where:
      - `@caller` is compiled and native-dispatched,
      - `@caller` invokes demoted `@"uvm_pkg::uvm_demo::wide_ret"` via
        trampoline (`Trampoline calls: 1`),
      - callee returns `i128` with bit 100 set.
    - Results before fix:
      - interpreted: `hi=-1`
      - compiled: `hi=0`
      - proving upper bits were dropped on trampoline return.
- Root cause:
  - In `tools/circt-sim/LLHDProcessInterpreter.cpp` return marshaling used:
    - `packTrampolineResult(...)` only for `LLVMStructType`,
    - raw `getUInt64()` for all other return types.
  - `i128` is non-struct, so only one 64-bit slot was written.
- Fix:
  - Always use `packTrampolineResult(retTy, ...)` when return type is known.
  - Keep one-slot fallback only when type metadata is unavailable.
- Added regression:
  - `test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir`
- Verification:
  - `ninja -C build_test circt-sim`
  - `CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir`
  - `CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim-compile test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir -o /tmp/aot_i128_ret_fixed.so`
  - `CIRCT_AOT_INTERCEPT_ALL_UVM=1 build_test/bin/circt-sim test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir --compiled=/tmp/aot_i128_ret_fixed.so --aot-stats`
  - `python3 llvm/llvm/utils/lit/lit.py -sv \
      build_test/test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir \
      build_test/test/Tools/circt-sim/aot-trampoline-i128-multi-slot-packing.mlir \
      build_test/test/Tools/circt-sim/aot-trampoline-f32-packing-no-invalid-bitcast.mlir \
      build_test/test/Tools/circt-sim/aot-trampoline-personality-no-crash.mlir`
  - Result: targeted tests pass; compiled runtime now matches interpreted
    (`hi=-1`) with one native call and one trampoline call.

- Additional trampoline packing audit (bf16 struct fields):
  - Found and fixed another deterministic ABI mismatch:
    `getScalarOrStructBitWidth` in
    `tools/circt-sim/LLHDProcessInterpreter.cpp` hardcoded float widths for
    `f16/f32/f64` and defaulted other floats to 64 bits.
  - Impact:
    - Struct packing/unpacking in interpreted trampoline dispatch could misplace
      fields for `bf16` (and any non-f16/f32/f64 float), causing compiled vs
      interpreted divergence.
  - Repro and regression added:
    - `test/Tools/circt-sim/aot-trampoline-bf16-struct-runtime.mlir`
    - Before fix:
      - interpreted: `ok=-1`
      - compiled (via trampoline): `ok=0`
    - After fix:
      - interpreted and compiled both: `ok=-1`
  - Fix:
    - Use `floatTy.getWidth()` directly in trampoline bit-width helper.
  - Verification:
    - `ninja -C build_test circt-sim`
    - `python3 llvm/llvm/utils/lit/lit.py -sv \
        build_test/test/Tools/circt-sim/aot-trampoline-bf16-struct-runtime.mlir \
        build_test/test/Tools/circt-sim/aot-trampoline-i128-return-runtime.mlir \
        build_test/test/Tools/circt-sim/aot-trampoline-i128-multi-slot-packing.mlir \
        build_test/test/Tools/circt-sim/aot-trampoline-f32-packing-no-invalid-bitcast.mlir \
        build_test/test/Tools/circt-sim/aot-trampoline-personality-no-crash.mlir`
    - Result: all targeted tests pass.

- Differential stress audit run (circt-sim vs xrun, 1000-cycle compact cases):
  - Added utility:
    - `utils/run_compact_sv_xrun_diff.py`
    - Generates deterministic compact/intricate SV cases with mixed
      `always_comb`/`always_ff`, `casez`, `priority case`, rotation/shift, and
      function calls; runs both simulators and compares per-cycle traces + final
      signatures.
  - Tooling fixes while developing the runner:
    - avoid parser false positives from `[circt-sim]` diagnostics interleaved in
      `$display` output,
    - switch `circt-verilog` invocation to `-o` file output to avoid huge
      stdout buffering.
  - Audit outcome:
    - Ran 19 generated cases total at 1000 cycles each across multiple seed
      batches (3 + 3 + 3 + 10), no semantic mismatches detected.
    - Additional targeted 4-state/X-propagation stress case also matched final
      signature between `circt-sim` and `xrun`.
  - Notes:
    - During this pass, shared `build_test` binaries were intermittently
      unstable due concurrent rebuild activity; differential runs used a stable
      tool snapshot to keep audit execution deterministic.

- Additional trampoline audit finding (unsupported ABI declarations):
  - Deterministic repro discovered while auditing:
    - A demoted function with unsupported trampoline ABI type (e.g.
      `vector<2xi32>` return) could be left without a generated trampoline.
    - `circt-sim-compile` then still produced a `.so`, but compiled-mode load
      failed later with unresolved symbol at runtime:
      `undefined symbol: uvm_pkg::uvm_demo::vec_ret`.
  - Root cause:
    - `generateTrampolines()` silently skipped unsupported external signatures,
      allowing broken artifacts to be emitted.
  - Fix:
    - `tools/circt-sim-compile/circt-sim-compile.cpp`
      - `generateTrampolines()` now returns `FailureOr<...>` and emits
        explicit diagnostics for unsupported external trampoline ABI shapes
        (vararg or unflattenable parameter/return types).
      - AOT compilation now fails fast when such declarations are present,
        instead of deferring failure to runtime symbol lookup.
  - Regression added:
    - `test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir`
      - checks compile-time diagnostic for unsupported `vector<2xi32>` return.
  - Verification:
    - `ninja -C build_test circt-sim-compile`
    - `python3 llvm/llvm/utils/lit/lit.py -sv \
        build_test/test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir`
    - `python3 llvm/llvm/utils/lit/lit.py -sv --filter='aot-trampoline-' \
        build_test/test/Tools/circt-sim`
    - Result: new diagnostic regression passes; full trampoline cluster passes
      (including existing vtable trampoline coverage).

- Differential execution audit (additional batch):
  - Ran:
    - `python3 utils/run_compact_sv_xrun_diff.py --cases 20 --cycles 1000 \
       --seed 20260226 --out-dir /tmp/circt-compact-sv-diff-audit-20260226 \
       --circt-verilog <stable-snapshot> --circt-sim <stable-snapshot> --xrun xrun`
  - Result:
    - `20/20` matched, `0` mismatches.

- Ongoing audit sweep (no new correctness regressions found in this pass):
  - Trampoline ABI fuzz (scalar):
    - 120 randomized identity-call cases spanning `i1..i256` sparse widths,
      `bf16/f16/f32/f64`, and `!llvm.ptr`.
    - For each case: interpreted run + compiled run compared via in-design
      bit-exact checks (`ok=-1` expected).
    - Result: `120/120` pass.
  - Trampoline ABI fuzz (aggregate):
    - 70 randomized nested `!llvm.struct` / `!llvm.array` cases with mixed
      integer, float, and pointer leaves.
    - Each case constructs aggregate via `llvm.insertvalue`, round-trips through
      demoted identity callee, recursively compares all leaves.
    - Result: `70/70` pass.
  - Differential simulation vs xrun:
    - `utils/run_compact_sv_xrun_diff.py --cases 40 --cycles 1000`
      with stable snapshot binaries.
    - Result: `40/40` matched, `0` mismatches.
  - Note:
    - One attempted differential run using `build_test/bin` aborted due
      concurrent binary replacement (`Permission denied` on `circt-sim`);
      rerun with snapshot binaries completed cleanly.

- New audit finding: dynamic 4-state `casez`/`casex` matching was lowered
  incorrectly in MooreToCore.
  - Differential fuzzing (`circt-sim` vs `xrun`) found deterministic mismatches
    on `casex` with unknown (`x/z`) select bits.
  - Minimized SV repro showed `xrun` produced:
    - `OBS sel=11z y=00000111`
    - `OBS sel=zz1 y=00000111`
    while `circt-sim` produced:
    - `OBS sel=111 y=11101111`
    - `OBS sel=111 y=11101111`
  - Root cause identified in
    `lib/Conversion/MooreToCore/MooreToCore.cpp`
    (`CaseXZEqOpConversion`):
    - only constant operand unknown bits were used to build ignore masks,
    - dynamic unknown bits from 4-state operands were dropped by comparing only
      extracted value lanes.
  - Fix:
    - For 4-state operands (`!hw.struct<value, unknown>`), build runtime ignore
      masks from unknown lanes:
      - `casex`: ignore `lhs.unknown | rhs.unknown`
      - `casez`: ignore `lhs.z | rhs.z` where `z = value & unknown`
    - `casez` now also compares masked unknown lanes so `X` remains
      significant on non-ignored bits.
  - Regression added:
    - `test/Conversion/MooreToCore/case-xz-dynamic-fourstate.mlir`
      checks generated core IR contains dynamic unknown-aware masking for both
      `casez_eq` and `casexz_eq`.
  - Verification:
    - `ninja -C build_test circt-opt`
    - `python3 llvm/llvm/utils/lit/lit.py -sv \
        build_test/test/Conversion/MooreToCore/case-xz-dynamic-fourstate.mlir`
    - Result: regression passes.
  - Note:
    - End-to-end `circt-verilog` rebuild was blocked by an unrelated concurrent
      `ImportVerilog.cpp` compile crash in the dirty worktree; runtime
      confirmation with newly linked `circt-verilog` is pending that unblock.

- New audit finding: wildcard equality (`==?` / `!=?`) 4-state lowering had two
  independent correctness bugs in `WildcardEqOpConversion`.
  - Repro 1 (deterministic):
    - `a=2'b1x; a ==? 2'b1z` expected `1`, got `X`.
  - Root cause 1:
    - LHS unknown taint used full `lhs.unknown`, not just bits that survive RHS
      wildcard masking.
  - Repro 2 (deterministic):
    - `a=5'b1z11z; b=5'bx000z; a ==? b` expected `0`, got `X`.
  - Root cause 2:
    - Known mismatches were not given precedence; result was forced to `X` when
      any compared LHS unknown bit existed, even when known bits already proved
      inequality.
  - Fix in `lib/Conversion/MooreToCore/MooreToCore.cpp`:
    - Compute `lhsRelevantUnk = lhsUnk & rhsMask`.
    - Compute `knownMask = rhsMask & ~lhsRelevantUnk`.
    - Compare only known bits for mismatch/eq (`knownEq`).
    - Set result unknown to `lhsHasUnk & knownEq`.
    - Set result value:
      - `==?`: `knownEq & !lhsHasUnk`
      - `!=?`: `!knownEq`
  - New regressions:
    - `test/Conversion/MooreToCore/wildcard-eq-dynamic-fourstate.mlir`
      - checks lowering uses masked unknowns and known-mismatch gating.
    - `test/Tools/circt-sim/wildcard-eq-rhs-mask-lhs-unknown.sv`
      - checks masked unknown case and known-mismatch-dominates case.
  - Verification:
    - `python3 llvm/llvm/utils/lit/lit.py -sv \
        build_test/test/Conversion/MooreToCore/wildcard-eq-dynamic-fourstate.mlir \
        build_test/test/Tools/circt-sim/wildcard-eq-rhs-mask-lhs-unknown.sv \
        build_test/test/Tools/circt-sim/case-xz-dynamic-fourstate.sv`
      - Result: 3/3 passed.
    - Differential fuzz vs xrun (`--mode interpret`) after fix:
      - targeted wildcard eq/ne fuzz: `240/240` matched.
      - extended wildcard eq/ne fuzz: `1000/1000` matched.
      - nearby logical eq/ne fuzz: `240/240` matched.
      - nearby case eq/ne fuzz: `240/240` matched.
  - Note:
    - Running `test/Conversion/MooreToCore/basic.mlir` in this dirty tree still
      hits an unrelated pre-existing CHECK mismatch outside this change.

- New audit finding: integer format handling had two additional correctness
  issues (ImportVerilog decimal zero-pad parsing and interpreter 4-state
  formatting).
  - Issue 1 (ImportVerilog `%0wd` parsing):
    - `"%011d"` was lowered as `pad space width 11` instead of zero-padded.
    - Root cause: fallback `zeroPad` detection was computed but not used when
      selecting decimal padding.
    - Fix:
      - `lib/Conversion/ImportVerilog/FormatStrings.cpp` now uses the computed
        `zeroPad` flag in decimal padding selection.
      - fixed stale `emitInteger(...)` call site after signature extension.
    - Regressions:
      - `test/Conversion/ImportVerilog/format-decimal-zero-pad-width.sv`
      - `test/Tools/circt-sim/format-decimal-zero-pad-width.sv`
    - Validation:
      - target checks pass via `circt-verilog|circt-sim + FileCheck`.
      - differential integer formatting script:
        - `PASS: 320 int-format cases matched xrun`.

  - Issue 2 (4-state `%b/%o/%h/%d` formatting in interpreter):
    - Symptom:
      - unknown lanes (`x/z`) were dropped and printed as known digits.
    - Root cause:
      - `FormatIntOpConversion` extracted only struct field `value` from
        lowered 4-state values (`!hw.struct<value, unknown>`), discarding
        `unknown`.
    - Fixes:
      - `lib/Conversion/MooreToCore/MooreToCore.cpp`:
        - preserve 4-state payload by bitcasting `{value,unknown}` struct to
          packed integer and attach new `fourStateWidth` attr on
          `sim.fmt.bin/oct/hex/dec`.
      - `include/circt/Dialect/Sim/SimOps.td`:
        - added optional `fourStateWidth` attr to integer format ops.
      - `lib/Dialect/Sim/SimOps.cpp`:
        - disable constant folding for integer format ops when
          `fourStateWidth` is present.
      - `tools/circt-sim/LLHDProcessInterpreter.cpp`:
        - decode packed 4-state payloads using `fourStateWidth`.
        - implement 4-state digit mapping for binary/octal/hex/decimal,
          including `x/z/X/Z` handling and width/padding behavior.
    - New regression:
      - `test/Tools/circt-sim/format-fourstate-int.sv`
        - reproduces and checks `x/z` behavior for `%b/%o/%h/%d/%04d`.
    - Validation:
      - regression passes.
      - targeted formatting regressions still pass.
      - randomized 4-state differential vs xrun:
        - `218/220` matched.
        - residual mismatches (2 cases) are in compacting of leading `Z` for
          `%h/%o` with specific mixed-unknown patterns.

- Follow-up audit: completed 4-state grouped-radix compaction parity with xrun.
  - Starting point:
    - prior residual mismatch count was `2/220` on randomized 4-state integer
      formatting differential.
  - New deterministic mismatches found by rerunning
    `/tmp/fuzz_fmt_fourstate_diff.py`:
    - `4/220` first pass (examples: `%0o` `0Z` preservation, `%0h/%0o`
      uppercase `X` run over-collapsing, `%03o` mixed `Z` compaction).
    - after first patch: `3/220` remained (`%1h/%1o` preserving leading zero
      before unknown in partial-width groups).
  - Key realizations from direct xrun probes:
    - uppercase `X` runs are not compacted (`XX` stays `XX` even for `%1h`).
    - uppercase/lowercase `Z` runs compact only as needed down to one leading
      `Z/z` while keeping suffix digits.
    - lowercase all-unknown `x/z` runs compact with width (`xxx` -> `x` for
      `%1o`, `xx` for `%2o`, etc.).
    - a leading zero before unknown in a partial top group is structural and
      must be preserved (`0Z`, `0XX` cases).
  - Fix:
    - rewrote grouped-radix compaction in
      `tools/circt-sim/LLHDProcessInterpreter.cpp` (`formatFourStateGrouped`):
      - width-aware leading-zero stripping with partial-group/unknown
        preservation.
      - selective unknown-run collapse matching xrun behavior:
        - collapse leading `Z/z` and lowercase `x` runs toward requested width,
        - never collapse uppercase `X` runs.
      - keep all-zero rendering consistent for `%0` and explicit widths.
  - New/expanded regression:
    - `test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv`
      - added checks for:
        - `%0o` `0Z` preservation,
        - `%0o/%0h` uppercase `XX` behavior,
        - `%03o` `ZX3` compaction,
        - `%1h` `0X`, `%1o/%0o` `0XX`,
        - lowercase all-`x` `%0o/%1o` compaction.
  - Validation:
    - `ninja -C build_test circt-sim`
    - `llvm/build/bin/llvm-lit -sv \
        build_test/test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv \
        build_test/test/Tools/circt-sim/format-radix-zero-flag-minwidth.sv \
        build_test/test/Tools/circt-sim/format-fourstate-int.sv \
        build_test/test/Tools/circt-sim/format-decimal-zero-pad-width.sv`
      - Result: `4/4` passed.
    - randomized formatter differential:
      - `python3 /tmp/fuzz_fmt_fourstate_diff.py`
      - Result: `PASS: 220 four-state format cases matched xrun`.
    - compact intricate SV differential (non-format stress):
      - `PYTHONUNBUFFERED=1 python3 utils/run_compact_sv_xrun_diff.py --cases 8 --cycles 1000 --stop-on-first-mismatch --out-dir out/compact_sv_diff_audit_postfix_20260226`
      - Result: `total=8 mismatches=0`.

- Audit continuation (2026-02-27): case/casez four-state differential + compact stress.
  - Re-verified recently added regressions:
    - `test/Tools/circt-sim/casez-x-to-z-retrigger.sv`
    - `test/Tools/circt-sim/case-no-default-preserves-initial-x.sv`
    - (Adjusted `case-no-default-preserves-initial-x.sv` to avoid a false failure
      caused by a `3'bxxx` arm matching at time 0.)

  - Compact intricate differential run:
    - `PYTHONUNBUFFERED=1 python3 utils/run_compact_sv_xrun_diff.py --cases 25 --cycles 1000 --seed 20260227 --stop-on-first-mismatch --out-dir out/compact_sv_diff_audit_20260227`
    - Result: `total=25 mismatches=0`.

  - New deterministic bug found and fixed (TDD): ImportVerilog case default elision with X/Z item constants.
    - Reproducer class:
      - `case` statements whose item constants include X/Z and whose raw constant
        values incidentally span all `2^N` combinations.
      - Existing two-state exhaustiveness heuristic incorrectly treated these as
        full two-state coverage and suppressed the explicit `default`, falling
        back to the last case item.
    - Symptom:
      - Example: `sel=3'b000` incorrectly produced a non-default arm value
        (`17`/`d4`) instead of explicit `default` (`56`).
    - Root cause:
      - In `lib/Conversion/ImportVerilog/Statements.cpp`, the heuristic pushed
        all constant case item values into `itemConsts`, including values with
        unknown bits, and then checked raw-value coverage.
    - Fix:
      - Count constants for two-state exhaustiveness only when
        `!defOp.getValue().hasUnknown()`.
    - Regression added:
      - `test/Tools/circt-sim/case-default-preserved-with-xz-items.sv`
        - checks explicit `default` is preserved (`A<56>`, `B<56>`) while a
          matching X/Z item still works (`C<17>`).
    - Validation:
      - `ninja -C build_test circt-verilog circt-sim`
      - `llvm/build/bin/llvm-lit -sv \
          build_test/test/Tools/circt-sim/case-default-preserved-with-xz-items.sv \
          build_test/test/Tools/circt-sim/casez-x-to-z-retrigger.sv \
          build_test/test/Tools/circt-sim/case-no-default-preserves-initial-x.sv`
      - Result: `3/3` passed.

  - Differential follow-up on the bug class:
    - Ran randomized 4-state `case/casez/casex` differential fuzzer against xrun.
    - Initial rerun surfaced one mismatch at `t=0`; reduced to scheduling-race
      noise (uninitialized selector before first delay), not semantic divergence.
    - Hardened fuzzer harness (initialized selector + delayed stimulus start) to
      avoid time-zero ordering artifacts.
    - Post-fix results:
      - `--num 80 --seed 20260227`: `ALL_OK`.
      - `--num 200 --seed 20260301`: `ALL_OK`.

  - Realizations:
    - The two-state exhaustiveness workaround is high-risk around 4-state
      literals; raw-value coverage is insufficient unless unknown-bearing
      constants are excluded.
    - Differential fuzzers need explicit protection against time-zero scheduling
      races, otherwise false mismatches dominate and mask real bugs.

- Coverage regression capture (2026-02-27): implicit covergroup event sampling.
  - Context:
    - User repro showed `covergroup cg @(posedge clk)` with no explicit
      `cg.sample()` produced a coverage report with `0 hits` for all
      coverpoints.
  - Realizations / surprises:
    - Existing `circt-sim` coverage tests mostly validate explicit sampling
      (`cg.sample()`), while the event-driven sampling path had no dedicated
      end-to-end regression in `test/Tools/circt-sim`.
    - The best CI-stable way to lock the bug in place is an expected-fail test
      that asserts non-zero hits from implicit sampling and flips to XPASS once
      fixed.
  - Added regression:
    - `test/Tools/circt-sim/coverage-event-implicit-sample.sv`
      - `XFAIL: *`
      - checks that `cp_addr` and `cp_we` report non-zero hits when sampling is
        driven only by `@(posedge clk)`.

- Coverage code audit (2026-02-27): additional bugs in cross/options paths.
  - Scope:
    - Audited ImportVerilog coverage lowering, MooreToCore coverage conversion,
      runtime cross accounting, and circt-sim runtime-call interception.
  - Key findings:
    - Cross bin accounting in runtime is keyed only by value tuple, not by
      cross index, so multiple crosses with equal arity can interfere.
    - Cross named-bin filters are shallow-copied in runtime (pointer fields),
      while lowering passes stack-allocated filter storage, creating dangling
      filter pointers after init returns.
    - Interpreter path currently drops cross named-bin filters entirely
      (`__moore_cross_add_named_bin(..., nullptr, numFilters)`), making
      filtered named bins behave as unfiltered.
    - Covergroup/coverpoint options parsed into Moore attrs are not propagated
      to runtime in MooreToCore conversion (`goal`, `weight`, `at_least`,
      `auto_bin_max`, `per_instance`, etc. effectively default).
    - Cross sampling is called unconditionally with all values even when
      coverpoint `iff` guards suppress some coverpoint samples.

- Coverage regression additions from audit (2026-02-27).
  - Added XFAIL regressions to lock current failures in place:
    - `test/Tools/circt-sim/coverage-cross-equal-arity-isolation.sv`
      - catches cross-bin map contamination between same-arity crosses.
    - `test/Tools/circt-sim/coverage-cross-named-bin-filters-interpret.sv`
      - catches loss of named cross-bin filters in interpret mode.
    - `test/Tools/circt-sim/coverage-option-at-least-propagation.sv`
      - catches missing runtime propagation of `coverpoint option.at_least`.
  - Notes:
    - All three are `XFAIL: *` to keep CI green until implementation catches up.
    - Local execution is pending availability of `circt-verilog`/`circt-sim`
      binaries in `PATH`.

- Coverage semantic gap repro + fix plan (2026-02-27): method query paths.
  - Reproduced with standalone SV snippets under `circt-sim --mode interpret`:
    - `cg_inst.cp.get_inst_coverage()` and `cg_inst.cp.get_coverage()` printed
      `x` before fix.
    - `cg_inst.xab.get_inst_coverage()` and `cg_inst.xab.get_coverage()`
      printed `x` before fix.
    - Coverage report still showed non-zero hits, confirming sampling worked and
      query wiring was the broken path.
  - Root cause:
    - ImportVerilog handled covergroup methods, but not coverpoint/cross
      coverage query methods, so query expressions lowered to invalid/unknown
      values.
    - LLHD interpreter lacked handlers for coverpoint/cross runtime query
      helpers even when those symbols are emitted.
  - Changes made:
    - `lib/Conversion/ImportVerilog/Expressions.cpp`
      - added lowering for coverpoint methods:
        - `get_coverage` -> `__moore_coverpoint_get_coverage(cg, cp_idx)`
        - `get_inst_coverage` -> `__moore_coverpoint_get_inst_coverage(cg, cp_idx)`
      - added lowering for cross methods:
        - `get_coverage` -> `__moore_cross_get_coverage(cg, cross_idx)`
        - `get_inst_coverage` -> `__moore_cross_get_inst_coverage(cg, cross_idx)`
      - both resolve indices from enclosing covergroup body order.
    - `tools/circt-sim/LLHDProcessInterpreter.cpp`
      - added dispatch for:
        - `__moore_coverpoint_get_coverage`
        - `__moore_coverpoint_get_inst_coverage`
        - `__moore_cross_get_coverage`
        - `__moore_cross_get_inst_coverage`
  - New regressions:
    - `test/Tools/circt-sim/coverpoint-get-inst-coverage.sv`
      - checks numeric coverpoint `get_coverage` and `get_inst_coverage`
        values (0/50/100) instead of `x`.
    - `test/Tools/circt-sim/cross-get-inst-coverage.sv`
      - checks numeric cross `get_coverage` and `get_inst_coverage` values.
    - removed stale TODO comments in:
      - `test/Tools/circt-sim/syscall-covergroup.sv`
      - `test/Tools/circt-sim/syscall-coverage.sv`

- Follow-up correction (2026-02-27): keep bug-revealing tests, do not delete.
  - Two new semantic tests were initially removed after discovering they
    currently fail in frontend parsing, not runtime.
  - Restored as expected-fail bug trackers:
    - `test/Tools/circt-sim/coverpoint-get-inst-coverage.sv`
    - `test/Tools/circt-sim/cross-get-inst-coverage.sv`
  - Current failure mode (verified):
    - `error: expression of type cg has no member fields`
    - Triggered by valid SV-style member method access:
      `cg_inst.cp.get_inst_coverage()` and `cg_inst.xab.get_inst_coverage()`.

- Coverage regression execution update (2026-02-27): direct run with `build_clang_test` tools.
  - Environment notes:
    - `llvm-lit` remained blocked by missing `FileCheck` in `build_clang_test/bin`.
    - Used direct `circt-verilog` + `circt-sim` invocations for repro checks.
  - Repro confirmations:
    - `test/Tools/circt-sim/coverage-event-implicit-sample.sv`
      - simulation completed and coverage report still showed `cp_addr: 0 hits` and
        `cp_we: 0 hits` despite `@(posedge clk)` sampling event.
    - `test/Tools/circt-sim/coverage-option-at-least-propagation.sv`
      - printed `inst_cov=100` after one hit with `option.at_least = 2`
        (expected behavior is `inst_cov=0`).
  - Frontend blocker still present:
    - `test/Tools/circt-sim/coverage-cross-equal-arity-isolation.sv`
    - `test/Tools/circt-sim/coverage-cross-named-bin-filters-interpret.sv`
    - both currently fail in `circt-verilog` with:
      - `error: expression of type cg has no member fields`
      - for `cg_inst.xab.get_inst_coverage()` member access.

- Runtime regression additions (2026-02-27): cross-specific bug capture in gtests.
  - Added disabled regression tests in `unittests/Runtime/MooreRuntimeTest.cpp`:
    - `MooreRuntimeCrossCoverageTest.DISABLED_CrossBinsIsolatedPerCrossWithSameArity`
    - `MooreRuntimeCrossCoverageTest.DISABLED_CrossNamedBinFiltersAreDeepCopied`
  - Build/run notes:
    - `ninja -C build_clang_test MooreRuntimeTests` initially failed due ccache temp
      permissions under `/run/user/...`; reran with `CCACHE_DISABLE=1`.
    - Running disabled tests explicitly with `--gtest_also_run_disabled_tests`
      reproduces both runtime bugs:
      - same-arity cross isolation test observed `bins_hit=4` and `coverage=100`
        for each cross instead of expected `2` and `50`.
      - filter deep-copy test observed `__moore_cross_is_ignored == false` and
        `bins_hit=1` after mutating caller-owned filter memory; expected ignored
        sample with `bins_hit=0`.
  - Realization:
    - These runtime regressions are valuable now because they isolate runtime
      correctness independently of current frontend cross-method lookup failures.

- Coverage implicit-sampling bug fix (2026-02-27): late `new` assignment path.
  - TDD repro:
    - Added `test/Tools/circt-sim/coverage-event-implicit-sample-late-new.sv` for:
      - `cg cov;` followed by `cov = new;` in an `initial` block.
    - Before fix, direct run with `build_clang_test/bin/circt-verilog` +
      `build_clang_test/bin/circt-sim` reported:
      - `cp_addr: 0 hits`
      - `cp_we: 0 hits`
      - despite declared sampling event `@(posedge clk)`.
  - Root cause:
    - ImportVerilog synthesized implicit sampling procedures only when the
      covergroup variable declaration initializer was `new`.
    - If instantiation happened later via assignment, no sampling process was
      emitted, so implicit event sampling never occurred.
  - Fix:
    - `lib/Conversion/ImportVerilog/Structure.cpp`
      - changed implicit sampling process synthesis trigger from
        declaration-initializer `new` to declared variable type:
        any `covergroup`-typed variable with `getCoverageEvent()` now gets the
        synthesized `always` sampling procedure.
      - behavior is now consistent for both:
        - `cg cov = new;`
        - `cg cov; ... cov = new;`
  - Regression coverage:
    - Existing test `test/Tools/circt-sim/coverage-event-implicit-sample.sv`
      now passes in this tree and no longer needs `XFAIL`.
    - New test `test/Tools/circt-sim/coverage-event-implicit-sample-late-new.sv`
      passes after fix.
  - Validation:
    - `build_clang_test/bin/llvm-lit -sv`
      on both tests passed (2/2).
  - Realization:
    - Presence of a sampling event on the covergroup type, not placement of
      `new`, is the correct condition for emitting implicit sampling wiring.

- Coverage implicit-sampling follow-up (2026-02-27): avoided alias overcount regression.
  - Surprise during post-fix audit:
    - Broad "all covergroup vars with sampling_event" synthesis caused
      overcount when two handles alias the same instance (`b = a`).
    - Repro (`/tmp/cov_alias_double_sample.sv`) showed 8 hits for 4 events
      before this adjustment.
  - Adjustment:
    - Added `Context::covergroupImplicitSamplingVars` to track which variables
      already have synthesized implicit sampling procedures.
    - Restored declaration-side synthesis in `Structure.cpp` to
      initializer-only (`cg var = new;`) and record the variable in that set.
    - Added assignment-side synthesis in `Expressions.cpp` for `var = new;`
      (`AssignmentExpression`), also recording in that set.
    - Result: late `new` works, but alias assignment (`b = a`) no longer
      synthesizes an extra sampling process for `b`.
  - Validation (current environment):
    - `circt-verilog` IR checks:
      - declaration `new` test contains implicit sample call sites.
      - late `new` test contains implicit sample call sites.
      - alias repro now lowers to a single implicit `__moore_coverpoint_sample`
        call site (no duplicate sampling process).
  - Environment blocker note:
    - `circt-sim` binary is currently unavailable due unrelated dirty-tree
      compile failures in UVM fast-path files; avoided touching those files.

- Coverage options propagation bug root cause + fix (2026-02-27).
  - Repro:
    - `test/Tools/circt-sim/coverage-option-at-least-propagation.sv`
      produced `inst_cov=100` after a single hit with `option.at_least = 2`.
  - Root cause:
    - `MooreToCore` now emits option setter runtime calls
      (`__moore_covergroup_set_*`, `__moore_coverpoint_set_*`), but
      `tools/circt-sim/LLHDProcessInterpreter.cpp` coverage dispatch only
      handled create/init/sample/add_bin/get_coverage-style calls.
    - Option setter calls were not dispatched in interpret mode, so runtime
      option state remained default.
  - Fix:
    - Added explicit interpreter handlers for:
      - `__moore_covergroup_set_weight`
      - `__moore_covergroup_set_goal`
      - `__moore_covergroup_set_per_instance`
      - `__moore_covergroup_set_at_least`
      - `__moore_covergroup_set_comment`
      - `__moore_coverpoint_set_weight`
      - `__moore_coverpoint_set_goal`
      - `__moore_coverpoint_set_at_least`
      - `__moore_coverpoint_set_comment`
      - `__moore_coverpoint_set_auto_bin_max`
    - Added explicit interpreter handlers for:
      - `__moore_coverpoint_get_coverage`
      - `__moore_coverpoint_get_inst_coverage`
      to avoid relying on fallback dispatch for these methods.
  - Regression tests:
    - Removed `XFAIL` from
      `test/Tools/circt-sim/coverage-option-at-least-propagation.sv`.
    - Added
      `test/Tools/circt-sim/coverage-option-auto-bin-max-propagation.sv`.
  - Validation status:
    - Updated `LLHDProcessInterpreter.cpp` compiles as an object target.
    - Full `circt-sim` link/run validation currently blocked by unrelated,
      pre-existing `circt-sim.cpp`↔`LLHDProcessInterpreter.h` API mismatch
      errors in this dirty tree.
  - Added compile-stage regressions (do not depend on `circt-sim` binary):
    - `test/Conversion/ImportVerilog/coverage-event-late-new-lowering.sv`
      - asserts late `cov = new;` emits implicit `__moore_coverpoint_sample`.
    - `test/Conversion/ImportVerilog/coverage-event-alias-single-sample-lowering.sv`
      - asserts alias case emits a single implicit sample call site.
  - Lit status:
    - `llvm-lit -sv` on both new ImportVerilog tests passed.

- Coverage fail-loud audit follow-up (2026-02-27): call-indirect + wrapper absorption gaps.
  - TDD repros added before code changes:
    - `test/Tools/circt-sim/coverage-unhandled-runtime-call-error-call-indirect.mlir`
      - unresolved `func.call_indirect` target to `@__moore_cross_get_bins_hit`
        (via vtable entry to `llvm.func`) previously returned X/success silently.
    - `test/Tools/circt-sim/coverage-unhandled-runtime-call-error-func-wrapper.mlir`
      - nested `func.call @call_cov` -> inner `llvm.call @__moore_cross_get_bins_hit`
        previously emitted error but was absorbed with
        `WARNING: func.call ... failed internally (absorbing)` and simulation success.
    - `test/Tools/circt-sim/aot-coverage-unhandled-runtime-call-error-func-wrapper.mlir`
      - same wrapper case in AOT/native mode previously returned default values
        without hard failure.
  - Root causes found:
    - `LLHDProcessInterpreterCallIndirect.cpp` unresolved `!funcOp` fallback set X and
      returned success with no coverage-prefix guard.
    - `LLHDProcessInterpreter.cpp` `interpretFuncCall` and
      `interpretFuncCallCachedPath` failure paths intentionally absorbed internal
      failures, which also swallowed unhandled-coverage failures from nested calls.
    - AOT native dispatch could bypass strict coverage error handling in wrapper
      functions that directly call coverage runtime symbols.
  - Fixes:
    - Added shared coverage helper APIs:
      - `isCoverageRuntimeCallee(...)`
      - `shouldPropagateCoverageRuntimeFailure(...)`
      - `functionHasDirectCoverageRuntimeCall(...)` (+ per-function cache)
    - Added fail-loud coverage checks to additional symbol-miss/external fallback
      branches in `interpretFuncCall` / `interpretLLVMCall`.
    - Added fail-loud guard in `interpretFuncCallIndirect` unresolved-function path.
    - Added sticky per-process flag `sawUnhandledCoverageRuntimeCall` in
      `ProcessExecutionState`; set at all unhandled-coverage error emit sites;
      wrapper paths now propagate failure instead of absorbing.
    - Disabled native func.call dispatch for functions with direct coverage runtime
      calls so AOT wrappers still execute through strict interpreted checks.
  - Test maintenance:
    - Updated `test/Tools/circt-sim/syscall-covergroup.sv` to construct covergroup
      instance at module scope (`cg cg_inst;` + `cg_inst = new();`) so it remains
      valid under the local implicit-event fail-loud rule.
  - Validation:
    - `ninja -C build_clang_test circt-sim circt-sim-compile` (incremental) passed.
    - New regressions:
      - `coverage-unhandled-runtime-call-error-call-indirect.mlir` passed.
      - `coverage-unhandled-runtime-call-error-func-wrapper.mlir` passed.
      - `aot-coverage-unhandled-runtime-call-error-func-wrapper.mlir` passed.
    - Broader coverage sweep passed:
      - `coverage-dispatch-parity.test`
      - `coverage-*.sv`
      - `coverage-*.mlir`
      - `syscall-covergroup.sv`
      - `syscall-coverage.sv`
    - Focused historical coverage-event/error suite passed:
      - `coverage-unhandled-runtime-call-error*.mlir`
      - `coverage-event-implicit-sample*.sv`
      - `test/Conversion/ImportVerilog/coverage-event-*-new-error.sv`

- UVM by-type factory null fix (2026-02-27 11:03:08Z)
  - TDD baseline:
    - `test/Tools/circt-sim/uvm-sequence-body-runtime-dispatch.sv` failed with
      `UVM_FATAL FCTTYP` (`create_4759` path initially; then
      `.automatic_phase_objection` object creation in `create_1547`).
    - `test/Tools/circt-sim/uvm-run-phase-objection-runtime.sv` passed.
  - Root cause:
    - Lowered helper wrappers (`create_by_type_*`) and object helper
      `create_by_type` can transiently return null via factory call-indirect
      startup paths despite having a valid wrapper pointer.
    - Existing fallback only covered `create_by_type` with `parent == null`,
      missing non-root object creation (`.automatic_phase_objection`) and
      wrapper-suffixed helper symbols.
  - Fixes in `tools/circt-sim/LLHDProcessInterpreter.cpp`:
    - Added narrow fallback for `create_by_type_*` helper functions.
      - Uses wrapper vtable dispatch directly via `tryInvokeWrapperFactoryMethod`.
      - Prefers component slot (`slot 1`) when parent is non-null, with safe
        fallback to object slot (`slot 0`).
    - Broadened `create_by_type` fallback to always attempt object creation via
      wrapper slot 0 (removed root-only parent-null gate).
  - Test hardening:
    - Added `// CHECK-NOT: FCTTYP` to
      `test/Tools/circt-sim/uvm-sequence-body-runtime-dispatch.sv`.
  - Validation:
    - `CCACHE_DISABLE=1 ninja -C build_clang_test circt-sim` passed.
    - `build_clang_test/bin/llvm-lit -sv test/Tools/circt-sim/uvm-sequence-body-runtime-dispatch.sv` passed.
    - `build_clang_test/bin/llvm-lit -sv test/Tools/circt-sim/uvm-run-phase-objection-runtime.sv` passed.

- AOT UVM phase-domain parity fix (2026-02-27)
  - TDD/root-cause repro:
    - `build_test/bin/circt-sim /tmp/uvm_seq_body.mlir --top top --compiled=/tmp/uvm_seq_body.so --max-time=50000000000`
      failed at time 0 with
      `UVM_FATAL ... PH/INTERNAL get_domain: m_phase_type is DOMAIN but $cast to uvm_domain fails`.
    - Interpreter baseline on the same MLIR completed to `50000000000 fs`.
  - Diagnostics added/used:
    - `CIRCT_SIM_TRACE_DYN_CAST_CHECK=1` showed compiled-only failure:
      `func=uvm_pkg::uvm_phase::get_domain src=0 target=620`.
    - Added env-gated function-entry trace:
      `CIRCT_SIM_TRACE_GET_DOMAIN_ARG=1`.
      Failing pointer was host-looking and untracked:
      `phase=0x0ca70470 class_id=0 source=none`.
  - Root cause:
    - Mixed native/interpreted execution can pass host object pointers into
      interpreted UVM paths without registering them in `nativeMemoryBlocks`.
    - Interpreted loads on those pointers then miss all memory block maps and
      collapse to unknown/zero, producing false class-id failures.
  - Fix in `tools/circt-sim/LLHDProcessInterpreter.cpp`:
    - Native `func.call` slow path: when a native-dispatched function returns a
      pointer result, auto-register unknown host pointers with a conservative
      fallback native block (`512` bytes).
    - Native `func.call` cached path: same pointer-return registration logic.
    - Trampoline dispatch (`dispatchTrampoline`): for UVM-style pointer args,
      auto-register unknown host pointers before interpreted execution.
    - All registrations are guarded to skip virtual/tagged ranges and already
      tracked pointers; optional pointer tracing uses existing
      `CIRCT_AOT_TRACE_PTR` flow.
  - New regression (AOT-specific):
    - `test/Tools/circt-sim/aot-native-pointer-return-to-interpreted-load.mlir`
      - native `@mk_obj` returns host pointer
      - interpreted intercepted `uvm_pkg::uvm_phase::get_domain` loads class id
      - verifies compiled mode returns `class_id=620` (not 0/X).
  - Validation:
    - `aot-native-pointer-return-to-interpreted-load.mlir` passed.
    - `aot-call-indirect-may-yield-default-demote.mlir` passed (default + opt-in).
    - `aot-queue-pop-front-ptr-invalid-len-guard.mlir` passed.
    - Large workload recheck:
      - `uvm_seq_body` compiled now completes (`status=0`) at
        `Simulation completed at time 50000000000 fs`.

- AOT pointer-boundary hardening (Phase 1 completion pass, 2026-02-27 12:16:13Z)
  - TDD repro:
    - Added runtime unit tests:
      - `MooreRuntimeAssocTest.UnresolvedVirtualPointerReturnsSafeDefaults`
      - `MooreRuntimeQueueTest.UnresolvedVirtualQueuePointerReturnsZeroSize`
    - Before fix: running those tests crashed in `__moore_assoc_exists` on
      unresolved virtual pointer dereference.
  - Root cause:
    - Moore runtime pointer normalization still returned unresolved virtual
      pointers to callers, allowing raw dereference paths in assoc/queue
      helpers.
    - LLHD interpreter destructor did not clear Moore runtime resolver/accessor
      callback bindings, leaving stale callback state possible during shutdown.
  - Fixes:
    - `lib/Runtime/MooreRuntime.cpp`
      - `normalizeHostPtr` now returns `nullptr` (safe default) for:
        - unresolved virtual pointers (`0x10000000..0x1fffffff`)
        - tagged FuncId pointers (`0xf0000000..0xffffffff`)
        - obviously invalid low addresses (`<0x10000`)
      - tracing strings updated to include `returning null`.
    - `tools/circt-sim/LLHDProcessInterpreter.cpp`
      - destructor now clears runtime bridge callbacks:
        - `__moore_assoc_set_ptr_resolver(nullptr, nullptr)`
        - `__moore_signal_registry_set_accessor(nullptr, ..., nullptr)`
      - resets `currentInterpreter` when destroying the active bridge owner.
  - Validation:
    - `utils/ninja-with-lock.sh -C build_test MooreRuntimeTests` passed.
    - `MooreRuntimeAssocTest.*` + queue guard subset passed.
    - `llvm-lit` targeted AOT regressions passed:
      - `aot-native-pointer-return-to-interpreted-load.mlir`
      - `aot-queue-pop-front-ptr-invalid-len-guard.mlir`.

- Phase 2 all-entry interpreter call_indirect dispatch fix (2026-02-27 12:28:57Z)
  - TDD/root-cause repro:
    - `aot-entry-table-trampoline-counter.mlir` failed in compiled mode:
      expected `Trampoline calls: 1` / `Entry-table trampoline calls: 1`,
      observed both counters at `0`.
    - direct repro with `build_test/bin/circt-sim` confirmed:
      tagged FuncId entry table had `0 native, 1 non-native` but runtime still
      executed pure interpreter fallback.
  - Root cause:
    - interpreter `call_indirect` entry-table paths still had a native-only
      dispatch gate:
      `if (!compiledFuncIsNative[fid]) goto interpreted;`
    - this existed in all 4 interpreter dispatch paths
      (X-fallback, static fallback, site-cache, main), violating Phase 2
      all-entry dispatch intent.
  - Fix in `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`:
    - removed non-native dispatch gate in all 4 paths.
    - kept pointer-arg normalization native-only (`isNativeEntry`).
    - kept MAY_YIELD skip native-only (`isNativeEntry &&
      shouldSkipMayYieldNative(fid)`).
    - preserved deny/trap policy behavior.
  - Regression updates:
    - `test/Tools/circt-sim/aot-call-indirect-trampoline-pointer-preserve.mlir`
      now expects `Entry-table trampoline calls: 1` (was `0`).
  - Validation:
    - Build:
      - `utils/ninja-with-lock.sh -C build_test circt-sim` passed.
    - Focused checks (manual RUN-line equivalent with `build_test/bin/FileCheck`):
      - `aot-entry-table-trampoline-counter.mlir` passed (red -> green).
      - `aot-call-indirect-trampoline-pointer-preserve.mlir` passed.
      - `aot-call-indirect-native-pointer-tag-normalize.mlir` passed.
      - `aot-call-indirect-may-yield-default-demote.mlir` passed (default + opt-in).
    - Profiling sanity (`uvm-sequence-body-runtime-dispatch.sv`):
      - `Simulation completed at time 50000000000 fs`, `EXIT_CODE=0`
      - `Entry-table native calls: 6`
      - `Entry-table trampoline calls: 35`
      - `Max AOT depth: 1`
      - `WALL=9.780s`.
