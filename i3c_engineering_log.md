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

### 2026-02-19: Root cause found and fixed for nested struct field clobber

Scope:
- Investigated first-write divergence with a minimal standalone reproducer:
  - `/tmp/i3c_struct_inout_nested.sv`
  - shape: nested `inout struct` update under `fork/join_none` + `disable fork`

Pre-fix observation:
- `circt-sim` result:
  - `addr=0 op=0 no=7 wd0=4e` (`FAIL`)
  - log: `/tmp/i3c_struct_inout_nested.circt.log`
- Xcelium result:
  - `addr=68 op=0 no=8 wd0=4e` (`PASS`)
  - log: `/tmp/i3c_struct_inout_nested.xcelium.log`

Instrumentation finding:
- Temporary trace in `LLHDProcessInterpreter` for signal-backed struct field
  writeback showed first child update reading parent as `X` and not from pending
  drive:
  - log: `/tmp/i3c_struct_inout_nested_dbg.trace2.log`
  - key symptom:
    - `field=no_of_i3c_bits_transfer ... fromPending=0 ... parent=0xX`
- This identified wrong driver-id semantics in subfield signal write paths:
  they used per-process IDs, creating false multi-driver resolution behavior for
  procedural updates.

Fix:
- File:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
- Updated signal-backed subfield writeback for:
  - `llhd.sig.struct_extract`
  - `llhd.sig.array_get`
- Policy now matches regular `llhd.drv`:
  - `sigId` by default
  - `getDistinctContinuousDriverId(...)` only for
    `distinctContinuousDriverSignals`.
- Removed temporary trace instrumentation after diagnosis.

Regression added:
- `test/Tools/circt-sim/fork-struct-field-last-write.sv`
- Intent: ensure sibling struct fields are preserved and forked procedural field
  updates keep last-write semantics.

Focused validation:
- Build:
  - `ninja -C build-test circt-sim circt-verilog` (`PASS`)
- lit:
  - `fork-disable-ready-wakeup.sv` (`PASS`)
  - `fork-disable-defer-poll.sv` (`PASS`)
  - `disable-fork-halt.mlir` (`PASS`)
  - `fork-struct-field-last-write.sv` (`PASS`)
- Reproducer after fix:
  - `/tmp/i3c_struct_inout_nested.afterfix.log`
  - `/tmp/i3c_struct_inout_nested_dbg.afterfix.log`
  - output now: `addr=68 op=0 no=8 wd0=4e` (`PASS`)

I3C AVIP lane status after fix:
- Default resource guard run:
  - `/tmp/avip-circt-sim-i3c-after-struct-driver-fix-20260219-010837/matrix.tsv`
  - compile `OK` (`142s`), sim `FAIL` (resource guard):
    `RSS 4625 MB exceeded limit 4096 MB`
- Raised guard run (`CIRCT_MAX_RSS_MB=8192`):
  - `/tmp/avip-circt-sim-i3c-after-struct-driver-fix-rss8g-20260219-011323/matrix.tsv`
  - compile `OK` (`165s`), sim exits `OK` (`300s`)
  - remaining mismatch:
    - `UVM_ERROR ... i3c_scoreboard.sv(162)`
  - coverage printout: `100% / 100%`

Current conclusion:
- The struct-field clobber bug is real, fixed, and regression-protected.
- I3C line-162 parity gap remains open; continue root-cause work on remaining
  payload divergence path.

### 2026-02-19: wait_event signal-ref hardening + deterministic lane refresh
Files:
- `lib/Conversion/MooreToCore/MooreToCore.cpp`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `test/Conversion/MooreToCore/fork-wait-event-runtime-signal-ref.mlir`

Key changes:
1. Fork-lowered `WaitEventOpConversion` now recovers signal refs through
   `moore.read`, direct ref-typed values, conversion casts, and remapped values.
2. `__moore_wait_event` runtime signal lookup now uses `resolveSignalId(...)`
   so block-arg/input-mapped refs can resolve back to concrete signals.
3. Added conversion regression locking non-null signal-pointer call wiring for
   fork wait-event lowering.

Focused validation:
- Build PASS:
  - `ninja -C build-test circt-opt circt-verilog circt-sim`
- lit PASS (`2/2`):
  - `fork-wait-event-runtime-signal-ref.mlir`
  - `fork-struct-field-last-write.sv`
- IR spot-check confirms `__moore_wait_event` callsites in the fork-struct test
  now carry traced `%signal_ptr` operands instead of null pointer fallback.

I3C deterministic compile lane:
- Command:
  - `AVIPS=i3c SEEDS=1 CIRCT_SIM_MODE=compile CIRCT_SIM_WRITE_JIT_REPORT=1 COMPILE_TIMEOUT=180 SIM_TIMEOUT=180 MEMORY_LIMIT_GB=20 utils/run_avip_circt_sim.sh /tmp/avip-i3c-wait-event-20260219-032229`
- Artifacts:
  - `/tmp/avip-i3c-wait-event-20260219-032229/matrix.tsv`
  - `/tmp/avip-i3c-wait-event-20260219-032229/i3c/sim_seed_1.log`
- Result:
  - compile `OK` (`32s`)
  - sim `OK` (`68s`)
  - parity mismatch persists:
    - `UVM_ERROR ... i3c_scoreboard.sv(162) @ 4273 ns`
  - coverage print remains `100% / 100%`

Status:
- This pass improves wait-event fidelity and preserves prior struct-field fix.
- I3C scoreboard parity (line 162) remains the active blocker.

### 2026-02-19: wait-sensitivity diagnostic run on compiled I3C MLIR
Command:
- `env CIRCT_SIM_TRACE_WAIT_SENS=1 CIRCT_UVM_ARGS='+UVM_TESTNAME=i3c_writeOperationWith8bitsData_test +ntb_random_seed=1 +UVM_VERBOSITY=UVM_LOW' build-test/bin/circt-sim /tmp/avip-i3c-wait-event-20260219-032229/i3c/i3c.mlir --top hdl_top --top hvl_top --mode=compile --timeout=180 --max-time=7940000000000 --max-wall-ms=210000`
- log: `/tmp/i3c-wait-sens-20260219-032917.log`

Observed:
1. Runtime wait-sensitivity tracing emits repeated observed entries for core
   signals (`clk`, `rst`, `I3C_SCL`, `I3C_SDA`).
2. No immediate evidence of wait-event no-op starvation in this trace slice.
3. I3C scoreboard parity issue remains open (line-162 check-phase failure still
   reproducible in bounded compile-mode lanes).

Interpretation:
- The current mismatch is likely above raw wait-list population (transaction
  lifecycle/conversion ordering still suspect), though wait semantics remain
  part of the broader timing surface.

### 2026-02-19: ruled-out hypothesis - basic struct->class `to_class` conversion is functional
Reproducer:
- `/tmp/i3c_to_class_min.sv`
- compile: `build-test/bin/circt-verilog /tmp/i3c_to_class_min.sv --no-uvm-auto-include -o /tmp/i3c_to_class_min.mlir`
- run: `build-test/bin/circt-sim /tmp/i3c_to_class_min.mlir --top i3c_to_class_min`

Observed output:
- `ta=68 op=0 n=1 wd0=4e`
- `PASS`

Implication:
- The remaining I3C line-162 mismatch is unlikely to be caused by a generic
  failure of `output class` assignment in `to_class` helpers.
- Focus remains on monitor transaction lifecycle/order and wait/fork timing
  interactions in the full AVIP flow.

### 2026-02-19: intra-interface copy-link trial (reverted)

Goal:
- Test whether missing same-interface copy links (notably `sig_1.field_3 ->
  sig_1.field_7`) caused stale `sda_i` sampling in target BFM.

Attempt:
- Added constrained same-interface copy-pair linking in
  `LLHDProcessInterpreter` copy-pair resolution.
- Used traced replay to confirm link materialization and propagation activity.

Observed:
- Traces confirmed new link presence and runtime toggles on `sig_1.field_7`.
- Functional parity did not improve; scoreboard mismatch persisted.
- Variant runs exposed regressions (target-side coverage collapse in some
  deterministic lanes), indicating this was not a safe fix path.

Action:
- Reverted the change.
- Re-validated stable baseline lane:
  - `/tmp/avip-i3c-post-revert-20260219-1/matrix.tsv`
  - compile `OK` (`30s`), sim `OK` (`70s`)
  - mismatch still at `i3c_scoreboard.sv(162)`.

Conclusion:
- The primary bug is likely above raw interface shadow propagation.
- Next pass should focus on target transaction construction/conversion ordering
  rather than additional interface auto-link heuristics.

### 2026-02-19: nested `sig.extract` memory-layout fix (struct->array->bit refs)

Problem observed while chasing I3C monitor parity:
- Nested bit drives through refs shaped like
  `sig.extract(sig.array_get(sig.struct_extract(...)))` were not preserving
  correct array element mapping on memory-backed refs.
- This can corrupt/lose writes in monitor-style task code that updates bits of
  struct array fields over waits.

Minimal reproducer (outside AVIP):
- `/tmp/inout_struct_wait_inc.sv`
- pre-fix output on circt-sim:
  - `no_bits=8 w0=0` (`FAIL`)
- expected/xcelium-style behavior:
  - `no_bits=8 w0=fb`

Root cause:
- In `LLHDProcessInterpreter` sig.extract drive handling, offset accumulation
  for nested `sig.array_get` used HW indexing (`idx * elemWidth`) even on
  memory-backed refs (LLVM aggregate layout requires reversed array index
  mapping).
- The same accumulated offset was reused across both signal-backed and
  memory-backed paths, mixing layout rules.

Fix:
- File:
  - `tools/circt-sim/LLHDProcessInterpreter.cpp`
- Changes:
  1. Track two offsets during nested extraction tracing:
     - `signalBitOffset` (HW signal layout)
     - `memoryBitOffset` (LLVM memory layout)
  2. For nested `sig.array_get`, use `computeMemoryBackedArrayBitOffset(...)`
     when forming memory-backed offset.
  3. For nested `sig.struct_extract`, compute signal vs memory field offsets
     separately.
  4. Use `memoryBitOffset` in all memory-backed sig.extract drive paths and
     `signalBitOffset` in signal-backed path.

Regression added:
- `test/Tools/circt-sim/sig-extract-struct-array-bit-memory-layout.sv`
- Checks:
  - `no_bits=8 w0=fb`
  - `PASS`

Validation:
1. Build PASS:
   - `ninja -C build-test circt-sim`
2. Reproducer PASS after fix:
   - `build-test/bin/circt-sim /tmp/inout_struct_wait_inc.mlir --top inout_struct_wait_inc`
   - output: `no_bits=8 w0=fb`
3. Focused regressions PASS:
   - `sig-extract-struct-array-bit-memory-layout.sv`
   - `timewheel-slot-collision-wait-event.mlir`
   - `func-start-monitoring-resume-fast-path.mlir`
   - `moore-wait-event-sensitivity-cache.mlir`

I3C status after this fix:
- Deterministic bounded replay still ends with
  `UVM_ERROR ... i3c_scoreboard.sv(162)` and coverage `21.4286 / 21.4286`
  (`/tmp/i3c-after-sigextract-fix.log`).
- So this fix removes a real low-level correctness bug but does not yet close
  the remaining I3C parity gap.

### 2026-02-19: field-drive trace follow-up (post sig.extract fix)

Command:
- `env CIRCT_SIM_TRACE_I3C_FIELD_DRIVES=1 CIRCT_UVM_ARGS='+UVM_TESTNAME=i3c_writeOperationWith8bitsData_test +ntb_random_seed=1 +UVM_VERBOSITY=UVM_LOW' build-test/bin/circt-sim /tmp/avip-i3c-post-revert-20260219-1/i3c/i3c.mlir --top hdl_top --top hvl_top --mode=compile --timeout=180 --max-time=4500000000 --max-wall-ms=120000`
- log: `/tmp/i3c-field-after-fix.log`

Observed:
1. The new nested `sig.extract` fix is active: trace shows memory-backed field
   writes to `writeData`, `writeDataStatus`, and
   `no_of_i3c_bits_transfer` in conversion paths (e.g. `from_class_5760`).
2. Target monitor activity still looks incomplete in this bounded lane:
   - repeated `i3c_target_monitor_bfm::sample_target_address` writes are present,
   - but no trace evidence of target-monitor `sample_write_data` field writes
     before check-phase.
3. Scoreboard error remains unchanged:
   - `UVM_ERROR ... i3c_scoreboard.sv(162)`
   - coverage remains `21.4286 / 21.4286`.

Interpretation:
- Remaining I3C issue is likely in monitor sampling/lifecycle timing (target
  monitor not progressing through expected data sampling path), not in the
  newly fixed nested sig.extract memory-layout bug.

### 2026-02-19: tri-state suppression transition gating replay

Runtime change:
1. `tools/circt-sim/LLHDProcessInterpreter.cpp/.h`
   - kept source->destination propagation links intact under suppressed
     mirror-store handling (removed destructive link-removal behavior).
   - added per-destination condition tracking:
     - `interfaceTriStateCondLastValue`
     - `interfaceTriStateCondSeen`
   - retained-value derivation now triggers only when:
     - destination is `X`, or
     - tri-state condition transitions `1 -> 0`.

Rationale:
- release-edge derivation is still needed to avoid latch-style stale drive
  behavior in tri-state regressions.
- repeated derivation while condition stays `0` can clobber observed values.

Validation:
1. Build PASS:
   - `ninja -C build-test circt-sim`
2. Tri-state regression set PASS:
   - `interface-tristate-suppression-cond-false.sv`
   - `interface-tristate-signal-callback.sv`
   - `interface-intra-tristate-propagation.sv`
3. I3C AVIP replay:
   - command:
     - `CIRCT_VERILOG=build-test/bin/circt-verilog CIRCT_SIM=build-test/bin/circt-sim AVIPS=i3c SEEDS=1 SIM_TIMEOUT=240 COMPILE_TIMEOUT=300 MEMORY_LIMIT_GB=20 MATRIX_TAG=i3c-prop-fix utils/run_avip_circt_sim.sh`
   - matrix:
     - `/tmp/avip-circt-sim-20260219-091053/matrix.tsv`
   - result:
     - compile `OK` (`32s`)
     - sim `OK` (`162s`)
     - coverage still `21.4286 / 21.4286`
     - scoreboard mismatch persists:
       - `UVM_ERROR ... i3c_scoreboard.sv(179)`
4. Bounded traced window:
   - command:
     - `CIRCT_SIM_TRACE_IFACE_PROP=1 ... --max-time=340000000`
   - log:
     - `/tmp/i3c-bounded-340-prop.log`
   - observed:
     - no `I3C_SCL/I3C_SDA fanout=0` collapse in this bounded slice.
     - data path still converges to reserved/zero transaction values.

Status:
- suppression behavior is now less destructive and trace behavior is cleaner.
- primary I3C blocker remains transaction sampling/conversion ordering (not
  fully resolved by suppression-path changes).

### 2026-02-19: bounded cast-layout diagnostic (340ns window)

Command:
- `CIRCT_UVM_ARGS='+UVM_TESTNAME=i3c_writeOperationWith8bitsData_test +ntb_random_seed=1 +UVM_VERBOSITY=UVM_LOW' CIRCT_SIM_TRACE_I3C_CAST_LAYOUT=1 build-test/bin/circt-sim /tmp/avip-circt-sim-20260219-091053/i3c/i3c.mlir --top hdl_top --top hvl_top --max-time=340000000 --timeout=90`
- log: `/tmp/i3c-bounded-340-cast.log`

Key findings:
1. Layout conversion itself appears stable in this slice:
   - trace lines show `in{...} == out{...}` for controller/target monitor
     and driver conversion points.
2. Wrong transaction values are already present before/at monitor sampling:
   - controller driver launch: `ta=104 bits=8` (expected command origin).
   - monitor/target sample paths later observe `ta=1 bits=0`.
3. Scoreboard debug at same window confirms collapsed payload:
   - `[I3C-SBDBG] ... ctrl_wsz=0 tgt_wsz=0 ctrl_w0=0 tgt_w0=0`
   - coverage remains `21.43%`.

Conclusion:
- current evidence points away from LLVM<->HW struct layout remap as the
  primary culprit; issue is more likely monitor-side sampling/progression timing
  before conversion.
