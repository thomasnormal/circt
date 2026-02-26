# CIRCT Sim Engineering Log

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
