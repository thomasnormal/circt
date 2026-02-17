# AVIP Performance and JIT Maturation Plan

## 1. Problem Statement

Current AVIP simulation throughput is bottlenecked by interpreter-heavy UVM runtime paths, with AHB particularly prone to high memory pressure and occasional OOM. Recent targeted intercepts improved specific workloads, but the approach does not scale if every AVIP requires manual hot-function extraction.

We need a long-term plan that:

1. Preserves semantic parity with UVM behavior.
2. Delivers predictable speedups across all AVIPs.
3. Avoids per-AVIP custom hacks.
4. Keeps memory usage bounded on heavy tests (especially AHB).

## 2. Success Criteria

## Functional

1. Deterministic parity harness exists for `circt-sim` vs Xcelium baselines.
2. All supported AVIPs run with stable pass/fail behavior under fixed seeds.
3. New fast paths are guarded by regression tests and kill-switch env flags.

## Performance

1. 3-10x speedup on current hot-path dominated AVIPs.
2. AHB no longer OOMs at standard run limits.
3. Startup overhead remains acceptable (no large compile latency spikes).

## Maintainability

1. Fast paths are centralized and table-driven where possible.
2. Interpreter source is split/refactored to keep file size manageable.
3. JIT and non-JIT execution paths share common correctness tests.

## 3. Constraints and Invariants

1. Correctness first: no silent semantic drift for UVM phase/port behavior.
2. Every bug fix or feature gets at least one focused regression test.
3. Keep fallback interpreter behavior available behind feature flags.
4. Optimize for long-term architecture, not AVIP-specific special casing.

## 4. Architecture Direction

## 4.1 Two-Tier Execution Model

1. Tier A: Fast native intercepts for extremely hot, semantically simple UVM helpers.
2. Tier B: JIT-compiled hot function bodies for stable, repeatedly executed regions.

## 4.2 Dispatch Strategy

1. Introduce a dispatch registry keyed by fully qualified callee name + call form (`func.call`, `call_indirect`, `llvm.call`).
2. Registry entries declare:
3. Eligibility predicate.
4. Native fast-path handler or JIT thunk.
5. Fallback policy.

This removes scattered ad-hoc string matching from large interpreter functions.

## 4.3 JIT Scope

1. Start with leaf-like pure/near-pure helper functions and hot loops.
2. Compile once, cache by function symbol + relevant specialization key.
3. Keep deopt path to interpreter for unsupported ops or dynamic corner cases.

## 5. Workstreams

## WS1: Deterministic Benchmark and Parity Infrastructure

1. Harden `utils/run_avip_circt_sim.sh`:
2. Fixed seed matrix.
3. Uniform timeout/memory policy.
4. Structured TSV/JSON outputs for machine diffing.
5. Harden `utils/run_avip_xcelium_reference.sh` with matching matrix and output schema.
6. Add comparison tool/script producing:
7. UVM fatal/error deltas.
8. Coverage deltas.
9. Sim-time and wall-time deltas.
10. Status verdict per AVIP.

Deliverable: reproducible baseline report artifact for each branch.

## WS2: Interpreter Refactor for Scale

1. Split `LLHDProcessInterpreter.cpp` into focused components:
2. UVM dispatch/intercepts.
3. memory/config/resource helpers.
4. phase and scheduler hooks.
5. Add unit-testable helper classes for connection maps, FIFO routing, cache invalidation.
6. Enforce local invariants (e.g., sequencer routing cache invalidates on connect mutations).

Deliverable: reduced monolith complexity and lower risk of regressions.

## WS3: Hot-Path Fast Paths

1. Expand targeted native handlers only where:
2. Semantics are clear.
3. call count is high.
4. measurable speedup is expected.
5. Keep each fast path:
6. gated by env flag initially.
7. backed by direct regression test.
8. benchmarked before/after on AVIPs.
9. Candidate areas:
10. UVM report/printer helpers.
11. sequencer pull-port get/try_next_item/item_done routing.
12. phase objection helpers.
13. string/format conversion helpers in critical loops.

Deliverable: short-term speed gains while JIT matures.

## WS4: Mature JIT Pipeline

1. Build a JIT compilation manager:
2. identifies hot functions by profile counters.
3. compiles MLIR/LLVM IR region to native callable stubs.
4. caches compiled code per process/module context.
5. Add speculative guards + fallback/deopt.
6. Add telemetry:
7. compile count.
8. hit rate.
9. deopt/fallback count.
10. wall-time split (compile vs execute).

Deliverable: scalable performance gains without one-off function interception.

## WS5: Memory and OOM Hardening (AHB Priority)

1. Add memory profiling counters:
2. native block allocations.
3. dynamic string table growth.
4. sequencer queues and connection maps.
5. Implement bounded retention policies where semantically safe.
6. Reduce duplicate object snapshots/copies in hot loops.
7. Validate with dedicated AHB stress profile.

Deliverable: stable RSS under configured limits for AHB.

## 6. Milestones and Gates

## M1: Baseline and Tooling Complete

1. Deterministic AVIP and Xcelium scripts produce comparable artifacts.
2. One-command parity report generation.
3. CI-ready summary format.

Gate: repeatable parity report with no manual parsing.

## M2: Refactor + Fast-Path Hardening

1. Core interpreter split complete.
2. Existing fast paths migrated into registry.
3. Reconnect/cache invalidation regressions covered.

Gate: no functional regressions on full AVIP sweep; measurable speedup on top 3 hot AVIPs.

## M3: Initial JIT Rollout

1. Hotness-based JIT for first function class enabled behind flag.
2. Deopt fallback proven.
3. Performance telemetry exported in run summary.

Gate: improvement on AHB/APB/AXI4 with parity maintained.

## M4: JIT Expansion + Default-On Candidate

1. Wider function coverage.
2. compile-cache stability validated.
3. memory regressions addressed.

Gate: default-on recommendation with rollback switch still available.

## 7. Testing Plan

1. Per-change local tests:
2. focused `lit` regression(s).
3. relevant `unittests`.
4. Integration sweeps:
5. `~/mbit/*avip*` full matrix.
6. `~/sv-tests/`.
7. `~/verilator-verification/`.
8. `~/yosys/tests/`.
9. `~/opentitan/` targeted suites.
10. Every fast path change includes:
11. positive behavior check.
12. fallback behavior check (flag off).
13. reconnect/rebinding/cache invalidation checks where applicable.

## 8. Risks and Mitigations

1. Risk: semantic drift in UVM behavior.
2. Mitigation: strict parity harness + staged flag rollout + fallback path.
3. Risk: compile overhead from JIT outweighs runtime gain.
4. Mitigation: hotness threshold, caching, compile budget controls.
5. Risk: monolithic interpreter remains difficult to evolve.
6. Mitigation: modular split and dedicated helper APIs with unit tests.
7. Risk: memory regressions (AHB OOM).
8. Mitigation: explicit memory telemetry + stress tests in gating runs.

## 9. Immediate Next Actions

1. Land deterministic AVIP/Xcelium runner refactor and parity diff report.
2. Land reconnect cache invalidation regression for pull-port routing.
3. Migrate existing fast paths into a first-pass dispatch registry.
4. Add hotness counters and profiling output for candidate JIT targets.
5. Prototype first JIT-compiled helper region behind opt-in flag.

## 10. Reporting and Process

1. Update `avip_engineering_log.md` after each major benchmark pass.
2. Update `CHANGELOG.md` for significant runtime behavior/perf changes.
3. Track milestone status in this document with dates and commit IDs.

## 11. Progress Notes (2026-02-17)

1. WS5 memory hardening (sequencer retention) advanced:
   - pull-port sequencer queue cache now has bounded-entry policy with optional
     eviction (`CIRCT_SIM_UVM_SEQ_QUEUE_CACHE_MAX_ENTRIES`,
     `CIRCT_SIM_UVM_SEQ_QUEUE_CACHE_EVICT_ON_CAP`) and summary telemetry.
   - sequence item ownership mapping (`item -> sequencer`) is now consumed when
     `finish_item` enqueues, reducing historical retention in long runs.
   - sequencer native-state telemetry now reports ownership-map
     stores/erases/peak/live and FIFO/live waiter dimensions.
2. Regression coverage added for both cap policy and ownership-map reclamation:
   - `test/Tools/circt-sim/uvm-sequencer-queue-cache-cap.mlir`
   - `test/Tools/circt-sim/finish-item-blocks-until-item-done.mlir`
3. WS5 observability expanded with profile-summary memory telemetry:
   - summary now reports global/malloc/native/process memory footprint, dynamic
     string + config_db sizes, analysis connection graph size, and sequencer
     FIFO dimensions.
   - covered by `test/Tools/circt-sim/profile-summary-memory-state.mlir`.
4. WS5 now includes sampled runtime peak memory telemetry:
   - added periodic sampling hook in execution hot loops
     (`executeStep`, `interpretFuncBody`, `interpretLLVMFuncBody`) with shared
     snapshot helper.
   - sampling is controlled by
     `CIRCT_SIM_PROFILE_MEMORY_SAMPLE_INTERVAL` (default `65536` in
     summary mode).
   - summary now emits:
     `[circt-sim] Memory peak: samples=... sample_interval_steps=...`.
   - covered by `test/Tools/circt-sim/profile-summary-memory-peak.mlir`.
5. Current WS5 limitation:
   - peak visibility is global-only; we still need low-overhead attribution
     buckets (by process/function class) for deterministic AHB OOM root-cause
     prioritization.
6. WS5 attribution step landed for largest-process peak visibility:
   - memory snapshots now record `largest_process` and
     `largest_process_bytes`.
   - peak summary now reports `largest_process_func` for the largest process
     at the global memory peak sample.
   - covered by:
     - `test/Tools/circt-sim/profile-summary-memory-state.mlir`
     - `test/Tools/circt-sim/profile-summary-memory-peak.mlir`
7. Remaining WS5 gap after this pass:
   - attribution is single-winner only; we still need top-N process buckets and
     growth-delta attribution categories for robust AHB closure triage.
8. WS5 attribution now includes top-N process ranking at summary time:
   - `CIRCT_SIM_PROFILE_MEMORY_TOP_PROCESSES` controls rank depth
     (default `3` in summary mode).
   - summary emits:
     `[circt-sim] Memory process top[N]: proc=... bytes=... name=... func=...`.
9. Remaining WS5 gap after top-N landing:
   - ranking is point-in-time only; we still need time-window delta attribution
     and map-level growth buckets for deterministic OOM root cause closure.
