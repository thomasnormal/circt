
# UVM Support Plan: Semantic Core-First and Interceptor Surface Reduction

## Summary

Build UVM support by treating failures as SystemVerilog pipeline/runtime semantic bugs first (ImportVerilog → MooreToCore → Sim runtime), then reducing circt-sim UVM interceptors in controlled waves
with semantic parity gates.
Current state indicates high interceptor load in tools/circt-sim/LLHDProcessInterpreter.cpp (roughly 40 interceptor references, 86 uvm_pkg:: references, and ~400 calleeName-driven branches), so
reduction must be staged and measurable.

## Current Status (2026-03-01)

- [x] Added semantic UVM child-iteration regression:
  - `test/Runtime/uvm/uvm_component_child_iteration_semantic_test.sv`
- [x] Root-caused and fixed `uvm_simple_test` run-phase regression by reducing fragile component-child fastpaths:
  - `CIRCT_SIM_ENABLE_UVM_COMPONENT_CHILD_FASTPATHS` now opt-in.
- [x] Root-caused and fixed analysis fanout regression (`10/100` deliveries) by reducing fragile native analysis interceptors:
  - `CIRCT_SIM_ENABLE_UVM_ANALYSIS_NATIVE_INTERCEPTS` now opt-in.
- [x] Semantic parity gate slice green:
  - `uvm_component_child_iteration_semantic_test.sv`
  - `uvm_simple_test.sv`
  - `uvm-tlm-analysis-100.sv`
- [ ] Continue Wave C reduction with parity gates for:
  - phase-hopper intercept family
  - factory/type-resolution intercept family
  - sequencer-specific fallback intercept family

## Locked Decisions

- Rollout mode: Semantic-first gates.
- uvm-core policy: No uvm-core edits (treat lib/Runtime/uvm-core as read-only for this program).
- Scope priority: UVM first, but fixes should land in ImportVerilog/MooreToCore/Sim core paths whenever possible.
- Out of scope for this plan: AOT-specific feature work.

## Success Criteria

1. UVM tests are semantic (runtime-checked), not parse/lowering-only.
2. No silent pass behavior for core UVM workflows (config_db, phase graph, objections, factory, ports/TLM, sequencing, RAL, reporting).
3. Interceptor surface area is reduced with zero net regression in semantic suites.
4. Each removed interceptor has replacement semantic coverage and a root-cause note in docs/SVA_ENGINEERING_LOG.md.
5. Gap tracker remains current in docs/PROJECT_GAPS_MANUAL_WRITEUP.md.

## Public API / Interface / Type Changes

1. Add structured interceptor telemetry in circt-sim:
- New optional runtime summary (counts by interceptor key, hit/miss/fallback).
- Stable reason codes for fallback/unsupported paths.
2. Add lit-facing feature toggle strategy:
- Per-test env knobs to disable specific interceptor families (for parity A/B tests).
3. Add semantic test conventions:
- Standardized UVM runtime check macros/patterns for pass/fail determinism.
4. Add diagnostics IDs (not free-form text only) for unsupported/fallback categories used by scripts.

## Workstreams

### 1. Semantic Test Hardening (TDD Foundation)

1. Inventory UVM tests in test/Runtime/uvm and classify:
- semantic-strong, semantic-weak, parse-only.
2. For each high-value behavior, create/upgrade one minimal semantic test that:
- Fails before fix.
- Asserts runtime behavior via explicit checks/logs/FileCheck.
3. Priority test themes:
- config_db scope/path precedence.
- phase traversal/order/scope (find, find_by_name, is_before, is_after).
- objections raise/drop/drain semantics.
- factory resolution/create-by-type and create-by-name.
- sequencer arbitration/get/lock/grab.
- port connect/size/write dispatch.
4. Add negative tests for silent-failure modes (wrong value but no crash).

### 2. Interceptor Taxonomy and Risk Map

1. Build a machine-readable interceptor inventory from tools/circt-sim/LLHDProcessInterpreter.cpp:
- Category: semantic-fix, perf-fastpath, bug-workaround, startup/bootstrap, legacy-compat.
- Dependency: frontend bug, runtime representation gap, scheduler semantics, pointer model.
2. Label each interceptor with:
- Removal precondition (which semantic tests must be green).
- Replacement owner (ImportVerilog/MooreToCore/Sim runtime).
- Risk tier (low/medium/high).

### 3. Core Semantics Closure (Replace Root Causes)

1. Fix root causes in preferred order:
- ImportVerilog symbol/scope/name resolution correctness.
- MooreToCore object/ref/value representation correctness.
- Sim runtime semantics (scheduler/process wait/resume, assoc array/object identity).
2. For each bug:
- Add minimal non-UVM semantic repro test first.
- Confirm fail.
- Implement core fix.
- Re-run repro + affected UVM semantic tests.
3. Avoid adding new UVM-specific interceptors unless temporary and explicitly time-boxed with removal issue.

### 4. Interceptor Reduction Waves

1. Wave A (low-risk/perf-only fastpaths):
   - Remove or gate interceptors that do not change semantics, only shortcut cost.
     - Acceptance: no semantic test changes, only perf delta.
     2. Wave B (deterministic wrapper interceptors):
        - Replace with canonical core semantics where behavior is already equivalent.
          3. Wave C (high-risk semantic interceptors):
             - config_db, phase hopper/objections, factory/type resolution, port/TLM virtual dispatch.
               - Remove only after corresponding semantic suites are green under interceptor-off mode.
               4. Each wave ships with:
               - before/after telemetry snapshot,
               - removed interceptor list,
               - explicit rollback criterion.

### 5. CI and Regression Strategy

               1. Add new lit groups:
               - uvm-semantic-core (must pass without targeted interceptor families).
               - uvm-semantic-compat (default behavior).
               2. Add parity job:
               - run same semantic suite with interceptors enabled vs disabled (selected families).
               - fail on output/behavior divergence.
               3. Add trend metrics:
               - interceptor hit count over time,
               - count of tests requiring interceptor families,
               - unresolved root-cause backlog size.

### 6. Documentation and Engineering Log Discipline

               1. Every root cause entry in docs/SVA_ENGINEERING_LOG.md includes:
               - failing semantic test,
               - minimal repro,
               - core cause,
               - why interceptor was removed/retained.
               2. Keep docs/PROJECT_GAPS_MANUAL_WRITEUP.md checklist authoritative:
               - [x] only when semantic runtime proof exists.
               - include test file references for closure evidence.

## Detailed Execution Sequence (Decision-Complete)

### Phase 0: Baseline (Day 1-2)

               1. Freeze baseline test matrix for UVM + key ImportVerilog/MooreToCore semantic tests.
               2. Generate interceptor inventory and assign categories/risk tiers.
               3. Establish telemetry output format and capture baseline interceptor usage.

### Phase 1: High-Value Semantic Gaps (Week 1)

               1. config_db semantic correctness first.
               2. Phase graph lookup/order semantics second.
               3. Objection lifecycle and drain semantics third.
               4. For each, TDD cycle: add failing semantic test → core fix → parity run.

### Phase 2: Wave A/B Reduction (Week 2)

               1. Disable/remove low-risk and perf-only interceptors.
               2. Re-run semantic suites with parity checks.
               3. Fix any regression only in core pipeline/runtime.

### Phase 3: Wave C Reduction (Week 3+)

               1. Tackle high-risk interceptors one family at a time:
               - factory resolution,
               - phase hopper,
               - port/TLM dispatch,
               - sequencing helpers.
               2. Keep temporary gates only where semantic proof is incomplete.

### Phase 4: Stabilization and Exit

               1. Interceptor surface report (remaining interceptors, justification, owner).
               2. Lock CI gates requiring semantic parity for all retained families.
               3. Publish final closure summary in both docs files.

## Test Cases and Scenarios (Must Exist Before Interceptor Removal)

               1. config_db:
               - scoped overrides, wildcard precedence, component-context get/set, phase-time visibility.
               2. Phase graph:
               - cross-scope find/find_by_name, ordering (is_before/is_after) with and without scope constraints.
               3. Objections:
               - raise/drop count integrity, drain-time behavior, phase transition correctness.
               4. Factory:
               - type/name override chains, inst override precedence, create_* parity.
               5. Ports/TLM:
               - connect/resolve/size semantics, analysis port fanout, virtual interface dispatch behavior.
               6. Sequencer:
               - get/put/item_done ordering, lock/grab arbitration, starvation/timeout behavior.
               7. RAL/reporting:
               - mirrored vs desired state transitions, reporting severity/path formatting accuracy.
               8. Cross-layer semantic repros:
               - minimal non-UVM versions of each discovered bug class.

## Risks and Mitigations

               1. Risk: Removing interceptors causes broad regressions.
               - Mitigation: wave gates + parity suite + immediate rollback threshold.
               2. Risk: Hidden dependence on uvm-core local behavior.
               - Mitigation: read-only uvm-core policy, solve in CIRCT core semantics only.
               3. Risk: tests still pass syntactically but not semantically.
               - Mitigation: enforce runtime assertions and negative checks in every upgraded test.
               4. Risk: performance regressions after fastpath removal.
               - Mitigation: separate perf interceptors from semantic interceptors; keep perf-only fastpaths with explicit label if needed.

## Assumptions and Defaults

               1. Assume existing dirty workspace may include unrelated parallel work; this plan avoids requiring cleanup of unrelated files.
               2. Default policy is semantic correctness over short-term throughput.
               3. Default is no edits in vendored uvm-core; if blocked by strict conformance conflict, document exception request explicitly before changing.
               4. Default CI requirement is parity on targeted UVM semantic suites with selected interceptor families disabled.

