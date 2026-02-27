# Scoped Gap Fix Plan (ImportVerilog, MooreToCore, Sim Runtime, UVM)

This plan is intentionally scoped to:
- ImportVerilog
- MooreToCore
- circt-sim runtime (interpreter/runtime semantics only; AOT excluded)
- UVM runtime

Out of scope for this plan:
- FIRRTL, ExportVerilog, ESI, PyCDE, Arc, circt-mut
- scanner/writeup tooling changes

## 1. Triage Baseline

Source: `docs/PROJECT_GAPS_MANUAL_WRITEUP.md` (scoped extraction only).

Approximate scoped volume from manual audit:
- ImportVerilog: 322 entries (many duplicates around a few capability clusters)
- UVM: 117 entries
- Sim runtime: 102 entries
- MooreToCore: 39 entries

Important normalization rule:
- Treat test-fixture/oracle lines as evidence, not backlog items.
- Track implementation work by feature cluster and owning source file(s), not by raw line count.

## 2. Work Already Started

Validated runtime regressions that are currently green (using `build_clang_test/bin/circt-verilog` + `circt-sim` + `FileCheck`):
- `test/Tools/circt-sim/syscall-strobe.sv`
- `test/Tools/circt-sim/syscall-shortrealtobits.sv`
- `test/Tools/circt-sim/syscall-randomize-with.sv`
- `test/Tools/circt-sim/syscall-random.sv`
- `test/Tools/circt-sim/syscall-monitor.sv`
- `test/Tools/circt-sim/syscall-isunbounded.sv`
- `test/Tools/circt-sim/syscall-generate.sv`
- `test/Tools/circt-sim/syscall-fread.sv`
- `test/Tools/circt-sim/syscall-feof.sv`
- `test/Tools/circt-sim/syscall-ungetc.sv`
- `test/Tools/circt-sim/syscall-save-restart-warning.sv`

These tests had stale "TODO bug" headers; comments were updated to regression-style intent text.

## 3. Priority Waves

### Wave P0: ImportVerilog Semantic Coverage

#### P0.1 SVA sampled-value and `$past` controls
Files:
- `lib/Conversion/ImportVerilog/AssertionExpr.cpp`
- `lib/Conversion/ImportVerilog/Statements.cpp`

Focus:
- Unsupported sampled-value type classes for `$stable/$rose/$fell/$past`
- Event/sample-control forms currently routed to unsupported/placeholder paths
- Continue-on-unsupported placeholder paths should shrink as real lowering expands

Tests:
- Existing `test/Conversion/ImportVerilog/sva-*continue-on-unsupported*.sv`
- Add functional semantics regressions (not only diagnostics)
- For each new regression: run equivalent `xrun` test and compare outcomes

Exit criteria:
- Reduced unsupported diagnostics for sampled-value controls
- New regressions assert behavior (value/temporal semantics), not only parse/diagnostic text

#### P0.2 Cross-select/intersect generalization
Files:
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`

Focus:
- Non-constant intersect range/value forms
- Nested with/set/negation limitations
- Tuple/container shape restrictions and domain-expansion hard failures

Tests:
- Extend `cross-select-*supported/unsupported*.sv` with semantic checks
- Add bounded functional tests for expressions currently rejected
- Validate interpretation with `xrun`

Exit criteria:
- Non-constant intersect support expanded for common legal forms
- Unsupported set narrowed to truly unsupported or pathological forms

#### P0.3 Timing/event-control coverage
Files:
- `lib/Conversion/ImportVerilog/TimingControls.cpp`

Focus:
- Unsupported timing-control kinds and non-canonical event forms
- Global/default clocking symbol-shape restrictions

Tests:
- Add focused event-control lowering regressions with behavior checks
- Cross-validate event semantics against `xrun`

Exit criteria:
- Fewer fallthrough unsupported timing-control diagnostics
- Event-control behavior parity for covered forms

### Wave P1: MooreToCore Core Semantic Gaps

#### P1.1 `wait_event/detect_event` lowering in `func.func` paths
Files:
- `lib/Conversion/MooreToCore/MooreToCore.cpp`

Evidence:
- `test/Conversion/MooreToCore/interface-timing-after-inlining.sv` (`XFAIL`, explicit FIXME)

Tests:
- Keep current reproducer, convert from XFAIL when fixed
- Add a second minimal case for both `posedge` and `negedge` through interface tasks

Exit criteria:
- No residual `moore.wait_event`/`moore.detect_event` after conversion in covered path
- XFAIL removed

#### P1.2 4-state and condition semantics
Files:
- `lib/Conversion/MooreToCore/MooreToCore.cpp`

Focus:
- two-valued-only conditional lowering caveat
- out-of-bounds extract semantics in 4-state contexts
- caseX/caseZ non-constant mask/value behavior

Tests:
- Add focused semantic regressions (value-level checks)
- Cross-check against `xrun`

Exit criteria:
- Correctness-first behavior on targeted 4-state corner cases

### Wave P2: Sim Runtime Remaining Open Set (non-AOT)

#### P2.1 System-call/runtime parity residuals
Files:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- related interpreter helpers

Focus:
- Re-verify which runtime TODOs remain truly open after current green test set
- Keep legacy/policy unsupported tasks explicit (`$getpattern`, legacy queue tasks) unless scope is expanded

Tests:
- Continue self-checking syscall regressions
- Prefer strengthening expected behavior checks over diagnostic-only checks

Exit criteria:
- Open runtime list reduced to explicit policy boundaries or newly tracked bugs with repro

### Wave P3: UVM Runtime Reliability

#### P3.1 Eliminate accidental dependence on base TLM stubs
Files:
- `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh`
- `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh`
- `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh`

Focus:
- Ensure supported flows hit concrete overrides, not base "not implemented" macros

Tests:
- Add/extend runtime tests to fail loud if stub macros are reached in supported paths

Exit criteria:
- Supported UVM flows demonstrate concrete implementation paths for required TLM APIs

#### P3.2 Component lifecycle gaps (`suspend/resume`)
Files:
- `lib/Runtime/uvm-core/src/base/uvm_component.svh`

Focus:
- Implement or explicitly hard-error unsupported lifecycle APIs

Tests:
- Add suspend/resume state-transition regression tests

Exit criteria:
- No warning-only stubs for suspend/resume in supported runtime profile

## 4. Execution Rules

- TDD mandatory: add failing unit/regression first, then fix.
- ImportVerilog semantic tests: validate with `xrun` for expected meaning.
- Prefer small PR-sized slices by feature cluster.
- Each completed slice updates `project_gap_audit_engineering_log.md` with:
  - repro test
  - root cause
  - fix summary
  - exact test command(s) and results

## 5. Immediate Next Tasks (starting now)

1. Build actionable ImportVerilog P0.1 ticket set from `AssertionExpr.cpp` sampled-value failures (`$past/$stable/$rose` family).
2. Add one new failing functional regression for sampled-value controls (value semantics, not just diagnostic).
3. Validate expected behavior in `xrun` for the same testcase.
4. Implement minimal lowering support for that testcase and re-run both toolchains.
