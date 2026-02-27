# Scoped Gap Execution Board

Scope:
- ImportVerilog
- MooreToCore
- circt-sim runtime (non-AOT)
- UVM

Out of scope:
- FIRRTL, ExportVerilog, ESI, PyCDE, Arc, circt-mut
- scanner/report tooling

## Completed This Iteration

1. `circt-sim` syscall regression headers normalized from stale TODO/Bug phrasing to regression intent wording.
Files:
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
Validation:
- All 11 tests pass with `build_clang_test/bin/circt-verilog` + `build_clang_test/bin/circt-sim` + `FileCheck`.

2. MooreToCore stale XFAIL retired.
File:
- `test/Conversion/MooreToCore/interface-timing-after-inlining.sv`
Change:
- Removed `XFAIL: *`.
- Replaced stale FIXME with regression-intent comment.
Validation:
- `build_clang_test/bin/circt-verilog --ir-hw ... | FileCheck ...` passes.

3. ImportVerilog parity check against Xcelium for sampled `$past` control on non-value objects.
Case:
- `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv`
Result:
- `xrun` rejects `$past(cg, ...)` with: classes/queues/strings/dynamic arrays unsupported as sampled-value argument.
- CIRCT strict mode also rejects this case.
Conclusion:
- Keep this diagnostic path as expected parity behavior, not as a feature-to-implement item.

## Active Backlog (Prioritized)

## P0 ImportVerilog

1. Sampled-value SVA coverage where Xcelium accepts semantics but CIRCT still rejects.
Primary file:
- `lib/Conversion/ImportVerilog/AssertionExpr.cpp`
Tests to expand:
- `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv`
- `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv`

2. Cross-select non-constant and nested form handling.
Primary file:
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`
Tests to expand:
- `test/Conversion/ImportVerilog/cross-select-*.sv`

3. Timing control kind coverage.
Primary file:
- `lib/Conversion/ImportVerilog/TimingControls.cpp`
Tests to expand:
- `test/Conversion/ImportVerilog/delay-*-supported.sv`
- add new event-control semantic regressions.

## P1 MooreToCore

1. Convert remaining TODO-marked behavior tests into passing semantic checks or explicit expected-unsupported policy tests.
Primary file:
- `lib/Conversion/MooreToCore/MooreToCore.cpp`
Existing evidence tests:
- `test/Conversion/MooreToCore/basic.mlir`
- `test/Conversion/MooreToCore/interface-timing-after-inlining.sv` (now non-XFAIL)

## P2 circt-sim runtime (non-AOT)

1. Re-validate each remaining runtime TODO marker with current binaries before implementation.
Primary file:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`

2. Promote only still-reproducible mismatches into implementation tickets.

## P3 UVM

1. Ensure supported flows never hit base TLM "not implemented" stubs.
Primary files:
- `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh`
- `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh`
- `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh`

2. Resolve lifecycle stubs (`suspend`/`resume`) for supported runtime profile.
Primary file:
- `lib/Runtime/uvm-core/src/base/uvm_component.svh`

## Next Slice (in progress)

- Build first ImportVerilog P0 testcase where Xcelium accepts and CIRCT rejects sampled-value semantics.
- Add failing functional regression (not diagnostic-only).
- Validate expected semantics in `xrun`.
- Implement minimal lowering change and rerun both toolchains.
