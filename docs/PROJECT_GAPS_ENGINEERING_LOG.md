# Project Gaps Engineering Log

## 2026-02-27

### ImportVerilog: unconnected inout instance port
- Repro:
  - `module child(inout wire io); endmodule`
  - `module top; child u0(.io()); endmodule`
  - `circt-verilog --ir-moore` failed with `unsupported port 'io' (Port)`.
- Root cause:
  - Unconnected-port materialization handled `In` by synthesizing a placeholder net/var and reading it, but `InOut` fell through to unsupported.
- Fix:
  - For unconnected `InOut`, synthesize the same placeholder net/var and keep it as an lvalue ref (no `moore.read`), then wire that into `portValues`.
- Tests:
  - Added `test/Conversion/ImportVerilog/unconnected-inout-instance-port.sv`.
  - Verified with `llvm-lit` filter for that test and direct `circt-verilog` repro.

### ImportVerilog: capture module-scope interface refs in task/function regions
- Repro:
  - Access a module-scope interface instance (`vif.sig`) from an `automatic` task.
  - Lowering could produce `moore.read` that uses a value defined outside the task
    function region, violating region isolation.
- Root cause:
  - Interface instance refs used by hierarchical member access were read directly
    without ensuring they were captured into the isolated function/task region.
- Fix:
  - Call `context.captureRef(...)` before reading those interface refs in both
    relevant hierarchical access paths.
- Tests:
  - Added `test/Conversion/ImportVerilog/task-interface-instance-capture.sv`.
  - Verified with focused `llvm-lit` run for that test.

### UVM: re-enable `wait_for_state` compile-time regression
- Repro:
  - `test/Runtime/uvm/uvm_phase_wait_for_state_test.sv` was disabled with
    `UNSUPPORTED: true` and did not exercise compile-time API availability.
- Fix:
  - Converted the test to a real parse-only lit check using
    `--uvm-path=%S/../../../lib/Runtime/uvm-core`.
- Tests:
  - `llvm-lit -sv ... --filter 'Runtime/uvm/uvm_phase_wait_for_state_test.sv'`
- Note:
  - `uvm_phase_aliases_test.sv` still fails today on `final_ph` undeclared when
    run against upstream `uvm-core`; fixing that requires a coordinated
    submodule update rather than a superproject-only patch.

### MooreToCore: add explicit `always_comb` / `always_latch` lowering coverage
- Repro:
  - Existing MooreToCore coverage had TODO commentary for `always_comb` and
    `always_latch`, but no focused regression proving the implicit wait-loop
    lowering shape.
- Fix:
  - Added `test/Conversion/MooreToCore/procedure-always-comb-latch.mlir`.
  - The test checks that both procedure kinds lower to `llhd.process` with
    a body block and a wait block (`llhd.wait`) that loops back to the body.
- Tests:
  - `llvm-lit -sv ... --filter 'Conversion/MooreToCore/procedure-always-comb-latch.mlir'`
