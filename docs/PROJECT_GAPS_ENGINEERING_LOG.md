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
