// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_local_var_disable_iff_abort_unsat(input logic clk);
  logic [3:0] in;
  logic [3:0] out;
  logic start;
  // Always-disabled property: disable iff should vacuously suppress checks.
  logic reset;

  assign in = 4'd5;
  assign out = 4'd0;
  assign start = 1'b1;
  assign reset = 1'b1;

  property p;
    logic [3:0] x;
    @(posedge clk) disable iff (reset) (start, x = in) |-> ##1 (x == x + 4'd1);
  endproperty

  assert property (p);
endmodule

// JIT: BMC_RESULT=UNSAT
// SMTLIB: BMC_RESULT=UNSAT
