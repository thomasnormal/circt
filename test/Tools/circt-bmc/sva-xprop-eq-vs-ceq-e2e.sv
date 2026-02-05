// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_eq_sat - | FileCheck %s --check-prefix=EQ
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_ceq_unsat - | FileCheck %s --check-prefix=CEQ
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_eq_sat(input logic clk);
  logic [1:0] in;
  assign in = 2'bx1;
  // == yields X on unknown inputs, so the equality with 0 is symbolic.
  assert property (@(posedge clk) ((in == 2'b01) == 1'b0));
endmodule

module sva_xprop_ceq_unsat(input logic clk);
  logic [1:0] in;
  assign in = 2'bx1;
  // === treats X/Z as values, so this is definitively false.
  assert property (@(posedge clk) ((in === 2'b01) == 1'b0));
endmodule

// EQ: BMC_RESULT=SAT
// CEQ: BMC_RESULT=UNSAT
