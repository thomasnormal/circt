// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_stable_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_stable_unsat(input logic clk);
  logic sig;
  initial sig = 1'b1;
  assert property (@(posedge clk) $stable(sig));
endmodule

// CHECK: BMC_RESULT=UNSAT
