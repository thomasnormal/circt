// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_changed_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_changed_sat(input logic clk);
  logic sig;
  initial sig = 1'b1;
  assert property (@(posedge clk) $changed(sig));
endmodule

// CHECK: BMC_RESULT=SAT
