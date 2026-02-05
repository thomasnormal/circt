// RUN: circt-verilog --ir-llhd --no-uvm-auto-include %s | \
// RUN:   circt-bmc -b 4 --assume-known-inputs --ignore-asserts-until=1 \
// RUN:   --module=sva_llhd_overlap_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_llhd_overlap_sat(input logic clk, input logic a);
  logic b;
  always @(posedge clk)
    b <= a;
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK: BMC_RESULT=SAT
