// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module m(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk or posedge clk) (a |-> b));
endmodule

// CHECK-LABEL: moore.module @m
// CHECK-NOT: ltl.or
// CHECK: verif.clocked_assert
