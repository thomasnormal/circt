// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module m(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk or posedge clk) (a |-> b));
endmodule

// CHECK-LABEL: moore.module @m
// CHECK: ltl.clock
// CHECK-NOT: ltl.clock
// CHECK-NOT: ltl.or
// CHECK: verif.assert
