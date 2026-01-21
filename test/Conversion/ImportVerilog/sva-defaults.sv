// RUN: circt-verilog %s --parse-only | FileCheck %s

// CHECK-LABEL: moore.module @test_default
// CHECK: ltl.or
// CHECK: ltl.clock
module test_default(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);
  assert property (a |=> a);
endmodule
