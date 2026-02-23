// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @test_default
// CHECK: ltl.or
// CHECK: verif.clocked_assert
module test_default(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);
  assert property (a |=> a);
endmodule
