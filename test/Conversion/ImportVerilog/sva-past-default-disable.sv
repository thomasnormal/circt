// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module test_past_default_disable(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $past(a, 2);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_disable
// CHECK: moore.past
// CHECK: comb.or
// CHECK: verif.clocked_assert
