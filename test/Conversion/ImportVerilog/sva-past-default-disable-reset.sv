// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_past_default_disable_reset(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $past(a, 2);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_disable_reset
// CHECK: moore.clocking_block
// CHECK: moore.past %a delay 2 : l1
// CHECK: comb.or
// CHECK: verif.clocked_assert
