// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module test_past_default_clocking(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking

  property p;
    $past(a, 2, reset);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_clocking
// CHECK-NOT: moore.past
// CHECK: moore.procedure always
// CHECK: moore.wait_event
// CHECK: moore.blocking_assign
