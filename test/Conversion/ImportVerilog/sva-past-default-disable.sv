// RUN: circt-verilog %s --parse-only | FileCheck %s

module test_past_default_disable(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $past(a, 2);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_disable
// CHECK-NOT: moore.past
// CHECK: moore.procedure always
// CHECK: moore.wait_event
// CHECK: moore.blocking_assign
