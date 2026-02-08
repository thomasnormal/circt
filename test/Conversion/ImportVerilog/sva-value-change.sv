// RUN: circt-verilog --no-uvm-auto-include %s --parse-only | FileCheck %s

module test_value_change(input logic clk, a);
  assert property (@(posedge clk) $fell(a));
  assert property (@(posedge clk) $rose(a));
endmodule

// CHECK-LABEL: moore.module @test_value_change
// $fell(a) = !a_cur && a_prev
// CHECK: moore.past
// CHECK:   moore.not
// CHECK:   moore.and
// $rose(a) = a_cur && !a_prev
// CHECK: moore.past
// CHECK:   moore.not
// CHECK:   moore.and
