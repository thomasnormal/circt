// RUN: circt-verilog --no-uvm-auto-include %s --parse-only | FileCheck %s

module test_value_change(input logic clk, a);
  assert property (@(posedge clk) $fell(a));
  assert property (@(posedge clk) $rose(a));
endmodule

// CHECK-LABEL: moore.module @test_value_change
// $fell(a) = !a_cur && a_prev (explicit register tracking)
// CHECK: moore.procedure always
// CHECK:   moore.not
// CHECK:   moore.and
// CHECK:   moore.blocking_assign
// $rose(a) = a_cur && !a_prev
// CHECK: moore.procedure always
// CHECK:   moore.not
// CHECK:   moore.and
// CHECK:   moore.blocking_assign
