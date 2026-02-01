// RUN: circt-verilog %s --parse-only | FileCheck %s

module test_value_change(input logic clk, a);
  assert property (@(posedge clk) $fell(a));
  assert property (@(posedge clk) $rose(a));
endmodule

// CHECK-LABEL: moore.module @test_value_change
// $fell(a) = !a && $past(a)
// CHECK: moore.past {{.*}} delay 1 : l1
// CHECK: moore.not {{.*}} : l1
// CHECK: moore.and {{.*}}, {{.*}} : l1
// $rose(a) = a && !$past(a)
// CHECK: moore.past {{.*}} delay 1 : l1
// CHECK: moore.not {{.*}} : l1
// CHECK: moore.and {{.*}}, {{.*}} : l1
