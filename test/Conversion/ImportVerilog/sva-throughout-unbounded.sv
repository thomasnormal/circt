// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module throughout_unbounded(input logic clk, a, b, c);
  // Unbounded RHS should still compute a minimum length for throughout.
  assert property (@(posedge clk) a throughout (b ##[2:$] c));
endmodule

// CHECK-LABEL: moore.module @throughout_unbounded
// CHECK: [[CONV_A:%.+]] = moore.to_builtin_bool %a : l1
// CHECK: [[REPEAT_A:%.+]] = ltl.repeat [[CONV_A]], 3 : i1
// CHECK: ltl.intersect [[REPEAT_A]]
