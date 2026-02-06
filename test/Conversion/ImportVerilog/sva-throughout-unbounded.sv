// RUN: circt-verilog %s --parse-only | FileCheck %s

module throughout_unbounded(input logic clk, a, b, c);
  // Unbounded RHS should still compute a minimum length for throughout.
  assert property (@(posedge clk) a throughout (b ##[2:$] c));
endmodule

// CHECK-LABEL: moore.module @throughout_unbounded
// CHECK: [[READ_A:%.+]] = moore.read %{{[a-zA-Z0-9_]+}} : <l1>
// CHECK: [[CONV_A:%.+]] = moore.to_builtin_bool [[READ_A]] : l1
// CHECK: [[REPEAT_A:%.+]] = ltl.repeat [[CONV_A]], 3 : i1
// CHECK: ltl.intersect [[REPEAT_A]]
