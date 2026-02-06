// RUN: circt-verilog %s --parse-only | FileCheck %s

module within_unbounded(input logic clk, a, b, c);
  assert property (@(posedge clk) (a ##1 b) within (c ##[2:$] c));
endmodule

// CHECK-LABEL: moore.module @within_unbounded
// CHECK: ltl.repeat {{%.*}}, 0 : i1
// CHECK: ltl.delay {{%.*}}, 1, 0 : !ltl.sequence
// CHECK: ltl.concat
// CHECK: ltl.intersect
