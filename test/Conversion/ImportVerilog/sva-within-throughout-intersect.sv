// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test bounded within/throughout/intersect sequence composition.
module sva_within_throughout_intersect(
    input logic a,
    input logic b,
    input logic c,
    input logic d);
  sequence s1;
    a ##1 b;
  endsequence

  sequence s2;
    c ##1 d;
  endsequence

  assert property (s1 within s2);
  assert property (s1 throughout s2);
  assert property (s1 intersect s2);
endmodule

// CHECK-LABEL: moore.module @sva_within_throughout_intersect
// CHECK-DAG: ltl.concat
// CHECK-DAG: ltl.concat
// CHECK-DAG: ltl.repeat {{.*}}, 2, 0
// CHECK-COUNT-3: ltl.intersect
