// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test bounded within/throughout/intersect sequence composition.
// IEEE 1800-2017 §16.9.9: throughout requires a simple expression (not a
// sequence) on the left-hand side.
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
  assert property (a throughout s2);
  assert property (s1 intersect s2);
endmodule

// CHECK-LABEL: moore.module @sva_within_throughout_intersect
// CHECK: ltl.intersect
// CHECK: ltl.repeat {{.*}}, 2, 0
// CHECK: ltl.intersect
// CHECK: ltl.intersect
