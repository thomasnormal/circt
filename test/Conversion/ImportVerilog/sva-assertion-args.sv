// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test SVA sequence/property arguments.

// CHECK-LABEL: moore.module @sva_assertion_args
module sva_assertion_args(
  input logic clk,
  input logic a,
  input logic b,
  input logic c
);
  sequence base_seq(logic x, logic y);
    x ##1 y;
  endsequence

  sequence seq_arg(sequence s, logic z);
    s ##1 z;
  endsequence

  property base_prop(logic x, logic y);
    x |-> y;
  endproperty

  property prop_arg(property p);
    p;
  endproperty

  // CHECK: [[A:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[DA:%.*]] = ltl.delay [[A]], 0, 0 : i1
  // CHECK: [[B:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[DB:%.*]] = ltl.delay [[B]], 0, 0 : i1
  // CHECK: [[C:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[DC:%.*]] = ltl.delay [[C]], 0, 0 : i1
  // CHECK: [[SEQ:%.*]] = ltl.concat [[DA]], [[DB]], [[DC]] : !ltl.sequence, !ltl.sequence, !ltl.sequence
  // CHECK: [[CLK:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: verif.clocked_assert [[SEQ]], posedge [[CLK]] : !ltl.sequence
  assert property (@(posedge clk) seq_arg(base_seq(a, b), c));

  // CHECK: [[IMP:%.*]] = ltl.implication [[A]], [[B]] : i1, i1
  // CHECK: verif.clocked_assert [[IMP]], posedge [[CLK]] : !ltl.property
  assert property (@(posedge clk) prop_arg(base_prop(a, b)));
endmodule
