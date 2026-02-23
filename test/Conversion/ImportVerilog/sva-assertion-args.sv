// RUN: circt-translate --import-verilog %s | FileCheck %s
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
  // CHECK: [[BASE:%.*]] = ltl.concat [[DA]], [[DB]] : !ltl.sequence, !ltl.sequence
  // CHECK: [[BASE_DELAY:%.*]] = ltl.delay [[BASE]], 0, 0 : !ltl.sequence
  // CHECK: [[C:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[DC:%.*]] = ltl.delay [[C]], 0, 0 : i1
  // CHECK: [[SEQ:%.*]] = ltl.concat [[BASE_DELAY]], [[DC]] : !ltl.sequence, !ltl.sequence
  // CHECK: [[CLK:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[CLOCKED:%.*]] = ltl.clock [[SEQ]],{{ *}}posedge [[CLK]]{{.*}} : !ltl.sequence
  // CHECK: verif.assert [[CLOCKED]] : !ltl.sequence
  assert property (@(posedge clk) seq_arg(base_seq(a, b), c));

  // CHECK: [[A2:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[B2:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[IMP:%.*]] = ltl.implication [[A2]], [[B2]] : i1, i1
  // CHECK: [[CLK2:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[CLOCKED2:%.*]] = ltl.clock [[IMP]],{{ *}}posedge [[CLK2]]{{.*}} : !ltl.property
  // CHECK: verif.assert [[CLOCKED2]] : !ltl.property
  assert property (@(posedge clk) prop_arg(base_prop(a, b)));
endmodule
