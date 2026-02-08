// RUN: circt-bmc -b 1 --emit-mlir --module top %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NOT: 4-state inputs are unconstrained

// A 4-state input used only as a clock source is now constrained to known by
// default, so the generic unconstrained 4-state input warning should not fire.
hw.module @top(in %clk: !hw.struct<value: i1, unknown: i1>) {
  %true = hw.constant true
  %value = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
  %not_unknown = comb.xor %unknown, %true : i1
  %clk_i1 = comb.and bin %value, %not_unknown : i1
  %past_clk = ltl.delay %clk_i1, 1, 0 : i1
  %prop = ltl.implication %clk_i1, %past_clk : i1, !ltl.sequence
  verif.clocked_assert %prop, posedge %clk_i1 : !ltl.property
  hw.output
}
