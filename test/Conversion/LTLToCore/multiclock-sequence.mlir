// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

hw.module @top(in %clk0: i1, in %clk1: i1, in %a: i1, in %b: i1) {
  %clk0c = seq.to_clock %clk0
  %seq_a = ltl.delay %a, 0 : i1
  %seq_a_clk = ltl.clock %seq_a, posedge %clk0 : !ltl.sequence
  %seq_b = ltl.delay %b, 0 : i1
  %seq_b_clk = ltl.clock %seq_b, posedge %clk1 : !ltl.sequence
  %seq = ltl.concat %seq_a_clk, %seq_b_clk : !ltl.sequence, !ltl.sequence
  verif.assert %seq : !ltl.sequence
  hw.output
}

// CHECK: %[[CLK0C:.*]] = seq.to_clock %clk0
// CHECK: %[[CLK1_PREV:.*]] = seq.compreg sym @ltl_past %clk1, %[[CLK0C]]
// CHECK: %[[NOT_CLK1_PREV:.*]] = comb.xor %[[CLK1_PREV]], {{.*}} : i1
// CHECK: %[[CLK1_TICK:.*]] = comb.and bin %clk1, %[[NOT_CLK1_PREV]] : i1
// CHECK: comb.and bin %b, %[[CLK1_TICK]] : i1
// CHECK: verif.assert {{.*}} : i1
