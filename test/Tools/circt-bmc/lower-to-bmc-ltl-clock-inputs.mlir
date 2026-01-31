// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// Ensure ltl.clock operands are rewritten to the derived BMC clock input.

// CHECK: verif.bmc
// CHECK: ^bb0([[CLK:%.+]]: !seq.clock, [[CLK_IN:%.+]]: i1, [[SIG:%.+]]: i1):
// CHECK: verif.assume
// CHECK: ltl.delay
// CHECK: [[FROM_CLK:%.+]] = seq.from_clock [[CLK]]
// CHECK: ltl.clock {{.*}}, posedge [[FROM_CLK]] : !ltl.sequence
// CHECK-NOT: ltl.clock {{.*}}, posedge [[CLK_IN]]

hw.module @top(in %clk_i1: i1, in %sig: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk = seq.to_clock %clk_i1
  %clk_val = seq.from_clock %clk
  verif.assert %clk_val : i1
  %seq = ltl.delay %sig, 0, 0 : i1
  %clocked = ltl.clock %seq, posedge %clk_i1 : !ltl.sequence
  verif.assert %clocked : !ltl.sequence
  hw.output
}
