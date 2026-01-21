// RUN: circt-opt --lower-to-bmc="top-module=derived bound=4" %s | FileCheck %s

// CHECK: verif.bmc
// CHECK: ^bb0([[CLK:%.+]]: !seq.clock, [[CLK_IN:%.+]]: i1):
// CHECK: [[FROM:%.+]] = seq.from_clock [[CLK]]
// CHECK: [[EQ:%.+]] = comb.icmp eq [[FROM]], [[CLK_IN]]
// CHECK: verif.assume [[EQ]]

hw.module @derived(in %clk_in: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk = seq.to_clock %clk_in
  %clk_val = seq.from_clock %clk
  verif.assert %clk_val : i1
}
