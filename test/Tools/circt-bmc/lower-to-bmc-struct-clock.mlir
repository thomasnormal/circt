// RUN: circt-opt --lower-to-bmc="top-module=top bound=2 allow-multi-clock=true" %s | FileCheck %s

// CHECK: verif.bmc
// CHECK: ^bb0([[CLK:%.+]]: !seq.clock, [[CLK_IN:%.+]]: !hw.struct<value: i1, unknown: i1>, [[STATE:%.+]]: i1):
// CHECK: [[VAL:%.+]] = hw.struct_extract [[CLK_IN]]{{\[}}"value"{{\]}}
// CHECK: [[UNK:%.+]] = hw.struct_extract [[CLK_IN]]{{\[}}"unknown"{{\]}}
// CHECK: [[NOTUNK:%.+]] = comb.xor [[UNK]], {{%.+}} : i1
// CHECK: [[CLKVAL:%.+]] = comb.and [[VAL]], [[NOTUNK]] : i1
// CHECK: [[FROM:%.+]] = seq.from_clock [[CLK]]
// CHECK: [[EQ:%.+]] = comb.icmp eq [[FROM]], [[CLKVAL]]
// CHECK: verif.assume [[EQ]]

hw.module @top(in %clk: !hw.struct<value: i1, unknown: i1>, in %reg_state: i1, out reg_next: i1)
    attributes {num_regs = 1 : i32, initial_values = [false], bmc_reg_clocks = ["clk"]} {
  hw.output %reg_state : i1
}
