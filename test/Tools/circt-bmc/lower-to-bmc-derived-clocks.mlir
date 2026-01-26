// RUN: circt-opt --lower-to-bmc="top-module=derived2 bound=4 allow-multi-clock=true" %s | FileCheck %s

// CHECK: verif.bmc
// CHECK: ^bb0([[CLK:%.+]]: !seq.clock, [[CLK_A:%.+]]: i1, [[CLK_B:%.+]]: i1):
// CHECK-DAG: comb.icmp eq {{%.+}}, [[CLK_A]]
// CHECK-DAG: comb.icmp eq {{%.+}}, [[CLK_B]]
// CHECK-DAG: verif.assume {{%.+}}
// CHECK-DAG: verif.assume {{%.+}}

hw.module @derived2(in %clk_a: i1, in %clk_b: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk0 = seq.to_clock %clk_a
  %clk1 = seq.to_clock %clk_b
  %val0 = seq.from_clock %clk0
  %val1 = seq.from_clock %clk1
  %and = comb.and %val0, %val1 : i1
  verif.assert %and : i1
}
