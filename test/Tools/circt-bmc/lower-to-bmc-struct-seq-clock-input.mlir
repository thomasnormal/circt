// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// Ensure struct-carried seq.clock inputs are lowered through the derived-clock
// path instead of being rejected.

// CHECK: verif.bmc
// CHECK: ^bb0([[BMC_CLK:%.+]]: !seq.clock, [[CLK_STRUCT:%.+]]: !hw.struct<clk: !seq.clock>, [[SIG:%.+]]: i1):
// CHECK: [[EXTRACT:%.+]] = hw.struct_extract [[CLK_STRUCT]]{{\[}}"clk"{{\]}}
// CHECK: [[CLK_I1:%.+]] = seq.from_clock [[EXTRACT]]
// CHECK: [[FROM_BMC:%.+]] = seq.from_clock [[BMC_CLK]]
// CHECK: [[EQ:%.+]] = comb.icmp eq [[FROM_BMC]], [[CLK_I1]]
// CHECK: verif.assume [[EQ]]
// CHECK: [[FROM_BMC_2:%.+]] = seq.from_clock [[BMC_CLK]]
// CHECK: ltl.clock {{.*}}, posedge [[FROM_BMC_2]]{{.*}} : !ltl.sequence
// CHECK-NOT: ltl.clock {{.*}}, posedge [[CLK_I1]]

hw.module @top(in %clk_struct: !hw.struct<clk: !seq.clock>, in %sig: i1)
    attributes {num_regs = 0 : i32, initial_values = []} {
  %clk = hw.struct_extract %clk_struct["clk"] : !hw.struct<clk: !seq.clock>
  %clk_i1 = seq.from_clock %clk
  %seq = ltl.delay %sig, 0, 0 : i1
  %clocked = ltl.clock %seq, posedge %clk_i1 : !ltl.sequence
  verif.assert %clocked : !ltl.sequence
  hw.output
}
