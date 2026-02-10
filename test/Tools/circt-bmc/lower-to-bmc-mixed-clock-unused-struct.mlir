// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// In single-clock mode, a mixed explicit + struct clock input should not fail
// when the struct-carried clock domain is unused.

// CHECK: verif.bmc
// CHECK: ^bb0([[EXPLICIT:%.+]]: !seq.clock, [[CLK_STRUCT:%.+]]: !hw.struct<clk: !seq.clock>, [[SIG:%.+]]: i1):
// CHECK: [[EXPLICIT_I1:%.+]] = seq.from_clock [[EXPLICIT]]
// CHECK: ltl.clock {{.*}}, posedge [[EXPLICIT_I1]]{{.*}} : !ltl.sequence
// CHECK-NOT: bmc_clock_sources
// CHECK-NOT: verif.assume

hw.module @top(
    in %clk_explicit: !seq.clock,
    in %clk_struct: !hw.struct<clk: !seq.clock>,
    in %sig: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk_i1 = seq.from_clock %clk_explicit
  %seq = ltl.delay %sig, 0, 0 : i1
  %clocked = ltl.clock %seq, posedge %clk_i1 : !ltl.sequence
  verif.assert %clocked : !ltl.sequence
  hw.output
}
