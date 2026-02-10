// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// In single-clock mode, multiple explicit clock ports are allowed when only
// one explicit clock domain is actually used by clocked logic.

// CHECK: verif.bmc
// CHECK: ^bb0([[CLK0:%.+]]: !seq.clock, [[CLK1:%.+]]: !seq.clock, [[SIG:%.+]]: i1):
// CHECK: [[CLK0_I1:%.+]] = seq.from_clock [[CLK0]]
// CHECK: ltl.clock {{.*}}, posedge [[CLK0_I1]]{{.*}} : !ltl.sequence

hw.module @top(in %clk0: !seq.clock, in %clk1: !seq.clock, in %sig: i1)
    attributes {num_regs = 0 : i32, initial_values = []} {
  %clk0_i1 = seq.from_clock %clk0
  %seq = ltl.delay %sig, 0, 0 : i1
  %clocked = ltl.clock %seq, posedge %clk0_i1 : !ltl.sequence
  verif.assert %clocked : !ltl.sequence
  hw.output
}
