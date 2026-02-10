// RUN: circt-opt --lower-to-bmc="top-module=top bound=1 allow-multi-clock=true" %s | FileCheck %s

// Ensure mixed top-level + struct-carried clocks are accepted in multiclock
// mode. Struct-carried clock uses are rewritten through a synthesized BMC
// clock, while explicit top-level clock uses remain valid.

// CHECK: verif.bmc
// CHECK: ^bb0([[SYNTH:%.+]]: !seq.clock, [[EXPLICIT:%.+]]: !seq.clock, [[CLK_STRUCT:%.+]]: !hw.struct<clk: !seq.clock>, [[SIG:%.+]]: i1):
// CHECK: [[EXPLICIT_I1:%.+]] = seq.from_clock [[EXPLICIT]]
// CHECK: ltl.clock {{.*}}, posedge [[EXPLICIT_I1]]{{.*}} : !ltl.sequence
// CHECK: [[STRUCT_CLK:%.+]] = hw.struct_extract [[CLK_STRUCT]]{{\[}}"clk"{{\]}}
// CHECK: [[STRUCT_I1:%.+]] = seq.from_clock [[STRUCT_CLK]]
// CHECK: [[SYNTH_I1:%.+]] = seq.from_clock [[SYNTH]]
// CHECK: [[EQ:%.+]] = comb.icmp eq [[SYNTH_I1]], [[STRUCT_I1]]
// CHECK: verif.assume [[EQ]]
// CHECK: [[SYNTH_I1_2:%.+]] = seq.from_clock [[SYNTH]]
// CHECK: ltl.clock {{.*}}, posedge [[SYNTH_I1_2]]{{.*}} : !ltl.sequence

hw.module @top(
    in %clk_explicit: !seq.clock,
    in %clk_struct: !hw.struct<clk: !seq.clock>,
    in %sig: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %explicit_i1 = seq.from_clock %clk_explicit
  %seq_explicit = ltl.delay %sig, 0, 0 : i1
  %clocked_explicit = ltl.clock %seq_explicit, posedge %explicit_i1 : !ltl.sequence
  verif.assert %clocked_explicit : !ltl.sequence

  %struct_clk = hw.struct_extract %clk_struct["clk"] : !hw.struct<clk: !seq.clock>
  %struct_i1 = seq.from_clock %struct_clk
  %seq_struct = ltl.delay %sig, 0, 0 : i1
  %clocked_struct = ltl.clock %seq_struct, posedge %struct_i1 : !ltl.sequence
  verif.assert %clocked_struct : !ltl.sequence
  hw.output
}
