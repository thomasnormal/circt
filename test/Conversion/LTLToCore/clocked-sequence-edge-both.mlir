// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @clocked_sequence_edge_both(in %clk : i1, in %a : i1) {
    %seq = ltl.concat %a, %a : i1, i1
    verif.clocked_assert %seq, edge %clk : !ltl.sequence
    verif.clocked_assume %seq, edge %clk : !ltl.sequence
    verif.clocked_cover %seq, edge %clk : !ltl.sequence
    hw.output
  }
}

// CHECK-LABEL: hw.module @clocked_sequence_edge_both
// CHECK: seq.to_clock %clk
// CHECK: verif.assert {{.*}}bmc.clock_edge = #ltl<clock_edge edge>
// CHECK: verif.assert {{.*}}bmc.final
// CHECK: verif.assume {{.*}}bmc.clock_edge = #ltl<clock_edge edge>
// CHECK: verif.assume {{.*}}bmc.final
// CHECK: verif.cover {{.*}}bmc.clock_edge = #ltl<clock_edge edge>
// CHECK: verif.cover {{.*}}bmc.final
