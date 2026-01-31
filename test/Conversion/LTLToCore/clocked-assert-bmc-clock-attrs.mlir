// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @clocked_attrs(in %clk : i1, in %a : i1) {
    %seq = ltl.delay %a, 0, 0 : i1
    %prop = ltl.implication %a, %seq : i1, !ltl.sequence
    verif.clocked_assert %prop, negedge %clk : !ltl.property
    hw.output
  }
}

// CHECK-LABEL: hw.module @clocked_attrs
// CHECK: verif.assert {{.*}}bmc.clock = "clk"{{.*}}bmc.clock_edge = #ltl<clock_edge negedge>
// CHECK: verif.assert {{.*}}bmc.clock = "clk"{{.*}}bmc.clock_edge = #ltl<clock_edge negedge>{{.*}}bmc.final
