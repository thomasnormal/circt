// RUN: circt-opt %s --externalize-registers --lower-to-bmc="bound=1 top-module=top" | FileCheck %s

module {
  hw.module @top(in %clk : i1, in %a : i1) {
    %true = hw.constant true
    %sel = comb.xor %clk, %true : i1
    %gate = comb.mux %sel, %clk, %true : i1
    %clock = seq.to_clock %gate
    %reg = seq.compreg %a, %clock : i1
    verif.assert %a {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    hw.output
  }
}

// CHECK: verif.bmc
// CHECK: bmc_input_names = ["bmc_clock", "clk", "a"
// CHECK: verif.assert {{.*}}bmc.clock = "bmc_clock"
