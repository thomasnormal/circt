// RUN: circt-opt %s --lower-to-bmc="bound=1 top-module=top" | FileCheck %s

// If assertions carry only bmc.clock metadata (from LTL lowering) and there
// are no explicit seq.to_clock users or register clock metadata, lower-to-bmc
// must still discover the clock and insert a BMC clock input.
module {
  hw.module @top(in %clk : !hw.struct<value: i1, unknown: i1>) attributes {
    num_regs = 0 : i32,
    initial_values = []
  } {
    %true = hw.constant true
    verif.assert %true {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    hw.output
  }
}

// CHECK: verif.bmc
// CHECK: bmc_input_names = ["clk_0", "clk"]
// CHECK: ^bb0(%arg0: !seq.clock, %arg1: !hw.struct<value: i1, unknown: i1>):
// CHECK: verif.assert {{.*}} {bmc.clock = "clk_0", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
