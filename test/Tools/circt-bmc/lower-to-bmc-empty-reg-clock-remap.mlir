// RUN: circt-opt --lower-to-bmc="top-module=m bound=2 allow-multi-clock=true" %s | FileCheck %s

module {
  // Externalized register metadata may carry an empty bmc_reg_clocks entry
  // when the source clock name is unresolved. In the single-derived-clock
  // case, lower-to-bmc should remap that entry to the inserted BMC clock.
  hw.module @m(in %clk : i1, in %r_state : i1, out r_next : i1) attributes {num_regs = 1 : i32, initial_values = [false], bmc_reg_clocks = [""], bmc_reg_clock_sources = [{clock_key = "expr:test", invert = false}]} {
    %c = seq.to_clock %clk
    %t = hw.constant true
    verif.assert %r_state : i1
    hw.output %t : i1
  }
}

// CHECK: verif.bmc
// CHECK-SAME: bmc_reg_clock_sources = [{clock_key = "port:clk", invert = false}]
// CHECK-SAME: bmc_reg_clocks = ["clk_0"]
// CHECK-NOT: bmc_reg_clocks = [""]
