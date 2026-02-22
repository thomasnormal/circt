// RUN: circt-opt --lower-to-bmc="top-module=m bound=2 allow-multi-clock=true" %s | FileCheck %s

module {
  // Register clock metadata may only carry clock_key/invert entries.
  // LowerToBMC must not assume arg_index exists on every source dict.
  hw.module @m(in %clk : i1, in %r_state : i1, out r_next : i1)
      attributes {
        num_regs = 1 : i32,
        initial_values = [false],
        bmc_reg_clocks = ["clk"],
        bmc_reg_clock_sources = [{clock_key = "port:clk", invert = false}]
      } {
    hw.output %r_state : i1
  }
}

// CHECK: verif.bmc
