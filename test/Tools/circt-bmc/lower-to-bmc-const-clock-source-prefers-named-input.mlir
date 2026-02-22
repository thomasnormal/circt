// RUN: circt-opt --lower-to-bmc="top-module=m bound=2 allow-multi-clock=true" %s | FileCheck %s

module {
  // Unresolved register clock metadata can point at const0 even when the
  // module has a real top-level clock input. LowerToBMC should prefer the
  // named interface clock to avoid vacuous clocking.
  hw.module @m(in %clk : !hw.struct<value: i1, unknown: i1>, in %a : i1, in %b : i1, in %r_state : i1, out r_next : i1)
      attributes {
        num_regs = 1 : i32,
        initial_values = [false],
        bmc_reg_clocks = [""],
        bmc_reg_clock_sources = [{clock_key = "const0", invert = false}]
      } {
    hw.output %r_state : i1
  }
}

// CHECK: verif.bmc
// CHECK-SAME: bmc_clock_keys = ["port:clk"]
// CHECK-SAME: bmc_clock_sources = [{arg_index = 1 : i32, clock_pos = 0 : i32, invert = false}]
// CHECK-SAME: bmc_input_names = ["clk_0", "clk", "a", "b", "r_state"]
// CHECK-SAME: bmc_reg_clock_sources = [{clock_key = "port:clk", invert = false}]
// CHECK-SAME: bmc_reg_clocks = ["clk_0"]
// CHECK-NOT: bmc_clock_keys = ["const0"]
