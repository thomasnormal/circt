// RUN: circt-opt --lower-to-bmc="top-module=m bound=2 allow-multi-clock=true" %s | FileCheck %s

module {
  // ExternalizeRegisters may leave unresolved register clock metadata as unit
  // entries. LowerToBMC must still materialize a valid BMC clock when there is
  // a single clock-like interface input.
  hw.module @m(in %clk : !hw.struct<value: i1, unknown: i1>, in %r_state : i1, out r_next : i1)
      attributes {
        num_regs = 1 : i32,
        initial_values = [false],
        bmc_reg_clocks = [""],
        bmc_reg_clock_sources = [unit]
      } {
    hw.output %r_state : i1
  }
}

// CHECK: verif.bmc
// CHECK-SAME: bmc_clock_keys = ["port:clk"]
// CHECK-SAME: bmc_clock_sources = [{arg_index = 1 : i32, clock_pos = 0 : i32, invert = false}]
// CHECK-SAME: bmc_input_names = ["clk_0", "clk", "r_state"]
// CHECK-SAME: bmc_reg_clock_sources = [unit]
// CHECK-SAME: bmc_reg_clocks = ["clk_0"]
// CHECK: init {
// CHECK: verif.yield %{{.+}} : !seq.clock
// CHECK: loop {
// CHECK: verif.yield %{{.+}} : !seq.clock
