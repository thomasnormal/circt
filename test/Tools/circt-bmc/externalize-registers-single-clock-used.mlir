// RUN: circt-opt --externalize-registers %s | FileCheck %s

// In single-clock mode, modules may have multiple clock ports as long as all
// externalized registers use the same clock source.

hw.module @same_clock_regs(in %clk0: !seq.clock, in %clk1: !seq.clock,
                           in %in: i32, out out: i32) {
  %r0 = seq.compreg %in, %clk0 : i32
  %r1 = seq.compreg %r0, %clk0 : i32
  hw.output %r1 : i32
}

// CHECK-LABEL: hw.module @same_clock_regs(
// CHECK: bmc_reg_clock_sources =
// CHECK: arg_index = 0 : i32, invert = false
// CHECK: arg_index = 0 : i32, invert = false
// CHECK: bmc_reg_clocks = ["clk0", "clk0"]
// CHECK: initial_values = [unit, unit]
// CHECK: num_regs = 2
