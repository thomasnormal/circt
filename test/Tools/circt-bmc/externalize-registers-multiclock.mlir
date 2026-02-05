// RUN: circt-opt --externalize-registers="allow-multi-clock" %s | FileCheck %s

// CHECK: hw.module @two_clks
// CHECK-SAME: bmc_reg_clocks = ["clk0", "clk1"]
// CHECK-SAME: initial_values = [unit, unit]
// CHECK-SAME: num_regs = 2

hw.module @two_clks(in %clk0: !seq.clock, in %clk1: !seq.clock, in %in: i32, out out: i32) {
  %1 = seq.compreg %in, %clk0 : i32
  %2 = seq.compreg %1, %clk1 : i32
  hw.output %2 : i32
}
