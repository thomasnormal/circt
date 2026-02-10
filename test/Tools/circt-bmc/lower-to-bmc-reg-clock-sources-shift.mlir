// RUN: circt-opt %s --externalize-registers="allow-multi-clock=true" --lower-to-bmc="top-module=top bound=2 allow-multi-clock=true" | FileCheck %s

// Derived clock inputs are prepended by lower-to-bmc. Ensure existing
// bmc_reg_clock_sources arg indices are shifted to the new input numbering.
// CHECK: verif.bmc bound 8 num_regs 2 initial_values [unit, unit] attributes {
// CHECK-SAME: bmc_clock_sources = [{arg_index = 2 : i32, clock_pos = 0 : i32, invert = false}, {arg_index = 3 : i32, clock_pos = 1 : i32, invert = false}]
// CHECK-SAME: bmc_input_names = ["clk_a_0", "clk_b_0", "clk_a", "clk_b", "in", "r0_state", "r1_state"]
// CHECK-SAME: bmc_reg_clock_sources = [{arg_index = 2 : i32, invert = false}, {arg_index = 3 : i32, invert = false}]

hw.module @top(in %clk_a : i1, in %clk_b : i1, in %in : i1) {
  %clk0 = seq.to_clock %clk_a
  %clk1 = seq.to_clock %clk_b
  %r0 = seq.compreg %in, %clk0 : i1
  %r1 = seq.compreg %r0, %clk1 : i1
  verif.assert %r1 : i1
  hw.output
}
