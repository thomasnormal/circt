// RUN: circt-opt %s --externalize-registers --prune-bmc-registers | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  %reg_keep_a = seq.compreg %in, %clk : i1
  %reg_keep_b = seq.compreg %reg_keep_a, %clk : i1
  %reg_drop = seq.compreg %in, %clk : i1
  verif.assert %reg_keep_b : i1
  hw.output
}

// CHECK: hw.module @top(in %clk : !seq.clock, in %in : i1, in %reg_keep_a_state : i1, in %reg_keep_b_state : i1, out reg_keep_a_next : i1, out reg_keep_b_next : i1) attributes {bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}, {arg_index = 0 : i32, invert = false}, {arg_index = 0 : i32, invert = false}], bmc_reg_clocks = ["clk", "clk"], initial_values = [unit, unit], num_regs = 2 : i32} {
// CHECK: verif.assert %reg_keep_b_state : i1
// CHECK: hw.output %in, %reg_keep_a_state
// CHECK-NOT: reg_drop
