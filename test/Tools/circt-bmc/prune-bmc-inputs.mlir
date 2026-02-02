// RUN: circt-opt %s --externalize-registers --prune-bmc-registers | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %used: i1, in %unused: i1) {
  %reg = seq.compreg %used, %clk : i1
  verif.assert %reg : i1
  hw.output
}

// CHECK: hw.module @top(in %clk : !seq.clock, in %used : i1, in %reg_state : i1, out reg_next : i1) attributes {bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}], bmc_reg_clocks = ["clk"], initial_values = [unit], num_regs = 1 : i32} {
// CHECK: verif.assert %reg_state : i1
// CHECK-NOT: unused
// CHECK: hw.output %used
