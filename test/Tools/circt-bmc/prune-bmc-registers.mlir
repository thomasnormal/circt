// RUN: circt-opt %s --externalize-registers --prune-bmc-registers | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  %next_keep = comb.xor %in, %in : i1
  %keep_reg = seq.compreg %next_keep, %clk : i1
  %drop_reg = seq.compreg %in, %clk : i1
  %and = comb.and %keep_reg, %in : i1
  verif.assert %and : i1
  hw.output
}

// CHECK: hw.module @top(in %clk : !seq.clock, in %in : i1, in %keep_reg_state : i1, out keep_reg_next : i1) attributes {bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}], bmc_reg_clocks = ["clk"], initial_values = [unit], num_regs = 1 : i32} {
// CHECK: [[NEXT:%.+]] = comb.xor %in, %in
// CHECK: hw.output [[NEXT]]
// CHECK-NOT: drop_reg_state
// CHECK-NOT: drop_reg_next
