// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @from_clock_gate(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %clk_i1 = seq.from_clock %clk
  %true = hw.constant true
  %gate = comb.and %clk_i1, %true : i1
  %gated_clk = seq.to_clock %gate
  %reg = seq.compreg %in, %gated_clk : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @from_clock_gate(
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: in %in : i1
// CHECK-SAME: in %reg_state : i1
// CHECK-SAME: out out : i1
// CHECK-SAME: out reg_next : i1
// CHECK-SAME: attributes {{{.*}}bmc_reg_clocks = ["clk"]
// CHECK-SAME: initial_values = [unit], num_regs = 1 : i32}
// CHECK-NOT: seq.compreg
