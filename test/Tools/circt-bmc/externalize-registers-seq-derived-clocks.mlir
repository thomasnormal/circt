// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @seq_gate(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %true = hw.constant true
  %gated = seq.clock_gate %clk, %true
  %reg = seq.compreg %in, %gated : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @seq_gate(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}]
// CHECK-SAME: bmc_reg_clocks = ["clk"]
// CHECK-NOT: seq.compreg

hw.module @seq_mux_const(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %true = hw.constant true
  %mux = seq.clock_mux %true, %clk, %clk
  %reg = seq.compreg %in, %mux : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @seq_mux_const(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}]
// CHECK-SAME: bmc_reg_clocks = ["clk"]
// CHECK-NOT: seq.compreg

hw.module @seq_inv(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %inv = seq.clock_inv %clk
  %reg = seq.compreg %in, %inv : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @seq_inv(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = true}]
// CHECK-SAME: bmc_reg_clocks = ["clk"]
// CHECK-NOT: seq.compreg

hw.module @seq_div(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %div = seq.clock_div %clk by 0
  %reg = seq.compreg %in, %div : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @seq_div(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}]
// CHECK-SAME: bmc_reg_clocks = ["clk"]
// CHECK-NOT: seq.compreg
