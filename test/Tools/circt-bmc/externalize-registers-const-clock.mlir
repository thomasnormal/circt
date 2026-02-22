// RUN: circt-opt --externalize-registers="allow-multi-clock=true" %s | FileCheck %s

hw.module @const_low(in %in : i1, out out : i1) {
  %clk = seq.const_clock low
  %r = seq.compreg %in, %clk : i1
  hw.output %r : i1
}

// CHECK-LABEL: hw.module @const_low(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{clock_key = "const0", invert = false}]
// CHECK-SAME: bmc_reg_clocks = [""]
// CHECK-NOT: seq.compreg

hw.module @const_inv(in %in : i1, out out : i1) {
  %clk = seq.const_clock low
  %inv = seq.clock_inv %clk
  %r = seq.compreg %in, %inv : i1
  hw.output %r : i1
}

// CHECK-LABEL: hw.module @const_inv(
// CHECK-SAME: attributes {{.*}}bmc_reg_clock_sources = [{clock_key = "const1", invert = false}]
// CHECK-SAME: bmc_reg_clocks = [""]
// CHECK-NOT: seq.compreg
