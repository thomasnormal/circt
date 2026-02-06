// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @async_reset(in %clk: !seq.clock, in %rst: i1, in %in: i1, out out: i1) {
  %c0 = hw.constant 0 : i1
  %reg = seq.firreg %in clock %clk reset async %rst, %c0 : i1
  hw.output %reg : i1
}

// CHECK: hw.module @async_reset
// CHECK: reg_state
// CHECK: reg_next
// CHECK: comb.mux
