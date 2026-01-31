// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @llhd_delay_clock(in %clk: i1, in %in: i1, out out: i1) {
  %delay = llhd.delay %clk by <0ns, 0d, 0e> : i1
  %clock = seq.to_clock %delay
  %reg = seq.compreg %in, %clock : i1
  hw.output %reg : i1
}

// CHECK: hw.module @llhd_delay_clock
// CHECK: reg_state
// CHECK: reg_next
