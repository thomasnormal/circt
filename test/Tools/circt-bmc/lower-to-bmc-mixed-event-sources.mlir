// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=2" | FileCheck %s

hw.module @top(in %a : i1) attributes {initial_values = [], moore.event_sources = [["sequence", "signal[0]:posedge:iff"], ["sequence", "signal[0]:both"], ["sequence[0]", "sequence[1]:iff"]], num_regs = 0 : i32} {
  verif.assert %a : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK-SAME: bmc_event_sources
// CHECK-SAME: "sequence"
// CHECK-SAME: "signal[0]:posedge:iff"
// CHECK-SAME: "signal[0]:both"
// CHECK-SAME: "sequence[0]"
// CHECK-SAME: "sequence[1]:iff"
