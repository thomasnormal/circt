// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=2" | FileCheck %s

hw.module @top(in %a : i1) attributes {initial_values = [], moore.event_source_details = [[{kind = "sequence", label = "sequence"}, {edge = "posedge", iff_name = "a", kind = "signal", label = "signal[0]:posedge:iff", signal_index = 0 : i32, signal_name = "a"}], [{kind = "sequence", label = "sequence"}, {edge = "both", kind = "signal", label = "signal[0]:both", signal_index = 0 : i32, signal_name = "a"}], [{kind = "sequence", label = "sequence[0]", sequence_index = 0 : i32}, {kind = "sequence", label = "sequence[1]:iff", sequence_index = 1 : i32}]], moore.event_sources = [["sequence", "signal[0]:posedge:iff"], ["sequence", "signal[0]:both"], ["sequence[0]", "sequence[1]:iff"]], num_regs = 0 : i32} {
  verif.assert %a : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK-DAG: bmc_event_sources
// CHECK-DAG: "sequence"
// CHECK-DAG: "signal[0]:posedge:iff"
// CHECK-DAG: "signal[0]:both"
// CHECK-DAG: "sequence[0]"
// CHECK-DAG: "sequence[1]:iff"
// CHECK-DAG: bmc_event_source_details
// CHECK-DAG: signal_name = "a"
// CHECK-DAG: edge = "posedge"
