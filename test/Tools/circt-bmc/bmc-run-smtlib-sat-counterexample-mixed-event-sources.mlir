// RUN: circt-bmc -b 1 --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model.sh --print-counterexample --module top %s 2>&1 | FileCheck %s

hw.module @top(in %in: i1, out out: i1) attributes {initial_values = [], moore.event_source_details = [[{kind = "sequence", label = "sequence"}, {edge = "both", kind = "signal", label = "signal[0]:both", signal_index = 0 : i32, signal_name = "in"}]], moore.event_sources = [["sequence", "signal[0]:both"]], num_regs = 0 : i32} {
  verif.assert %in : i1
  hw.output %in : i1
}

// CHECK-DAG: sat
// CHECK-DAG: BMC_RESULT=SAT
// CHECK-DAG: mixed event sources:
// CHECK-DAG: [0] sequence, signal[0]:both
// CHECK-DAG: estimated event-arm activity:
// CHECK-DAG: [0][1] signal[0]:both -> step 1
// CHECK-DAG: estimated fired arms by step:
// CHECK-DAG: [0] step 1 -> signal[0]:both
// CHECK-DAG: counterexample inputs:
// CHECK-DAG: in = 1'd1
// CHECK-DAG: in_0 = 1'd0
