// RUN: circt-bmc -b 2 --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model-witness-activity.sh --print-counterexample --module top %s 2>&1 | FileCheck %s

hw.module @top(in %in: i1, in %seq: i1, in %en: i1, out out: i1) attributes {initial_values = [], moore.event_source_details = [[{kind = "sequence", label = "sequence", sequence_name = "seq"}, {edge = "both", iff_name = "en", kind = "signal", label = "signal[0]:both:iff", signal_index = 0 : i32, signal_name = "in"}]], moore.event_sources = [["sequence", "signal[0]:both:iff"]], num_regs = 0 : i32} {
  verif.assert %in : i1
  hw.output %in : i1
}

// CHECK-DAG: sat
// CHECK-DAG: BMC_RESULT=SAT
// CHECK-DAG: mixed event sources:
// CHECK-DAG: [0] sequence, signal[0]:both:iff
// CHECK-DAG: event-arm activity:
// CHECK-DAG: [0][0] sequence -> step 0
// CHECK-DAG: [0][1] signal[0]:both:iff -> step 1
// CHECK-DAG: fired arms by step:
// CHECK-DAG: [0] step 0 -> sequence
// CHECK-DAG: [0] step 1 -> signal[0]:both:iff
