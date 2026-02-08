// RUN: circt-bmc -b 1 --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model.sh --print-counterexample --module top %s 2>&1 | FileCheck %s

hw.module @top(in %in: i1, out out: i1) attributes {initial_values = [], moore.mixed_event_sources = [["sequence", "signal[0]:posedge:iff"]], num_regs = 0 : i32} {
  verif.assert %in : i1
  hw.output %in : i1
}

// CHECK-DAG: sat
// CHECK-DAG: BMC_RESULT=SAT
// CHECK-DAG: mixed event sources:
// CHECK-DAG: [0] sequence, signal[0]:posedge:iff
// CHECK-DAG: counterexample inputs:
// CHECK-DAG: in = 1'd1
// CHECK-DAG: in_0 = 1'd0
