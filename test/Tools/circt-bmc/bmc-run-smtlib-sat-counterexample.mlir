// RUN: circt-bmc -b 1 --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model.sh --print-counterexample --module top %s 2>&1 | FileCheck %s

hw.module @top(in %in: i1, out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  verif.assert %in : i1
  hw.output %in : i1
}

// CHECK-DAG: sat
// CHECK-DAG: BMC_RESULT=SAT
// CHECK-DAG: counterexample inputs:
// CHECK-DAG: in = 1'd1
// CHECK-DAG: in_0 = 1'd0
