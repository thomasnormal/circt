// RUN: circt-bmc -b 1 --k-induction --run-smtlib --z3-path=%S/Inputs/fake-z3-unsat.sh --module top %s | FileCheck %s

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.assert %true : i1
  hw.output %true : i1
}

// CHECK: BMC_BASE=UNSAT
// CHECK: BMC_STEP=UNSAT
// CHECK: BMC_RESULT=UNSAT
