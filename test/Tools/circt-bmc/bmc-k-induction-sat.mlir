// RUN: circt-bmc -b 1 --k-induction --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model.sh --module top %s 2>&1 | FileCheck %s

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %false = hw.constant false
  verif.assert %false : i1
  hw.output %false : i1
}

// CHECK: BMC_BASE=SAT
// CHECK-NOT: BMC_STEP=
// CHECK: BMC_RESULT=SAT
