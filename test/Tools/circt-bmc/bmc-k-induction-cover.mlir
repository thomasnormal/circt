// RUN: circt-bmc -b 1 --k-induction --run-smtlib --z3-path=%S/Inputs/fake-z3-unsat.sh --module top %s | FileCheck %s --check-prefix=UNSAT
// RUN: circt-bmc -b 1 --k-induction --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model.sh --module top %s 2>&1 | FileCheck %s --check-prefix=SAT

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.cover %true : i1
  hw.output %true : i1
}

// UNSAT: BMC_BASE=UNSAT
// UNSAT: BMC_STEP=UNSAT
// UNSAT: BMC_RESULT=UNSAT

// SAT: BMC_BASE=SAT
// SAT-NOT: BMC_STEP=
// SAT: BMC_RESULT=SAT
