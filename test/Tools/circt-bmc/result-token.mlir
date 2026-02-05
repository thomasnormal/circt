// REQUIRES: bmc-jit
// REQUIRES: z3
// RUN: circt-bmc -b 1 --module top %s | FileCheck %s

hw.module @top(in %a: i1, out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.assert %true : i1
  hw.output %a : i1
}

// CHECK: BMC_RESULT=UNSAT
