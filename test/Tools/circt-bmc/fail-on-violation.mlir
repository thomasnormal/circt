// REQUIRES: bmc-jit
// REQUIRES: z3
// RUN: not circt-bmc -b 1 --fail-on-violation --module top %s | FileCheck %s

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %false = hw.constant false
  verif.assert %false : i1
  %true = hw.constant true
  hw.output %true : i1
}

// CHECK: BMC_RESULT=SAT
