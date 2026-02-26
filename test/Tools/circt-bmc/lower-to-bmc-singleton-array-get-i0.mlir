// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=1" \
// RUN:   --convert-hw-to-smt --convert-comb-to-smt --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts | FileCheck %s

// Ensure singleton array_get with i0 index is canonicalized before SMT
// lowering. Otherwise hw.constant 0 : i0 can survive and fail legalization.
module {
  hw.module @top(in %arr : !hw.array<1xi1>) attributes {
    num_regs = 0 : i32,
    initial_values = []
  } {
    %c0_i0 = hw.constant 0 : i0
    %elt = hw.array_get %arr[%c0_i0] : !hw.array<1xi1>, i0
    verif.assert %elt : i1
    hw.output
  }
}

// CHECK: smt.solver
// CHECK-NOT: hw.constant
