// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// CHECK: smt.solver
// CHECK: [[IN:%[0-9a-z_]+]] = smt.declare_fun "in"
// CHECK: [[C1OUT:%[0-9a-z_]+]] = smt.declare_fun "c1_out"
// CHECK: [[C2OUT:%[0-9a-z_]+]] = smt.declare_fun "c2_out"
// CHECK: smt.eq [[C1OUT]], [[IN]]
// CHECK: smt.eq [[C2OUT]], [[IN]]
// CHECK: smt.distinct [[C1OUT]], [[C2OUT]]
// CHECK-NOT: smt.bv.xor
// CHECK: smt.check
