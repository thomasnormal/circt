// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// CHECK: smt.solver
// CHECK: [[IN:%[0-9a-z]+]] = smt.declare_fun
// CHECK: smt.distinct [[IN]], [[IN]]
// CHECK-NOT: smt.bv.xor
// CHECK: smt.check
