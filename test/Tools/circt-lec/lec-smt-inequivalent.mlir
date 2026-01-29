// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  %true = hw.constant true
  %0 = comb.xor %in, %true : i1
  hw.output %0 : i1
}

// CHECK: smt.solver
// CHECK: smt.declare_fun
// CHECK: smt.bv.xor
// CHECK: smt.distinct
// CHECK: smt.check
