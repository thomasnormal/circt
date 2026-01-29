// RUN: circt-opt %s --construct-lec="first-module=modA second-module=modB insert-mode=reporting" | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// The LEC result is reported via printf with format strings
// CHECK-DAG: llvm.func @printf
// CHECK-LABEL: func.func @modA
// CHECK: verif.lec
// CHECK: llvm.select
// CHECK: llvm.call @printf
// CHECK-DAG: llvm.mlir.global private constant @"c1 == c2\0A"
// CHECK-DAG: llvm.mlir.global private constant @"c1 != c2\0A"
// CHECK-NOT: hw.module @modA
// CHECK-NOT: hw.module @modB
