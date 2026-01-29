// RUN: circt-opt %s --construct-lec="first-module=modA second-module=modB insert-mode=main" | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// The LEC result is reported via printf with format strings
// CHECK-DAG: llvm.func @printf
// CHECK: func.func @modA
// CHECK: verif.lec
// CHECK: llvm.select
// CHECK: llvm.call @printf
// CHECK-LABEL: func.func @main
// CHECK: call @modA
// CHECK-NOT: hw.module @modA
// CHECK-NOT: hw.module @modB
