// RUN: circt-lec --emit-llvm --print-solver-output -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// With --print-solver-output, the LLVM IR contains format strings for solver and model output
// CHECK: Solver
// CHECK: Model
// The result is reported via printf, not circt_lec_report_result
// CHECK: printf
