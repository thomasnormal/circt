// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s --check-prefix=PRUNE
// RUN: not circt-lec --emit-mlir --prune-unreachable-symbols=false -c1=modA -c2=modB %s 2>&1 | FileCheck %s --check-prefix=NOPRUNE

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// `hw.type_scope` is intentionally unsupported by HWToSMT.
hw.type_scope @deadTypes {
  hw.typedecl @DeadType : i8
}

// PRUNE: smt.solver
// PRUNE-NOT: @deadTypes

// NOPRUNE: error: failed to legalize operation 'hw.type_scope'
