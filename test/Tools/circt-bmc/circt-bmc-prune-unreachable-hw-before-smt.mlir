// RUN: circt-bmc --emit-mlir -b 1 --module top %s | FileCheck %s --check-prefix=PRUNE
// RUN: not circt-bmc --emit-mlir -b 1 --module top --prune-unreachable-symbols=false %s 2>&1 | FileCheck %s --check-prefix=NOPRUNE

module {
  hw.module @top() {
    %false = hw.constant false
    verif.assert %false : i1
    hw.output
  }

  // `hw.type_scope` is intentionally unsupported by HWToSMT.
  hw.type_scope @deadTypes {
    hw.typedecl @DeadType : i8
  }
}

// PRUNE: func.func @top
// PRUNE-NOT: @deadTypes

// NOPRUNE: error: failed to legalize operation 'hw.type_scope'
