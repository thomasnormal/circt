// RUN: circt-bmc --emit-mlir -b 1 --module top %s | FileCheck %s --check-prefix=PRUNE
// RUN: circt-bmc --emit-mlir -b 1 --module top --prune-unreachable-symbols=false %s | FileCheck %s --check-prefix=KEEP

module {
  func.func @unused_a() -> i1 {
    %0 = func.call @unused_b() : () -> i1
    return %0 : i1
  }

  func.func @unused_b() -> i1 {
    %0 = func.call @unused_a() : () -> i1
    %false = hw.constant false
    %1 = comb.xor %0, %false : i1
    return %1 : i1
  }

  hw.module @top() {
    %false = hw.constant false
    verif.assert %false : i1
    hw.output
  }
}

// PRUNE: smt.solver
// PRUNE-NOT: @unused_a
// PRUNE-NOT: @unused_b
// KEEP: @unused_a
// KEEP: @unused_b
