// RUN: circt-opt --strip-unreachable-symbols="entry-symbol=top" %s | FileCheck %s --check-prefix=PRUNE
// RUN: circt-opt --strip-unreachable-symbols %s | FileCheck %s --check-prefix=KEEP

module {
  func.func @unused_a() -> i1 {
    %0 = func.call @unused_b() : () -> i1
    return %0 : i1
  }

  func.func @unused_b() -> i1 {
    %false = arith.constant false
    return %false : i1
  }

  llvm.func @init() {
    llvm.return
  }

  llvm.mlir.global_ctors ctors = [@init], priorities = [0 : i32], data = [#llvm.zero]

  hw.module @top() {
    hw.output
  }
}

// PRUNE: hw.module @top
// PRUNE-NOT: @unused_a
// PRUNE-NOT: @unused_b
// PRUNE-NOT: llvm.func @init
// PRUNE-NOT: llvm.mlir.global_ctors
// KEEP: @unused_a
// KEEP: @unused_b
// KEEP: llvm.func @init
// KEEP: llvm.mlir.global_ctors
