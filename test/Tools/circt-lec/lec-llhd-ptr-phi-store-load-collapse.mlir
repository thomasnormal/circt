// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK-NOT: llhd_comb

module {
  hw.module @top(in %cond : i1, in %a : i8, in %b : i8, out o : i8) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %0 = llhd.combinational -> i8 {
      %p1 = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      %p2 = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      cf.cond_br %cond, ^bb1(%a, %p1 : i8, !llvm.ptr), ^bb1(%b, %p2 : i8, !llvm.ptr)
    ^bb1(%val: i8, %ptr: !llvm.ptr):  // 2 preds: ^bb0, ^bb0
      llvm.store %val, %ptr : i8, !llvm.ptr
      %loaded = llvm.load %ptr : !llvm.ptr -> i8
      llhd.yield %loaded : i8
    }
    hw.output %0 : i8
  }
}

