// RUN: circt-opt --llhd-mem2reg %s | FileCheck %s

// Ensure llhd-mem2reg can materialize default values for LLVM pointer types.
// CHECK-LABEL: @PtrDefault
hw.module @PtrDefault(in %cond: i1) {
  // CHECK: [[ZERO:%.+]] = llvm.mlir.zero : !llvm.ptr
  %init = llvm.mlir.zero : !llvm.ptr
  %sig = llhd.sig %init : !llvm.ptr
  %t = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.process
  llhd.process {
    // CHECK: [[FALSE:%.+]] = hw.constant false
    // CHECK: [[PRB:%.+]] = llhd.prb %sig
    // CHECK: cf.cond_br %cond, ^bb1, ^bb2
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: ^bb1:
    // CHECK: cf.br ^bb3([[ZERO]], %true : !llvm.ptr, i1)
    llhd.drv %sig, %init after %t : !llvm.ptr
    cf.br ^bb3
  ^bb2:
    // CHECK: ^bb2:
    // CHECK: cf.br ^bb3([[PRB]], [[FALSE]] : !llvm.ptr, i1)
    cf.br ^bb3
  ^bb3:
    %v = llhd.prb %sig : !llvm.ptr
    func.call @use_ptr(%v) : (!llvm.ptr) -> ()
    llhd.halt
  }
}

func.func private @use_ptr(!llvm.ptr)
