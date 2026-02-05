// RUN: not circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: LLHD combinational control flow requires abstraction; rerun without --strict-llhd

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  hw.module @top(in %cond : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    %true = hw.constant true
    %false = hw.constant false
    scf.if %cond {
      llvm.store %true, %field : i1, !llvm.ptr
    } else {
      llvm.store %false, %field : i1, !llvm.ptr
    }
    hw.output
  }
}
