// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// Verify strict LEC can resolve multiple interface stores in a single block.

// CHECK: smt.solver
// CHECK: smt.check

module {
  llvm.mlir.global internal @iface_storage(#llvm.zero) : !llvm.struct<(i1)>

  hw.module @top() {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>

    %false = hw.constant false
    llvm.store %false, %field : i1, !llvm.ptr
    %true = hw.constant true
    llvm.store %true, %field : i1, !llvm.ptr

    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
