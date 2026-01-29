// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// Test that multiple sequential stores to the same field are resolved by
// dominance analysis, using the last dominating store value.
// This now passes because resolveStoresByDominance handles this case.

// CHECK: smt.solver
module {
  llvm.mlir.global internal @iface_storage(#llvm.zero) : !llvm.struct<(i1)>

  hw.module @top() {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %true = hw.constant true
    %false = hw.constant false
    llvm.store %true, %field : i1, !llvm.ptr
    llvm.store %false, %field : i1, !llvm.ptr
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
