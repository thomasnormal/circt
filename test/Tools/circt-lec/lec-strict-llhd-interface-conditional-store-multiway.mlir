// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

// CHECK: comb.mux
// CHECK: comb.mux
// CHECK: verif.assert
// CHECK-NOT: llvm.store
// CHECK-NOT: llhd.prb

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %true = hw.constant true
    %false = hw.constant false
    scf.if %c0 {
      llvm.store %true, %field : i1, !llvm.ptr
    } else {
      scf.if %c1 {
        llvm.store %false, %field : i1, !llvm.ptr
      } else {
        llvm.store %true, %field : i1, !llvm.ptr
      }
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
