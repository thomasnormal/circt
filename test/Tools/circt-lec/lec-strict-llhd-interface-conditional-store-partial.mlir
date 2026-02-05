// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

// Verify that partial conditional stores are resolved via abstraction
// with an unknown input.

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %cond : i1
// CHECK-SAME: in %sig_field0_unknown : i1
// CHECK-NOT: llhd.sig
// CHECK-NOT: llhd.prb
// CHECK-NOT: llhd.drv

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  hw.module @top(in %cond : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %true = hw.constant true
    scf.if %cond {
      llvm.store %true, %field : i1, !llvm.ptr
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
