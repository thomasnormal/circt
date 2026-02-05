// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

// CHECK-LABEL: hw.module @top(
// CHECK-SAME: in %c0 : i1, in %c1 : i1, in %{{.*}}_unknown : i1
// CHECK: comb.or
// CHECK: verif.assert
// CHECK-NOT: llvm.store
// CHECK-NOT: llhd.prb

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(!llvm.struct<(i1)>)>

  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.struct<(i1)>)>
    %true = hw.constant true
    %false = hw.constant false
    %llvm_true = builtin.unrealized_conversion_cast %true : i1 to !llvm.struct<(i1)>
    %llvm_false = builtin.unrealized_conversion_cast %false : i1 to !llvm.struct<(i1)>
    scf.if %c0 {
      llvm.store %llvm_true, %field : !llvm.struct<(i1)>, !llvm.ptr
    }
    scf.if %c1 {
      llvm.store %llvm_false, %field : !llvm.struct<(i1)>, !llvm.ptr
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
