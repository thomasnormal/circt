// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK: comb.or
// CHECK: verif.assert
// CHECK-NOT: llvm.store
// CHECK-NOT: llhd.prb

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(!llvm.struct<(i1, i1)>)>

  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.struct<(i1, i1)>)>
    %true = hw.constant true
    %false = hw.constant false
    %val_true = hw.struct_create (%true, %false) : !hw.struct<value: i1, unknown: i1>
    %val_false = hw.struct_create (%false, %false) : !hw.struct<value: i1, unknown: i1>
    %llvm_true = builtin.unrealized_conversion_cast %val_true : !hw.struct<value: i1, unknown: i1> to !llvm.struct<(i1, i1)>
    %llvm_false = builtin.unrealized_conversion_cast %val_false : !hw.struct<value: i1, unknown: i1> to !llvm.struct<(i1, i1)>
    scf.if %c0 {
      llvm.store %llvm_true, %field : !llvm.struct<(i1, i1)>, !llvm.ptr
    }
    scf.if %c1 {
      llvm.store %llvm_false, %field : !llvm.struct<(i1, i1)>, !llvm.ptr
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
    %val = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %val["value"] : !hw.struct<value: i1, unknown: i1>
    verif.assert %value : i1
    hw.output
  }
}
