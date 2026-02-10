// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  // CHECK-LABEL: hw.module @top
  // CHECK-SAME: in %c0 : i1
  // CHECK-SAME: in %c1 : i1
  // CHECK-SAME: in %sig_field0_unknown : i1
  // CHECK: circt.bmc_abstracted_llhd_interface_input_details =
  // CHECK-DAG: sig_field0_unknown
  // CHECK-DAG: interface_enable_resolution_unknown
  // CHECK-DAG: signal = "sig"
  // CHECK-DAG: field = 0
  // CHECK: circt.bmc_abstracted_llhd_interface_inputs = 1 : i32
  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %true = hw.constant true
    %false = hw.constant false
    scf.if %c0 {
      llvm.store %true, %field : i1, !llvm.ptr
    }
    scf.if %c1 {
      llvm.store %false, %field : i1, !llvm.ptr
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @clean
  // CHECK-NOT: circt.bmc_abstracted_llhd_interface_inputs
  // CHECK-NOT: circt.bmc_abstracted_llhd_interface_input_details
  hw.module @clean(in %in : i1) {
    verif.assert %in : i1
    hw.output
  }
}
