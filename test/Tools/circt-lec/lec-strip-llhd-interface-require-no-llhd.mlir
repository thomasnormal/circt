// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --strip-llhd-interface-signals='require-no-llhd=false' %s | FileCheck %s --check-prefix=RESIDUAL

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  // DEFAULT-LABEL: hw.module @top(
  // DEFAULT-SAME: in %sig_field0 : i1
  // DEFAULT: circt.bmc_abstracted_llhd_interface_input_details
  // DEFAULT: reason = "interface_field_requires_abstraction"
  // DEFAULT-NOT: llhd.sig
  // DEFAULT-NOT: llhd.prb
  // DEFAULT: verif.assert %sig_field0 : i1
  //
  // RESIDUAL-LABEL: hw.module @top(
  // RESIDUAL-NOT: circt.bmc_abstracted_llhd_interface_inputs
  // RESIDUAL: llhd.sig
  // RESIDUAL: llhd.prb
  // RESIDUAL: verif.assert
  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    scf.if %c0 {
      %t = hw.constant true
      llvm.store %t, %field : i1, !llvm.ptr
    }
    scf.if %c1 {
      %f = hw.constant false
      llvm.store %f, %field : i1, !llvm.ptr
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
