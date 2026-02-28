// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // Match the UVM failure shape: projected local llhd.ref in a plain func.func.
  func.func @projected_local_ref_dynamic_extract(%idx : i3, %bit : i1) -> i8 {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c0_i8 = hw.constant 0 : i8
    %ptr = llvm.alloca %c1 x i8 : (i64) -> !llvm.ptr
    %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i8>
    llhd.drv %ref, %c0_i8 after %t0 : i8
    %bit_ref = llhd.sig.extract %ref from %idx : <i8> -> <i1>
    llhd.drv %bit_ref, %bit after %t0 : i1
    %probe = llhd.prb %ref : i8
    return %probe : i8
  }
}

// CHECK-LABEL: func.func @projected_local_ref_dynamic_extract
// CHECK: llvm.store
// CHECK-NOT: llhd.
