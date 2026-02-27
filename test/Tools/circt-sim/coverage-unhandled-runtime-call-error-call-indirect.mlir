// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unhandled coverage runtime calls reached via func.call_indirect
// must fail loudly instead of silently returning X/default values.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  llvm.mlir.global internal @"cov::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @__moore_cross_get_bins_hit]
    ]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llhd.process {
      %vt = llvm.mlir.addressof @"cov::__vtable__" : !llvm.ptr
      %slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %f = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, i32) -> i64
      %0 = func.call_indirect %f(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
