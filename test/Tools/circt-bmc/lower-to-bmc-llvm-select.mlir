// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// CHECK: verif.bmc bound 2 num_regs 0
// CHECK: circuit {
// CHECK-NOT: comb.mux
// CHECK: llvm.select {{.*}} : i1, !llvm.struct<(ptr, i64)>

module {
  hw.module @top(in %sel : i1) attributes {num_regs = 0 : i32, initial_values = []} {
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %ptr = llvm.mlir.zero : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %tmp0 = llvm.insertvalue %ptr, %undef[0] : !llvm.struct<(ptr, i64)>
    %val0 = llvm.insertvalue %zero, %tmp0[1] : !llvm.struct<(ptr, i64)>
    %tmp1 = llvm.insertvalue %ptr, %undef[0] : !llvm.struct<(ptr, i64)>
    %val1 = llvm.insertvalue %one, %tmp1[1] : !llvm.struct<(ptr, i64)>
    %mux = comb.mux %sel, %val0, %val1 : !llvm.struct<(ptr, i64)>
    hw.output
  }
}
