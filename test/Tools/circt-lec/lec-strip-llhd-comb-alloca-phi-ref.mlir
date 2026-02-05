// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @top(out out_o : !hw.struct<value: i1, unknown: i1>) {
    %c0_i1 = hw.constant 0 : i1
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %out = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
      %ptr = llvm.alloca %c1_i64 x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      cf.cond_br %c0_i1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
      %init0 = llvm.insertvalue %c0_i1, %undef[0] : !llvm.struct<(i1, i1)>
      %init1 = llvm.insertvalue %c0_i1, %init0[1] : !llvm.struct<(i1, i1)>
      llvm.store %init1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %val = hw.struct_create (%c0_i1, %c0_i1) : !hw.struct<value: i1, unknown: i1>
      llhd.drv %ref, %val after %t0 : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb3(%ptr : !llvm.ptr)
    ^bb2:  // pred: ^bb0
      cf.br ^bb3(%ptr : !llvm.ptr)
    ^bb3(%arg0: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
      %loaded = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %cast = builtin.unrealized_conversion_cast %loaded : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      llhd.yield %cast : !hw.struct<value: i1, unknown: i1>
    }
    hw.output %out : !hw.struct<value: i1, unknown: i1>
  }
}

// CHECK-NOT: llhd.
