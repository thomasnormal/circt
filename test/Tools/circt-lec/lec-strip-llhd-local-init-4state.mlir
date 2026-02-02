// RUN: circt-opt -strip-llhd-interface-signals %s | FileCheck %s

hw.module @local_init_4state(out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %zero = hw.constant 0 : i1
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %comb = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
    %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
    %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
    %cast = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
    %tmp0 = llvm.insertvalue %zero, %undef[0] : !llvm.struct<(i1, i1)>
    %tmp1 = llvm.insertvalue %zero, %tmp0[1] : !llvm.struct<(i1, i1)>
    llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
    llhd.yield %cast : !hw.struct<value: i1, unknown: i1>
  }
  hw.output %comb : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @local_init_4state
// CHECK-NOT: llhd.combinational
// CHECK: %[[INIT:.*]] = hw.struct_create ({{.*false.*}}, {{.*false.*}}) : !hw.struct<value: i1, unknown: i1>
// CHECK: hw.output %[[INIT]] : !hw.struct<value: i1, unknown: i1>
