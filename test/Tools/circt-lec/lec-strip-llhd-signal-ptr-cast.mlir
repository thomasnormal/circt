// RUN: circt-opt -strip-llhd-interface-signals %s | FileCheck %s

hw.module @signal_ptr_cast(out out : !hw.struct<value: i1, unknown: i1>) {
  %init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
  %sig = llhd.sig %init : !hw.struct<value: i1, unknown: i1>
  %ptr = builtin.unrealized_conversion_cast %sig : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
  %val = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
  %val_llvm = builtin.unrealized_conversion_cast %val : !hw.struct<value: i1, unknown: i1> to !llvm.struct<(i1, i1)>
  llvm.store %val_llvm, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
  %load_hw = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %load_hw : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @signal_ptr_cast
// CHECK-NOT: llvm.
// CHECK-NOT: llhd.
