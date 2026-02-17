// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: hw.module @intDomainConversion
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i1, unknown: i1>
// CHECK-NOT: hw.bitcast
moore.module @intDomainConversion(out o: !moore.l1) {
  %c = moore.constant 1 : !moore.i1
  %l = moore.conversion %c : !moore.i1 -> !moore.l1
  moore.output %l : !moore.l1
}
