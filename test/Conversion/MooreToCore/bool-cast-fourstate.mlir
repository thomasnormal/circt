// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

moore.module @bool_cast_test(in %a : !moore.l3, out out0 : !moore.l1) {
  %0 = moore.bool_cast %a : l3 -> l1
  moore.output %0 : !moore.l1
}

// CHECK: hw.module @bool_cast_test
// CHECK: [[A_VAL:%.+]] = hw.struct_extract %a["value"]
// CHECK: [[A_UNK:%.+]] = hw.struct_extract %a["unknown"]
// CHECK: comb.icmp ne [[A_VAL]],
// CHECK: comb.icmp ne [[A_UNK]],
// CHECK: hw.struct_create ({{%.+}}, {{%.+}})
