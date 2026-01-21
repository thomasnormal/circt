// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

moore.module @eq_test(in %a : !moore.l4, in %b : !moore.l4, out out0 : !moore.l1) {
  %0 = moore.eq %a, %b : l4 -> l1
  moore.output %0 : !moore.l1
}

// CHECK: hw.module @eq_test
// CHECK: [[A_VAL:%.+]] = hw.struct_extract %a["value"]
// CHECK: [[A_UNK:%.+]] = hw.struct_extract %a["unknown"]
// CHECK: [[B_VAL:%.+]] = hw.struct_extract %b["value"]
// CHECK: [[B_UNK:%.+]] = hw.struct_extract %b["unknown"]
// CHECK: [[CMP:%.+]] = comb.icmp eq [[A_VAL]], [[B_VAL]]
// CHECK: comb.icmp ne [[A_UNK]],
// CHECK: comb.icmp ne [[B_UNK]],
// CHECK: hw.struct_create ({{%.+}}, {{%.+}})
