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
// CHECK: [[UNK_MASK:%.+]] = comb.or [[A_UNK]], [[B_UNK]]
// CHECK: [[HAS_UNK:%.+]] = comb.icmp ne [[UNK_MASK]], {{%.+}}
// CHECK: [[DIFF:%.+]] = comb.xor [[A_VAL]], [[B_VAL]]
// CHECK: [[KNOWN_BITS:%.+]] = comb.xor [[UNK_MASK]], {{%.+}}
// CHECK: [[KNOWN_DIFF:%.+]] = comb.and [[DIFF]], [[KNOWN_BITS]]
// CHECK: [[MISMATCH:%.+]] = comb.icmp ne [[KNOWN_DIFF]], {{%.+}}
// CHECK: [[NO_MISMATCH:%.+]] = comb.xor [[MISMATCH]], {{%.+}}
// CHECK: [[NO_UNK:%.+]] = comb.xor [[HAS_UNK]], {{%.+}}
// CHECK: [[VAL:%.+]] = comb.and [[NO_MISMATCH]], [[NO_UNK]]
// CHECK: [[UNK:%.+]] = comb.and [[NO_MISMATCH]], [[HAS_UNK]]
// CHECK: hw.struct_create ([[VAL]], [[UNK]])

moore.module @ne_test(in %a : !moore.l4, in %b : !moore.l4, out out0 : !moore.l1) {
  %0 = moore.ne %a, %b : l4 -> l1
  moore.output %0 : !moore.l1
}

// CHECK: hw.module @ne_test
// CHECK: [[A2_VAL:%.+]] = hw.struct_extract %a["value"]
// CHECK: [[A2_UNK:%.+]] = hw.struct_extract %a["unknown"]
// CHECK: [[B2_VAL:%.+]] = hw.struct_extract %b["value"]
// CHECK: [[B2_UNK:%.+]] = hw.struct_extract %b["unknown"]
// CHECK: [[UNK2_MASK:%.+]] = comb.or [[A2_UNK]], [[B2_UNK]]
// CHECK: [[HAS2_UNK:%.+]] = comb.icmp ne [[UNK2_MASK]], {{%.+}}
// CHECK: [[DIFF2:%.+]] = comb.xor [[A2_VAL]], [[B2_VAL]]
// CHECK: [[KNOWN2_BITS:%.+]] = comb.xor [[UNK2_MASK]], {{%.+}}
// CHECK: [[KNOWN2_DIFF:%.+]] = comb.and [[DIFF2]], [[KNOWN2_BITS]]
// CHECK: [[MISMATCH2:%.+]] = comb.icmp ne [[KNOWN2_DIFF]], {{%.+}}
// CHECK: [[NO_MISMATCH2:%.+]] = comb.xor [[MISMATCH2]], {{%.+}}
// CHECK: [[UNK2:%.+]] = comb.and [[NO_MISMATCH2]], [[HAS2_UNK]]
// CHECK: hw.struct_create ([[MISMATCH2]], [[UNK2]])
