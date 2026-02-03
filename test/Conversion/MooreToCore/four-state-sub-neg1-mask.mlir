// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateSubNeg1Mask
// CHECK: [[ONES:%.+]] = hw.constant -1 : i4
// CHECK: [[AVAL:%.+]] = hw.struct_extract %arg0["value"]
// CHECK: [[AUNK:%.+]] = hw.struct_extract %arg0["unknown"]
// CHECK: [[NOTVAL:%.+]] = comb.xor [[AVAL]], [[ONES]] : i4
// CHECK: [[KNOWN:%.+]] = comb.xor [[AUNK]], [[ONES]] : i4
// CHECK: [[MASKED:%.+]] = comb.and [[NOTVAL]], [[KNOWN]] : i4
// CHECK: hw.struct_create ([[MASKED]], [[AUNK]])
// CHECK-NOT: comb.icmp ne [[AUNK]]
func.func @FourStateSubNeg1Mask(%a: !moore.l4) -> !moore.l4 {
  %c = moore.constant b1111 : l4
  %x = moore.sub %c, %a : l4
  return %x : !moore.l4
}
