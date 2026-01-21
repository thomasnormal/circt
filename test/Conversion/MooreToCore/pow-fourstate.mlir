// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test unsigned power operation with 4-state types
// CHECK-LABEL: hw.module @powu_fourstate
moore.module @powu_fourstate(in %a : !moore.l32, in %b : !moore.l32, out out : !moore.l32) {
  // Extract value components and perform power operation
  // CHECK: [[A_VAL:%.+]] = hw.struct_extract %a["value"]
  // CHECK: [[A_UNK:%.+]] = hw.struct_extract %a["unknown"]
  // CHECK: [[B_VAL:%.+]] = hw.struct_extract %b["value"]
  // CHECK: [[B_UNK:%.+]] = hw.struct_extract %b["unknown"]
  // CHECK: comb.concat
  // CHECK: math.ipowi
  // CHECK: comb.extract
  // Unknown propagation: if any bit is unknown, entire result becomes X
  // CHECK: comb.icmp ne [[A_UNK]],
  // CHECK: comb.icmp ne [[B_UNK]],
  // CHECK: comb.or
  // CHECK: comb.mux
  // CHECK: hw.struct_create
  %0 = moore.powu %a, %b : l32
  moore.output %0 : !moore.l32
}

// Test signed power operation with 4-state types
// CHECK-LABEL: hw.module @pows_fourstate
moore.module @pows_fourstate(in %a : !moore.l32, in %b : !moore.l32, out out : !moore.l32) {
  // Extract value components and perform power operation
  // CHECK: [[A_VAL:%.+]] = hw.struct_extract %a["value"]
  // CHECK: [[A_UNK:%.+]] = hw.struct_extract %a["unknown"]
  // CHECK: [[B_VAL:%.+]] = hw.struct_extract %b["value"]
  // CHECK: [[B_UNK:%.+]] = hw.struct_extract %b["unknown"]
  // CHECK: math.ipowi [[A_VAL]], [[B_VAL]]
  // Unknown propagation
  // CHECK: comb.icmp ne [[A_UNK]],
  // CHECK: comb.icmp ne [[B_UNK]],
  // CHECK: comb.or
  // CHECK: comb.mux
  // CHECK: hw.struct_create
  %0 = moore.pows %a, %b : l32
  moore.output %0 : !moore.l32
}

// Test 2-state power operations still work (regression test)
// CHECK-LABEL: hw.module @powu_twostate
moore.module @powu_twostate(in %a : !moore.i32, in %b : !moore.i32, out out : !moore.i32) {
  // CHECK: comb.concat
  // CHECK: math.ipowi
  // CHECK: comb.extract
  // CHECK-NOT: hw.struct_create
  %0 = moore.powu %a, %b : i32
  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @pows_twostate
moore.module @pows_twostate(in %a : !moore.i32, in %b : !moore.i32, out out : !moore.i32) {
  // CHECK: math.ipowi %a, %b
  // CHECK-NOT: hw.struct_create
  %0 = moore.pows %a, %b : i32
  moore.output %0 : !moore.i32
}
