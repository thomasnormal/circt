// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateXorMask
// CHECK-DAG: [[ONES:%.+]] = hw.constant -1 : i4
// CHECK-DAG: [[LVAL:%.+]] = hw.struct_extract %arg0["value"]
// CHECK-DAG: [[LUNK:%.+]] = hw.struct_extract %arg0["unknown"]
// CHECK-DAG: [[RVAL:%.+]] = hw.struct_extract %arg1["value"]
// CHECK-DAG: [[RUNK:%.+]] = hw.struct_extract %arg1["unknown"]
// CHECK: [[XORVAL:%.+]] = comb.xor [[LVAL]], [[RVAL]] : i4
// CHECK: [[XORUNK:%.+]] = comb.or [[LUNK]], [[RUNK]] : i4
// CHECK: [[KNOWN:%.+]] = comb.xor [[XORUNK]], [[ONES]] : i4
// CHECK: [[MASKED:%.+]] = comb.and [[XORVAL]], [[KNOWN]] : i4
// CHECK: hw.struct_create ([[MASKED]], [[XORUNK]])
func.func @FourStateXorMask(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %x = moore.xor %a, %b : l4
  return %x : !moore.l4
}

// The FourStateAddMask function uses bit-by-bit full-adder expansion with
// propagated unknown masks. This tests the complex 4-state arithmetic path.
// CHECK-LABEL: func.func @FourStateAddMask
// CHECK-DAG: [[ONES2:%.+]] = hw.constant -1 : i4
// CHECK-DAG: [[AVAL:%.+]] = hw.struct_extract %arg0["value"]
// CHECK-DAG: [[AUNK:%.+]] = hw.struct_extract %arg0["unknown"]
// CHECK-DAG: [[BVAL:%.+]] = hw.struct_extract %arg1["value"]
// CHECK-DAG: [[BUNK:%.+]] = hw.struct_extract %arg1["unknown"]
// Bit-by-bit extraction for full-adder with unknown propagation
// CHECK: comb.extract [[AVAL]] from 0 : (i4) -> i1
// CHECK: comb.extract [[AUNK]] from 0 : (i4) -> i1
// CHECK: comb.extract [[BVAL]] from 0 : (i4) -> i1
// CHECK: comb.extract [[BUNK]] from 0 : (i4) -> i1
// Final result assembly: interleaved concats build the value and unknown vectors
// CHECK: comb.concat {{.*}} : i3, i1
// CHECK-NEXT: [[RESULT_UNK:%.+]] = comb.concat {{.*}} : i3, i1
// Final masking: invert unknown mask and apply to value
// CHECK-NEXT: [[KNOWN2:%.+]] = comb.xor [[RESULT_UNK]], [[ONES2]] : i4
// CHECK-NEXT: [[MASKED2:%.+]] = comb.and {{.*}}, [[KNOWN2]] : i4
// CHECK-NEXT: hw.struct_create ([[MASKED2]], [[RESULT_UNK]])
func.func @FourStateAddMask(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %x = moore.add %a, %b : l4
  return %x : !moore.l4
}
