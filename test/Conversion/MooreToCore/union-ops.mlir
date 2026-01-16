// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// Test union creation and extraction operations

// CHECK-LABEL: hw.module @UnionOps
moore.module @UnionOps(in %a : !moore.i32, in %b : !moore.i16, in %unionVal : !moore.union<{x: i32, y: i16}>, out out1 : !moore.union<{x: i32, y: i16}>, out out2 : !moore.i32, out out3 : !moore.i16) {
  // CHECK: hw.union_create "x", %a : !hw.union<x: i32, y: i16>
  %0 = moore.union_create %a {fieldName = "x"} : !moore.i32 -> !moore.union<{x: i32, y: i16}>

  // CHECK: hw.union_create "y", %b : !hw.union<x: i32, y: i16>
  %1 = moore.union_create %b {fieldName = "y"} : !moore.i16 -> !moore.union<{x: i32, y: i16}>

  // CHECK: hw.union_extract %unionVal["x"] : !hw.union<x: i32, y: i16>
  %2 = moore.union_extract %unionVal, "x" : !moore.union<{x: i32, y: i16}> -> i32

  // CHECK: hw.union_extract %unionVal["y"] : !hw.union<x: i32, y: i16>
  %3 = moore.union_extract %unionVal, "y" : !moore.union<{x: i32, y: i16}> -> i16

  moore.output %0, %2, %3 : !moore.union<{x: i32, y: i16}>, !moore.i32, !moore.i16
}

// CHECK-LABEL: hw.module @UnpackedUnionOps
moore.module @UnpackedUnionOps(in %a : !moore.i32, in %b : !moore.i16, in %unionVal : !moore.uunion<{x: i32, y: i16}>, out out1 : !moore.uunion<{x: i32, y: i16}>, out out2 : !moore.i32) {
  // CHECK: hw.union_create "x", %a : !hw.union<x: i32, y: i16>
  %0 = moore.union_create %a {fieldName = "x"} : !moore.i32 -> !moore.uunion<{x: i32, y: i16}>

  // CHECK: hw.union_extract %unionVal["x"] : !hw.union<x: i32, y: i16>
  %1 = moore.union_extract %unionVal, "x" : !moore.uunion<{x: i32, y: i16}> -> i32

  moore.output %0, %1 : !moore.uunion<{x: i32, y: i16}>, !moore.i32
}
