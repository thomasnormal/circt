// RUN: circt-lec --emit-smtlib --assume-known-inputs -c1=known_a -c2=known_b %s %s | FileCheck %s

module {
  hw.module @known_a(in %in : !hw.struct<value: i1, unknown: i1>, out out : i1) {
    %val = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
    hw.output %val : i1
  }
  hw.module @known_b(in %in : !hw.struct<value: i1, unknown: i1>, out out : i1) {
    %val = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
    hw.output %val : i1
  }
}

// CHECK: (assert
// CHECK: (_ extract 0 0)
// CHECK: #b0
