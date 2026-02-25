// RUN: circt-opt --hw-convert-bitcasts --convert-hw-to-smt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @struct_array_bitcast
  // CHECK-NOT: hw.struct_create
  // CHECK: smt.array.select
  // CHECK: smt.bv.concat
  // CHECK: return %{{.*}} : !smt.bv<32>
  hw.module @struct_array_bitcast(
      in %in : !hw.struct<value: i16, unknown: i16>,
      out out : !hw.struct<a: !hw.array<2x!hw.struct<value: i4, unknown: i4>>,
                           b: !hw.struct<value: i8, unknown: i8>>) {
    %cast = hw.bitcast %in : (!hw.struct<value: i16, unknown: i16>) ->
        !hw.struct<a: !hw.array<2x!hw.struct<value: i4, unknown: i4>>,
                   b: !hw.struct<value: i8, unknown: i8>>
    hw.output %cast : !hw.struct<a: !hw.array<2x!hw.struct<value: i4, unknown: i4>>,
                                 b: !hw.struct<value: i8, unknown: i8>>
  }
}
