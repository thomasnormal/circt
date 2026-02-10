// RUN: circt-lec --emit-mlir -o /dev/null --dump-unknown-sources -c1=unknown_a -c2=unknown_b %s %s 2>&1 | FileCheck %s --check-prefix=BASIC
// RUN: circt-lec --emit-mlir -o /dev/null --dump-unknown-sources -c1=unknown_inv -c2=unknown_inv %s %s 2>&1 | FileCheck %s --check-prefix=INVERT

module {
  hw.module @unknown_a(out out : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %out = hw.struct_create (%true, %true) : !hw.struct<value: i1, unknown: i1>
    hw.output %out : !hw.struct<value: i1, unknown: i1>
  }

  hw.module @unknown_b(out out : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %out = hw.struct_create (%true, %true) : !hw.struct<value: i1, unknown: i1>
    hw.output %out : !hw.struct<value: i1, unknown: i1>
  }

  hw.module @unknown_inv(in %in : !hw.struct<value: i1, unknown: i1>,
                         out out : !hw.struct<value: i1, unknown: i1>) {
    %unk = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
    %val = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
    %true = hw.constant true
    %inv = comb.xor %unk, %true : i1
    %out = hw.struct_create (%val, %inv) : !hw.struct<value: i1, unknown: i1>
    hw.output %out : !hw.struct<value: i1, unknown: i1>
  }
}

// BASIC: === Unknown slice: @unknown_a{{.*}} output#0 ===
// BASIC-DAG: hw.constant
// BASIC-DAG: hw.struct_create
// BASIC: summary: input-unknown-extracts=
// BASIC: unknown-xor-inversions=0
// BASIC: input-unknown-inversions=0
// BASIC: === End unknown slice ===

// INVERT: === Unknown slice: @unknown_inv{{.*}} output#0 ===
// INVERT: comb.xor
// INVERT: unknown-xor-inversion
// INVERT: input-unknown-inversion
// INVERT: summary: input-unknown-extracts=
// INVERT: unknown-xor-inversions=1
// INVERT: input-unknown-inversions=1
// INVERT: === End unknown slice ===
