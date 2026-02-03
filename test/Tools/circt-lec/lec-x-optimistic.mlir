// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=modA -c2=modB %s | FileCheck %s --check-prefix=STRICT
// RUN: circt-lec --run-smtlib --x-optimistic -c1=modA -c2=modB %s | FileCheck %s --check-prefix=OPT

hw.module @modA(in %in: !hw.struct<value: i2, unknown: i2>, out out: !hw.struct<value: i2, unknown: i2>) {
  hw.output %in : !hw.struct<value: i2, unknown: i2>
}

hw.module @modB(in %in: !hw.struct<value: i2, unknown: i2>, out out: !hw.struct<value: i2, unknown: i2>) {
  %val = hw.struct_extract %in["value"] : !hw.struct<value: i2, unknown: i2>
  %unk = hw.struct_extract %in["unknown"] : !hw.struct<value: i2, unknown: i2>
  %zero = hw.constant 0 : i2
  %any = comb.icmp ne %unk, %zero : i2
  %all = hw.constant -1 : i2
  %unk2 = comb.mux %any, %all, %unk : i2
  %out = hw.struct_create (%val, %unk2) : !hw.struct<value: i2, unknown: i2>
  hw.output %out : !hw.struct<value: i2, unknown: i2>
}

// STRICT: LEC_RESULT=NEQ
// OPT: LEC_RESULT=EQ
