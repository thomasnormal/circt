// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=modA -c2=modB %s | FileCheck %s --check-prefix=STRICT
// RUN: circt-lec --run-smtlib --x-optimistic -c1=modA -c2=modB %s | FileCheck %s --check-prefix=OPT

// Model a & ~a in 4-state form. This should be known-zero even if %in is X,
// but correlation-losing X-prop will treat it as unknown.
hw.module @modA(in %in: !hw.struct<value: i1, unknown: i1>,
                out out: !hw.struct<value: i1, unknown: i1>) {
  %val = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unk = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %one = hw.constant true
  %not_val = comb.xor %val, %one : i1
  %known = comb.xor %unk, %one : i1
  %val_masked = comb.and %val, %known : i1
  %not_masked = comb.and %not_val, %known : i1
  %result_val = comb.and %val_masked, %not_masked : i1
  %out = hw.struct_create (%result_val, %unk) : !hw.struct<value: i1, unknown: i1>
  hw.output %out : !hw.struct<value: i1, unknown: i1>
}

// Reference: known zero.
hw.module @modB(in %in: !hw.struct<value: i1, unknown: i1>,
                out out: !hw.struct<value: i1, unknown: i1>) {
  %zero = hw.constant false
  %out = hw.struct_create (%zero, %zero) : !hw.struct<value: i1, unknown: i1>
  hw.output %out : !hw.struct<value: i1, unknown: i1>
}

// STRICT: LEC_RESULT=NEQ
// OPT: LEC_RESULT=EQ
