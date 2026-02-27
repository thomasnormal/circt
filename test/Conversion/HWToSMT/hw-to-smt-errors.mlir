// RUN: circt-opt --convert-hw-to-smt --split-input-file --verify-diagnostics %s

hw.module @zeroBitStructField(in %b: i1, out out: !hw.struct<a: i0, b: i1>) {
  %c0 = arith.constant 0 : i0
  // expected-error @below {{failed to legalize operation 'hw.struct_create' that was explicitly marked illegal}}
  %s = hw.struct_create (%c0, %b) : !hw.struct<a: i0, b: i1>
  hw.output %s : !hw.struct<a: i0, b: i1>
}
