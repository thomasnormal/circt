// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(
      inout %io : !hw.array<4x!hw.struct<foo: i1, bar: i1>>, in %sel : i2,
      out o : i1) {
    %elem = sv.array_index_inout %io[%sel] : !hw.inout<array<4xstruct<foo: i1, bar: i1>>>, i2
    %field = sv.struct_field_inout %elem["bar"] : !hw.inout<struct<foo: i1, bar: i1>>
    %read = sv.read_inout %field : !hw.inout<i1>
    hw.output %read : i1
  }
}
