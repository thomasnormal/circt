// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : !hw.struct<a: i1, b: i1>, out o : i1) {
    %field_a = sv.struct_field_inout %io["a"] : !hw.inout<struct<a: i1, b: i1>>
    %read = sv.read_inout %field_a : !hw.inout<i1>
    hw.output %read : i1
  }
}
