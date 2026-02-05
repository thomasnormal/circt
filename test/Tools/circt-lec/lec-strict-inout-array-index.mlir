// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : !hw.array<4xi1>, in %sel : i2, out o : i1) {
    %idx2 = hw.constant 2 : i2
    %elem2 = sv.array_index_inout %io[%idx2] : !hw.inout<array<4xi1>>, i2
    %read_const = sv.read_inout %elem2 : !hw.inout<i1>
    %elem_dyn = sv.array_index_inout %io[%sel] : !hw.inout<array<4xi1>>, i2
    %read_dyn = sv.read_inout %elem_dyn : !hw.inout<i1>
    hw.output %read_dyn : i1
  }
}
