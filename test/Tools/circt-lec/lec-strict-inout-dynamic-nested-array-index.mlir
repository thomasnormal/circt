// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : !hw.array<4x!hw.array<2xi1>>, in %sel0 : i2,
                 in %sel1 : i1, out o : i1) {
    %elem0 = sv.array_index_inout %io[%sel0] : !hw.inout<array<4xarray<2xi1>>>, i2
    %elem1 = sv.array_index_inout %elem0[%sel1] : !hw.inout<array<2xi1>>, i1
    %read = sv.read_inout %elem1 : !hw.inout<i1>
    hw.output %read : i1
  }
}
