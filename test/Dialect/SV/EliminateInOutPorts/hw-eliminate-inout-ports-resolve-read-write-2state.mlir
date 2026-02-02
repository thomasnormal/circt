// RUN: circt-opt --hw-eliminate-inout-ports=resolve-read-write %s | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %io_rd : i1
// CHECK-SAME: in %io_unknown : i1
// CHECK-SAME: out io_wr : i1
// CHECK-NOT: sv.read_inout
// CHECK-NOT: sv.assign
// CHECK: comb.or

module {
  hw.module @top(inout %io : i1, out o : i1) {
    %c0 = hw.constant 0 : i1
    sv.assign %io, %c0 : i1
    %read = sv.read_inout %io : !hw.inout<i1>
    hw.output %read : i1
  }
}
