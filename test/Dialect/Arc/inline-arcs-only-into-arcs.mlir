// RUN: circt-opt %s --arc-inline=into-arcs-only=1 | FileCheck %s

// CHECK-LABEL: hw.module @onlyIntoArcs
hw.module @onlyIntoArcs(in %arg0: i4, in %arg1: i4, in %arg2: !seq.clock, out out0: i4) {
  %0 = arc.call @sub1(%arg0, %arg1) : (i4, i4) -> i4
  hw.output %0 : i4
}
// CHECK-LABEL: arc.define @sub1
arc.define @sub1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add
  %0 = arc.call @sub2(%arg0, %arg1) : (i4, i4) -> i4
  arc.output %0 : i4
}
// CHECK-NOT: arc.define @sub2
arc.define @sub2(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
