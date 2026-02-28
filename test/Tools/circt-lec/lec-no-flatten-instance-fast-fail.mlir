// RUN: circt-lec %s -c1=c1 -c2=c2 --emit-smtlib | FileCheck %s --check-prefix=FLATTENED
// RUN: not circt-lec %s -c1=c1 -c2=c2 --flatten-hw=false --emit-smtlib 2>&1 | FileCheck %s --check-prefix=NOFLAT

module {
  hw.module private @leaf(in %a: i1, out y: i1) {
    hw.output %a : i1
  }
  hw.module @c1(out result: i1) {
    %c0 = hw.constant false
    %u_leaf.y = hw.instance "u_leaf" @leaf(a: %c0: i1) -> (y: i1)
    hw.output %u_leaf.y : i1
  }
  hw.module @c2(out result: i1) {
    %c0 = hw.constant false
    %u_leaf.y = hw.instance "u_leaf" @leaf(a: %c0: i1) -> (y: i1)
    hw.output %u_leaf.y : i1
  }
}

// FLATTENED: (check-sat)
// NOFLAT: error: solver must not contain any non-SMT operations
// NOFLAT: %[[Y:.*]] = hw.instance "u_leaf" @leaf
