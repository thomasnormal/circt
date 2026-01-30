// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: hw.module @alwaysCombAssignFilter
// CHECK: llhd.process
// CHECK-NOT: llhd.prb %var
// CHECK: llhd.wait (%in0 : i1), ^bb1
moore.module @alwaysCombAssignFilter(in %in0: !moore.i1) {
  %var = moore.variable : <i1>
  moore.procedure always_comb {
    moore.blocking_assign %var, %in0 : !moore.i1
    moore.return
  }
  moore.output
}
