// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(lower-ltl-to-core))' | FileCheck %s

module {
  // CHECK-LABEL: hw.module @top
  // CHECK-NOT: verif.clocked_assert
  // CHECK: verif.assert {{.*}}bmc.clock_edge
  hw.module @top() {
    %false = hw.constant false
    %prop = ltl.boolean_constant true
    verif.clocked_assert %prop, posedge %false : !ltl.property
    hw.output
  }
}
