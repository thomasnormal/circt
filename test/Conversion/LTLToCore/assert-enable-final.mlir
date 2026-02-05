// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// Test that assertions with enable signals preserve the enable in the
// lowered final checks.

module {
  // CHECK-LABEL: hw.module @assert_enable
  hw.module @assert_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %prop = ltl.eventually %a : i1
    verif.assert %prop if %en : !ltl.property
    // CHECK: verif.assert {{.*}} if %en {bmc.final} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @assume_enable
  hw.module @assume_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %prop = ltl.eventually %a : i1
    verif.assume %prop if %en : !ltl.property
    // CHECK: verif.assume {{.*}} if %en {bmc.final} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @cover_enable
  hw.module @cover_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %prop = ltl.eventually %a : i1
    verif.cover %prop if %en : !ltl.property
    // CHECK: verif.cover {{.*}} if %en {bmc.final} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @clocked_assert_enable
  hw.module @clocked_assert_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %clk = seq.from_clock %clock
    %prop = ltl.eventually %a : i1
    verif.clocked_assert %prop if %en, posedge %clk : !ltl.property
    // CHECK: verif.assert {{.*}} if %en {{{.*}}bmc.final} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @clocked_assume_enable
  hw.module @clocked_assume_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %clk = seq.from_clock %clock
    %prop = ltl.eventually %a : i1
    verif.clocked_assume %prop if %en, posedge %clk : !ltl.property
    // CHECK: verif.assume {{.*}} if %en {{{.*}}bmc.final} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @clocked_cover_enable
  hw.module @clocked_cover_enable(in %clock : !seq.clock, in %en : i1, in %a : i1) {
    %clk = seq.from_clock %clock
    %prop = ltl.eventually %a : i1
    verif.clocked_cover %prop if %en, posedge %clk : !ltl.property
    // CHECK: verif.cover {{.*}} if %en {{{.*}}bmc.final} : i1
    hw.output
  }
}
