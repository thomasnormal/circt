// RUN: circt-bmc -b 8 --module impl_delay1 %s | FileCheck %s --check-prefix=UNSAT1
// RUN: circt-bmc -b 8 --module impl_delay4 %s | FileCheck %s --check-prefix=UNSAT4

// UNSAT1: BMC_RESULT=UNSAT
// UNSAT4: BMC_RESULT=UNSAT

module {
  hw.module @impl_delay1(in %clk : i1) {
    %true = hw.constant true
    %d1 = ltl.delay %true, 1, 0 : i1
    %imp = ltl.implication %true, %d1 : i1, !ltl.sequence
    verif.clocked_assert %imp, posedge %clk : !ltl.property
    hw.output
  }

  hw.module @impl_delay4(in %clk : i1) {
    %true = hw.constant true
    %d4 = ltl.delay %true, 4, 0 : i1
    %imp = ltl.implication %true, %d4 : i1, !ltl.sequence
    verif.clocked_assert %imp, posedge %clk : !ltl.property
    hw.output
  }
}
