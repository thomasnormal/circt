// RUN: circt-bmc -b 5 --module m_const_prop --run-smtlib %s | FileCheck %s
// CHECK: BMC_RESULT=UNSAT

// Regression test: when the PROPERTY inside a disable_iff is constant true
// (e.g. ltl.boolean_constant true), the assertion should be trivially UNSAT.
// Previously, lowerDisableIff created comb.or(disable, true) which was not
// recognized as a constant by getI1Constant, causing the ClockOp handler to
// wrap it in a shift register initialized to false — producing a false SAT
// on cycle 1.

module {
  hw.module @m_const_prop(in %clk : i1, in %rst_n : i1) {
    %true_prop = ltl.boolean_constant true
    %not_rst = comb.xor %rst_n, %true_const : i1
    %true_const = hw.constant true
    %prop = ltl.or %not_rst, %true_prop {sva.disable_iff} : i1, !ltl.property
    verif.clocked_assert %prop, posedge %clk : !ltl.property
    hw.output
  }
}
