// RUN: circt-opt %s --coverage-instrument=fsm=true,expression=true,assertion=true | FileCheck %s

// Test FSM state coverage instrumentation

// CHECK-LABEL: hw.module @FSMModule
hw.module @FSMModule(in %clk: !seq.clock, in %rst: i1, in %next_state: i3, out state_out: i3) {
  // A register with "state" in the name should get FSM coverage when fsm=true
  %state_reg = seq.compreg %next_state, %clk {name = "state_reg"} : i3

  // CHECK: coverage.fsm.state %state_reg name "state_reg"
  // CHECK: hw.output %state_reg

  hw.output %state_reg : i3
}

// Test expression coverage instrumentation

// CHECK-LABEL: hw.module @ExpressionModule
hw.module @ExpressionModule(in %a: i1, in %b: i1, in %c: i1, out result: i1) {
  // Boolean AND of multiple 1-bit values should get expression coverage
  // CHECK: coverage.expression %a, %b name "and_expr_
  %0 = comb.and %a, %b : i1

  // Boolean OR of multiple 1-bit values should get expression coverage
  // CHECK: coverage.expression %0, %c name "or_expr_
  %1 = comb.or %0, %c : i1

  hw.output %1 : i1
}

// Test assertion coverage instrumentation

// CHECK-LABEL: hw.module @AssertionModule
hw.module @AssertionModule(in %clk: !seq.clock, in %valid: i1, in %data: i8) {
  sv.always posedge %clk {
    // Assertion should get coverage
    // CHECK: coverage.assertion %valid name
    sv.assert %valid, immediate label "data_valid_assertion"
  }
}

// Test with all coverage types disabled except branch

// RUN: circt-opt %s --coverage-instrument=line=false,toggle=false,branch=true | FileCheck %s --check-prefix=BRANCH-ONLY

// BRANCH-ONLY-LABEL: hw.module @MuxModule
hw.module @MuxModule(in %sel: i1, in %a: i8, in %b: i8, out result: i8) {
  // BRANCH-ONLY: coverage.branch %sel
  %0 = comb.mux %sel, %a, %b : i8
  // BRANCH-ONLY-NOT: coverage.line
  // BRANCH-ONLY-NOT: coverage.toggle
  hw.output %0 : i8
}
