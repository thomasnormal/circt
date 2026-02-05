// RUN: circt-opt %s --convert-verif-to-smt --split-input-file --verify-diagnostics

// LTL sequence types are now supported - no error expected
func.func @assert_with_sequence_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.sequence
  verif.assert %0 : !ltl.sequence
  return
}

// -----

// LTL property types are now supported - no error expected
func.func @assert_with_property_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.property
  verif.assert %0 : !ltl.property
  return
}

// -----

// Multiple assertions are now supported - no error expected
func.func @multiple_assertions_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i32):
    %c1_i32 = hw.constant 1 : i32
    %cond1 = comb.icmp ugt %arg0, %c1_i32 : i32
    verif.assert %cond1 : i1
    %cond2 = comb.icmp ugt %arg1, %c1_i32 : i32
    verif.assert %cond2 : i1
    %sum = comb.add %arg0, %arg1 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

// -----

// Multiple asserting modules are now supported - no error expected
func.func @multiple_asserting_modules_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @OneAssertion(x: %arg1: i1) -> ()
    hw.instance "" @OneAssertion(x: %arg2: i1) -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @OneAssertion(in %x: i1) {
  verif.assert %x : i1
}

// -----

func.func @no_assertions() -> (i1) {
  // expected-warning @below {{no property provided to check in module - will trivially find no violations.}}
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32):
    hw.instance "" @empty() -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @empty() {
}

// -----

// Check that we don't see an error when there's one nested assertion

func.func @one_nested_assertion() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1):
    hw.instance "" @OneAssertion(x: %arg1: i1) -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @OneAssertion(in %x: i1) {
  verif.assert %x : i1
}


// -----

// Two separated assertions are now supported - no error expected
func.func @two_separated_assertions() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @OneAssertion(x: %arg1: i1) -> ()
    verif.assert %arg2 : i1
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @OneAssertion(in %x: i1) {
  verif.assert %x : i1
}

// -----

// Multiple nested assertions are now supported - no error expected
func.func @multiple_nested_assertions() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @TwoAssertions(x: %arg1: i1, y: %arg2: i1) -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @TwoAssertions(in %x: i1, in %y: i1) {
  verif.assert %x : i1
  verif.assert %y : i1
}

// -----

func.func @multiple_clocks() -> (i1) {
  // expected-error @below {{multi-clock BMC requires bmc_reg_clocks or bmc_reg_clock_sources with one entry per register}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk, %clk : !seq.clock, !seq.clock
  }
  loop {
    ^bb0(%clock0: !seq.clock, %clock1: !seq.clock):
    verif.yield %clock0, %clock1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clock0: !seq.clock, %clock1: !seq.clock, %arg0: i32):
    %c1_i32 = hw.constant 1 : i32
    %cond1 = comb.icmp ugt %arg0, %c1_i32 : i32
    verif.assert %cond1 : i1
    verif.yield %arg0 : i32
  }
  func.return %bmc : i1
}

// -----

func.func @multiple_clocks() -> (i1) {
  // expected-error @below {{unsupported integer initial value in BMC conversion}}
  // expected-error @below {{failed to legalize operation 'verif.bmc' that was explicitly marked illegal}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [0]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    verif.yield %clk: !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: !hw.array<2xi32>):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : !hw.array<2xi32>
  }
  func.return %bmc : i1
}

// -----

func.func @wrong_initial_type() -> (i1) {
  // expected-error @below {{bit width of initial value does not match register}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [-1 : i7]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    verif.yield %clk: !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i8):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i8
  }
  func.return %bmc : i1
}

// -----

func.func @refines_non_primitive_free_var() -> () {
  // expected-error @below {{failed to legalize operation 'verif.refines' that was explicitly marked illegal}}
  verif.refines first {
  ^bb0(%arg0: !smt.bv<4>):
    // expected-error @below {{Uninterpreted function of non-primitive type cannot be converted.}}
    %nondetar = smt.declare_fun : !smt.array<[!smt.bv<4> -> !smt.bv<32>]>
    %sel = smt.array.select %nondetar[%arg0] : !smt.array<[!smt.bv<4> -> !smt.bv<32>]>
    %cc = builtin.unrealized_conversion_cast %sel : !smt.bv<32> to i32
    verif.yield %cc : i32
  } second {
  ^bb0(%arg0: !smt.bv<4>):
    %const = smt.bv.constant #smt.bv<0> : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %const : !smt.bv<32> to i32
    verif.yield %cc : i32
  }
  return
}

// -----

func.func @ltl_delay_nonzero_outside_bmc(%a: i1) {
  // expected-error @below {{ltl.delay with delay > 0 must be lowered by the BMC multi-step infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.delay' that was explicitly marked illegal}}
  %seq = ltl.delay %a, 1, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// -----

func.func @ltl_past_nonzero_outside_bmc(%a: i1) {
  // expected-error @below {{ltl.past with delay > 0 must be lowered by the BMC multi-step infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.past' that was explicitly marked illegal}}
  %v = ltl.past %a, 1 : i1
  verif.assert %v : !ltl.sequence
  return
}

// -----

func.func @ltl_eventually_outside_bmc(%a: i1) {
  // expected-error @below {{ltl.eventually must be lowered by the BMC/LTLToCore infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.eventually' that was explicitly marked illegal}}
  %v = ltl.eventually %a : i1
  verif.assert %v : !ltl.property
  return
}

// -----

func.func @ltl_until_outside_bmc(%p: i1, %q: i1) {
  // expected-error @below {{ltl.until must be lowered by the BMC/LTLToCore infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.until' that was explicitly marked illegal}}
  %v = ltl.until %p, %q : i1, i1
  verif.assert %v : !ltl.property
  return
}

// -----

func.func @ltl_concat_multi_outside_bmc(%a: i1, %b: i1) {
  // expected-error @below {{ltl.concat with multiple inputs must be lowered by the BMC/LTLToCore infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.concat' that was explicitly marked illegal}}
  %v = ltl.concat %a, %b : i1, i1
  verif.assert %v : !ltl.sequence
  return
}

// -----

func.func @ltl_repeat_outside_bmc(%a: i1) {
  // expected-error @below {{ltl.repeat must be lowered by the BMC/LTLToCore infrastructure}}
  // expected-error @below {{failed to legalize operation 'ltl.repeat' that was explicitly marked illegal}}
  %v = ltl.repeat %a, 2, 0 : i1
  verif.assert %v : !ltl.sequence
  return
}
