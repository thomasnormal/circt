// RUN: circt-opt %s --convert-verif-to-smt=approx-temporal=true --reconcile-unrealized-casts | FileCheck %s

// Test conversion of LTL boolean operators to SMT

// CHECK-LABEL: func.func @test_ltl_and
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.ite
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_and(%a: i1, %b: i1) {
  %prop = ltl.and %a, %b : i1, i1
  verif.assert %prop : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_or
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.or
// CHECK:       smt.ite
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_or(%a: i1, %b: i1) {
  %prop = ltl.or %a, %b : i1, i1
  verif.assert %prop : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_intersect
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.ite
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_intersect(%a: i1, %b: i1) {
  %prop = ltl.intersect %a, %b : i1, i1
  verif.assert %prop : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_not
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_not(%a: i1) {
  %prop = ltl.not %a : i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_implication
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_implication(%a: i1, %b: i1) {
  %prop = ltl.implication %a, %b : i1, i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_eventually
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// For BMC, eventually(p) at each step just evaluates to p;
// the BMC loop accumulates with OR over time steps
func.func @test_ltl_eventually(%a: i1) {
  %prop = ltl.eventually %a : i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_eventually_weak
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Weak eventually is always satisfied in the SMT lowering.
func.func @test_ltl_eventually_weak(%a: i1) {
  %prop = ltl.eventually %a {ltl.weak} : i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_until
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// p until q = q || p (weak until: either q holds or p holds at this step)
func.func @test_ltl_until(%p: i1, %q: i1) {
  %prop = ltl.until %p, %q : i1, i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_boolean_constant_true
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_boolean_constant_true() {
  %prop = ltl.boolean_constant true
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_boolean_constant_false
// CHECK:       smt.constant false
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_boolean_constant_false() {
  %prop = ltl.boolean_constant false
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_ltl_nested
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.ite
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
func.func @test_ltl_nested(%a: i1, %b: i1, %c: i1) {
  // not(a and b)
  %and = ltl.and %a, %b : i1, i1
  %not_and = ltl.not %and : i1
  verif.assert %not_and : !ltl.property
  return
}

//===----------------------------------------------------------------------===//
// LTL Sequence Operators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_ltl_delay_zero
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// delay(seq, 0) is equivalent to seq itself
func.func @test_ltl_delay_zero(%a: i1) {
  %seq = ltl.delay %a, 0, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_delay_nonzero
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// delay(seq, N) with N>0 returns true (obligation is pushed to future step)
func.func @test_ltl_delay_nonzero(%a: i1) {
  %seq = ltl.delay %a, 2, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_concat_two
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// concat(a, b) for instantaneous sequences is AND
func.func @test_ltl_concat_two(%a: i1, %b: i1) {
  %seq = ltl.concat %a, %b : i1, i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_concat_three
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// concat(a, b, c) for instantaneous sequences is AND of all three
func.func @test_ltl_concat_three(%a: i1, %b: i1, %c: i1) {
  %seq = ltl.concat %a, %b, %c : i1, i1, i1
  verif.assert %seq : !ltl.sequence
  return
}

// Note: test_ltl_concat_single is not tested because the folder in ConcatOp
// canonicalizes a single-element concat to its input before the conversion
// pass runs. The conversion handles this case correctly though.

// CHECK-LABEL: func.func @test_ltl_repeat_zero
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// repeat(seq, 0, 0) is empty sequence (true)
func.func @test_ltl_repeat_zero(%a: i1) {
  %seq = ltl.repeat %a, 0, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_repeat_one
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// repeat(seq, 1, 0) is seq itself
func.func @test_ltl_repeat_one(%a: i1) {
  %seq = ltl.repeat %a, 1, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_repeat_multiple
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// repeat(seq, N, 0) with N>1: at single step, still just seq
func.func @test_ltl_repeat_multiple(%a: i1) {
  %seq = ltl.repeat %a, 3, 0 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_repeat_unbounded
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// repeat(seq, N) with N>1: at single step, still just seq
func.func @test_ltl_repeat_unbounded(%a: i1) {
  %seq = ltl.repeat %a, 2 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_goto_repeat_zero
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// goto_repeat(seq, 0, N) is empty sequence (true)
func.func @test_ltl_goto_repeat_zero(%a: i1) {
  %seq = ltl.goto_repeat %a, 0, 2 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_goto_repeat_one
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// goto_repeat(seq, 1, N) is seq itself at a single step
func.func @test_ltl_goto_repeat_one(%a: i1) {
  %seq = ltl.goto_repeat %a, 1, 2 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_non_consecutive_repeat_zero
// CHECK:       smt.constant true
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// non_consecutive_repeat(seq, 0, N) is empty sequence (true)
func.func @test_ltl_non_consecutive_repeat_zero(%a: i1) {
  %seq = ltl.non_consecutive_repeat %a, 0, 2 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_non_consecutive_repeat_one
// CHECK:       smt.bv.constant
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// non_consecutive_repeat(seq, 1, N) is seq itself at a single step
func.func @test_ltl_non_consecutive_repeat_one(%a: i1) {
  %seq = ltl.non_consecutive_repeat %a, 1, 2 : i1
  verif.assert %seq : !ltl.sequence
  return
}

// CHECK-LABEL: func.func @test_ltl_sequence_composition
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   smt.constant true
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Composed: concat(a, delay(b, 1)) - represents "a ##1 b" in SVA
func.func @test_ltl_sequence_composition(%a: i1, %b: i1) {
  %delayed = ltl.delay %b, 1, 0 : i1
  %seq = ltl.concat %a, %delayed : i1, !ltl.sequence
  verif.assert %seq : !ltl.sequence
  return
}

//===----------------------------------------------------------------------===//
// SVA Implication Operators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_overlapping_implication
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Overlapping implication (|->): antecedent |-> consequent
// Semantics: !antecedent || consequent (if antecedent holds, consequent must hold at same time)
func.func @test_overlapping_implication(%antecedent: i1, %consequent: i1) {
  %prop = ltl.implication %antecedent, %consequent : i1, i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_non_overlapping_implication
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   smt.constant true
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Non-overlapping implication (|=>): antecedent |=> consequent
// Represented as: implication(antecedent, delay(consequent, 1))
// Semantics: !antecedent || true (at current step, since consequent is delayed)
func.func @test_non_overlapping_implication(%antecedent: i1, %consequent: i1) {
  %delayed_consequent = ltl.delay %consequent, 1, 0 : i1
  %prop = ltl.implication %antecedent, %delayed_consequent : i1, !ltl.sequence
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_implication_with_sequence
// CHECK:       smt.bv.constant
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK-DAG:   builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.eq
// CHECK:       smt.and
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Implication with sequence antecedent: (a ##0 b) |-> c
// The sequence a && b (concat of instantaneous sequences) is the antecedent
func.func @test_implication_with_sequence(%a: i1, %b: i1, %c: i1) {
  %seq = ltl.concat %a, %b : i1, i1
  %prop = ltl.implication %seq, %c : !ltl.sequence, i1
  verif.assert %prop : !ltl.property
  return
}

// CHECK-LABEL: func.func @test_implication_with_delayed_antecedent
// CHECK-DAG:   smt.bv.constant
// CHECK-DAG:   smt.constant true
// CHECK:       builtin.unrealized_conversion_cast
// CHECK:       smt.eq
// CHECK:       smt.not
// CHECK:       smt.or
// CHECK:       smt.not
// CHECK:       smt.assert
// CHECK:       return
// Implication with delayed sequence: (##2 a) |-> b
// The delayed sequence evaluates to true at current step
func.func @test_implication_with_delayed_antecedent(%a: i1, %b: i1) {
  %delayed_seq = ltl.delay %a, 2, 0 : i1
  %prop = ltl.implication %delayed_seq, %b : !ltl.sequence, i1
  verif.assert %prop : !ltl.property
  return
}
