// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts | FileCheck %s

// Test conversion of LTL boolean operators to SMT

// CHECK-LABEL: func.func @test_ltl_and
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.and
// CHECK:   return
func.func @test_ltl_and(%a: i1, %b: i1) {
  %prop = ltl.and %a, %b : i1, i1
  return
}

// CHECK-LABEL: func.func @test_ltl_or
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.or
// CHECK:   return
func.func @test_ltl_or(%a: i1, %b: i1) {
  %prop = ltl.or %a, %b : i1, i1
  return
}

// CHECK-LABEL: func.func @test_ltl_not
// CHECK:   smt.eq
// CHECK:   smt.not
// CHECK:   return
func.func @test_ltl_not(%a: i1) {
  %prop = ltl.not %a : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_implication
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.not
// CHECK:   smt.or
// CHECK:   return
func.func @test_ltl_implication(%a: i1, %b: i1) {
  %prop = ltl.implication %a, %b : i1, i1
  return
}

// CHECK-LABEL: func.func @test_ltl_eventually
// CHECK:   smt.eq
// CHECK:   return
// For BMC, eventually(p) at each step just evaluates to p;
// the BMC loop accumulates with OR over time steps
func.func @test_ltl_eventually(%a: i1) {
  %prop = ltl.eventually %a : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_until
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.or
// CHECK:   return
// p until q = q || p (weak until: either q holds or p holds at this step)
func.func @test_ltl_until(%p: i1, %q: i1) {
  %prop = ltl.until %p, %q : i1, i1
  return
}

// CHECK-LABEL: func.func @test_ltl_boolean_constant_true
// CHECK:   smt.constant true
// CHECK:   return
func.func @test_ltl_boolean_constant_true() {
  %prop = ltl.boolean_constant true
  return
}

// CHECK-LABEL: func.func @test_ltl_boolean_constant_false
// CHECK:   smt.constant false
// CHECK:   return
func.func @test_ltl_boolean_constant_false() {
  %prop = ltl.boolean_constant false
  return
}

// CHECK-LABEL: func.func @test_ltl_nested
// CHECK:   smt.and
// CHECK:   smt.not
// CHECK:   return
func.func @test_ltl_nested(%a: i1, %b: i1, %c: i1) {
  // not(a and b)
  %and = ltl.and %a, %b : i1, i1
  %not_and = ltl.not %and : i1
  return
}

//===----------------------------------------------------------------------===//
// LTL Sequence Operators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_ltl_delay_zero
// CHECK:   smt.eq
// CHECK:   return
// delay(seq, 0) is equivalent to seq itself
func.func @test_ltl_delay_zero(%a: i1) {
  %seq = ltl.delay %a, 0, 0 : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_delay_nonzero
// CHECK:   smt.constant true
// CHECK:   return
// delay(seq, N) with N>0 returns true (obligation is pushed to future step)
func.func @test_ltl_delay_nonzero(%a: i1) {
  %seq = ltl.delay %a, 2, 0 : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_concat_two
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.and
// CHECK:   return
// concat(a, b) for instantaneous sequences is AND
func.func @test_ltl_concat_two(%a: i1, %b: i1) {
  %seq = ltl.concat %a, %b : i1, i1
  return
}

// CHECK-LABEL: func.func @test_ltl_concat_three
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.eq
// CHECK:   smt.and
// CHECK:   return
// concat(a, b, c) for instantaneous sequences is AND of all three
func.func @test_ltl_concat_three(%a: i1, %b: i1, %c: i1) {
  %seq = ltl.concat %a, %b, %c : i1, i1, i1
  return
}

// Note: test_ltl_concat_single is not tested because the folder in ConcatOp
// canonicalizes a single-element concat to its input before the conversion
// pass runs. The conversion handles this case correctly though.

// CHECK-LABEL: func.func @test_ltl_repeat_zero
// CHECK:   smt.constant true
// CHECK:   return
// repeat(seq, 0, 0) is empty sequence (true)
func.func @test_ltl_repeat_zero(%a: i1) {
  %seq = ltl.repeat %a, 0, 0 : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_repeat_one
// CHECK:   smt.eq
// CHECK:   return
// repeat(seq, 1, 0) is seq itself
func.func @test_ltl_repeat_one(%a: i1) {
  %seq = ltl.repeat %a, 1, 0 : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_repeat_multiple
// CHECK:   smt.eq
// CHECK:   return
// repeat(seq, N, 0) with N>1: at single step, still just seq
func.func @test_ltl_repeat_multiple(%a: i1) {
  %seq = ltl.repeat %a, 3, 0 : i1
  return
}

// CHECK-LABEL: func.func @test_ltl_sequence_composition
// CHECK:   smt.constant true
// CHECK:   smt.eq
// CHECK:   smt.and
// CHECK:   return
// Composed: concat(a, delay(b, 1)) - represents "a ##1 b" in SVA
func.func @test_ltl_sequence_composition(%a: i1, %b: i1) {
  %delayed = ltl.delay %b, 1, 0 : i1
  %seq = ltl.concat %a, %delayed : i1, !ltl.sequence
  return
}
