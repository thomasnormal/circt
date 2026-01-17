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
