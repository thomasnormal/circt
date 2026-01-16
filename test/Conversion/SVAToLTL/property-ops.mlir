// RUN: circt-opt --lower-sva-to-ltl %s | FileCheck %s

// Test conversions for SVA property operations to LTL property operations.

hw.module @test_property_ops(in %clk: i1, in %a: i1, in %b: i1, in %c: i1) {

  //===--------------------------------------------------------------------===//
  // Property Not
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.not %a : i1
  %prop_not = sva.prop.not %a : i1

  //===--------------------------------------------------------------------===//
  // Property And (with property inputs)
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT_A:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: [[NOT_B:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: ltl.and [[NOT_A]], [[NOT_B]] : !ltl.property, !ltl.property
  %prop_a = sva.prop.not %a : i1
  %prop_b = sva.prop.not %b : i1
  %prop_and = sva.prop.and %prop_a, %prop_b : !sva.property, !sva.property

  //===--------------------------------------------------------------------===//
  // Property Or (with property inputs)
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT_A2:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: [[NOT_B2:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: ltl.or [[NOT_A2]], [[NOT_B2]] : !ltl.property, !ltl.property
  %prop_a2 = sva.prop.not %a : i1
  %prop_b2 = sva.prop.not %b : i1
  %prop_or = sva.prop.or %prop_a2, %prop_b2 : !sva.property, !sva.property

  //===--------------------------------------------------------------------===//
  // Property Implication (Overlapping)
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.implication %a, %b : i1, i1
  %prop_impl_overlap = sva.prop.implication %a, %b overlapping : i1, i1

  //===--------------------------------------------------------------------===//
  // Property Implication (Non-Overlapping)
  //===--------------------------------------------------------------------===//

  // Non-overlapping implication adds a delay of 1 to the consequent
  // CHECK: [[DELAYED:%[a-z0-9]+]] = ltl.delay %b, 1, 0 : i1
  // CHECK: ltl.implication %a, [[DELAYED]] : i1, !ltl.sequence
  %prop_impl_nonoverlap = sva.prop.implication %a, %b : i1, i1

  //===--------------------------------------------------------------------===//
  // Property Eventually
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.eventually %a : i1
  %prop_eventually = sva.prop.eventually %a : i1

  //===--------------------------------------------------------------------===//
  // Property Until
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.until %a, %b : i1, i1
  %prop_until = sva.prop.until %a, %b : i1, i1

  //===--------------------------------------------------------------------===//
  // Property Clock (Posedge)
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT1:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: ltl.clock [[NOT1]], posedge %clk : !ltl.property
  %prop0 = sva.prop.not %a : i1
  %prop_clock_pos = sva.prop.clock %prop0, posedge %clk : !sva.property

  //===--------------------------------------------------------------------===//
  // Property Clock (Negedge)
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT2:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: ltl.clock [[NOT2]], negedge %clk : !ltl.property
  %prop1 = sva.prop.not %b : i1
  %prop_clock_neg = sva.prop.clock %prop1, negedge %clk : !sva.property

  //===--------------------------------------------------------------------===//
  // Property Clock (Both edges)
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT3:%[a-z0-9]+]] = ltl.not %c : i1
  // CHECK: ltl.clock [[NOT3]], edge %clk : !ltl.property
  %prop2 = sva.prop.not %c : i1
  %prop_clock_both = sva.prop.clock %prop2, edge %clk : !sva.property

  hw.output
}
