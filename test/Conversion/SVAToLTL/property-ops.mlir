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

  // Non-overlapping implication delays the antecedent by 1 cycle
  // CHECK: [[DELAYED:%[a-z0-9]+]] = ltl.delay %a, 1, 0 : i1
  // CHECK: ltl.implication [[DELAYED]], %b : !ltl.sequence, i1
  %prop_impl_nonoverlap = sva.prop.implication %a, %b : i1, i1

  // Non-overlapping implication with property consequent (no property delay)
  // CHECK: [[DELAYED_PROP:%[a-z0-9]+]] = ltl.delay %a, 1, 0 : i1
  // CHECK: ltl.implication [[DELAYED_PROP]], {{%[a-z0-9]+}} : !ltl.sequence, !ltl.property
  %prop_impl_prop = sva.prop.implication %a, %prop_or : i1, !sva.property

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
  // CHECK: [[UNTIL_STRONG:%[a-z0-9]+]] = ltl.until %a, %b : i1, i1
  // CHECK: [[EVENTUALLY_STRONG:%[a-z0-9]+]] = ltl.eventually %b : i1
  // CHECK: ltl.and [[UNTIL_STRONG]], [[EVENTUALLY_STRONG]] : !ltl.property, !ltl.property
  %prop_until_strong = sva.prop.until %a, %b strong : i1, i1

  //===--------------------------------------------------------------------===//
  // Property If/Else
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT_B_IF:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: [[NOT_C_IF:%[a-z0-9]+]] = ltl.not %c : i1
  // CHECK: [[IF_TRUE:%[a-z0-9]+]] = ltl.implication %a, [[NOT_B_IF]] : i1, !ltl.property
  // CHECK: [[IF_INV_ONE:%[a-z0-9]+]] = hw.constant true
  // CHECK: [[NOT_A_IF:%[a-z0-9]+]] = comb.xor %a, [[IF_INV_ONE]] : i1
  // CHECK: [[IF_FALSE:%[a-z0-9]+]] = ltl.implication [[NOT_A_IF]], [[NOT_C_IF]] : i1, !ltl.property
  // CHECK: ltl.and [[IF_TRUE]], [[IF_FALSE]] : !ltl.property, !ltl.property
  %prop_if_b = sva.prop.not %b : i1
  %prop_if_c = sva.prop.not %c : i1
  %prop_if = sva.prop.if %a, %prop_if_b, %prop_if_c : i1, !sva.property, !sva.property

  //===--------------------------------------------------------------------===//
  // Property Always
  //===--------------------------------------------------------------------===//

  // CHECK: [[ALWAYS_NOT_IN:%[a-z0-9]+]] = ltl.not [[NOT_B_IF]] : !ltl.property
  // CHECK: [[ALWAYS_EV:%[a-z0-9]+]] = ltl.eventually [[ALWAYS_NOT_IN]] : !ltl.property
  // CHECK: ltl.not [[ALWAYS_EV]] : !ltl.property
  %prop_always = sva.prop.always %prop_if_b : !sva.property

  //===--------------------------------------------------------------------===//
  // Property Nexttime
  //===--------------------------------------------------------------------===//

  // CHECK: [[NEXT_TRUE:%[a-z0-9_]+]] = hw.constant true
  // CHECK: [[NEXT_DELAY:%[a-z0-9]+]] = ltl.delay [[NEXT_TRUE]], 2, 0 : i1
  // CHECK: ltl.implication [[NEXT_DELAY]], %a : !ltl.sequence, i1
  %prop_next = sva.prop.nexttime %a, 2 : i1

  //===--------------------------------------------------------------------===//
  // Property Disable Iff
  //===--------------------------------------------------------------------===//

  // CHECK: [[DISABLE_PROP:%[a-z0-9_]+]] = ltl.not %b : i1
  // CHECK: ltl.or %a, [[DISABLE_PROP]] {sva.disable_iff} : i1, !ltl.property
  %prop_disable = sva.prop.not %b : i1
  %prop_disable_iff = sva.disable_iff %a, %prop_disable : i1, !sva.property

  //===--------------------------------------------------------------------===//
  // Property Expect
  //===--------------------------------------------------------------------===//

  // CHECK: [[EXPECT_PROP:%[a-z0-9_]+]] = ltl.not %c : i1
  // CHECK: verif.assert [[EXPECT_PROP]] label "expect_check" : !ltl.property
  %prop_expect = sva.prop.not %c : i1
  sva.expect %prop_expect label "expect_check" : !sva.property

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
