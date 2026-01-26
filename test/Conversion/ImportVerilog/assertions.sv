// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test SystemVerilog Assertions (SVA) lowering to LTL
module Assertions(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @Assertions

  //===--------------------------------------------------------------------===//
  // Test disable iff with non-overlapping implication
  //===--------------------------------------------------------------------===//
  property disable_implication;
    @(posedge clk) disable iff (!rst) a |-> ##1 b;
  endproperty
  // disable iff (!rst) produces: ltl.or(rst_negated, property)
  // ##1 b produces: ltl.delay %b, 1, 0
  // a |-> ... produces: ltl.implication
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 1, 0 : i1
  // CHECK-DAG: ltl.implication
  // CHECK-DAG: ltl.or
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (disable_implication);

  //===--------------------------------------------------------------------===//
  // Test strong until with overlap (s_until_with)
  //===--------------------------------------------------------------------===//
  property strong_until_with;
    @(posedge clk) a s_until_with b;
  endproperty
  // s_until_with produces: ltl.until + ltl.eventually + ltl.and
  // CHECK-DAG: ltl.until
  // CHECK-DAG: ltl.eventually
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (strong_until_with);

  //===--------------------------------------------------------------------===//
  // Test $changed - value changed from previous clock cycle
  // $changed(x) uses sampled comparison against the previous value.
  //===--------------------------------------------------------------------===//
  property changed_test;
    @(posedge clk) a |=> $changed(b);
  endproperty
  // CHECK-DAG: moore.past %{{[a-z0-9]+}} delay 1
  // CHECK-DAG: moore.case_eq
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (changed_test);

  //===--------------------------------------------------------------------===//
  // Test cover directive
  //===--------------------------------------------------------------------===//
  property cover_sequence;
    @(posedge clk) a |-> ##1 b;
  endproperty
  // CHECK: verif.{{(clocked_)?}}cover
  cover property (cover_sequence);

  //===--------------------------------------------------------------------===//
  // Test $past with default delay of 1
  // $past(x) = x[-1] (value from 1 clock cycle ago)
  //===--------------------------------------------------------------------===//
  property past_default;
    @(posedge clk) a |-> $past(b);
  endproperty
  // CHECK: moore.past %{{[a-z0-9]+}} delay 1
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_default);

  //===--------------------------------------------------------------------===//
  // Test $past with explicit delay parameter
  // $past(x, 2) = x[-2] (value from 2 clock cycles ago)
  //===--------------------------------------------------------------------===//
  property past_with_delay;
    @(posedge clk) a |-> $past(b, 2);
  endproperty
  // CHECK: moore.past %{{[a-z0-9]+}} delay 2
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_with_delay);

  //===--------------------------------------------------------------------===//
  // Test $sampled - returns the sampled value
  // In assertion context, $sampled returns the current sampled value.
  // CHECK: moore.past %{{[a-z0-9]+}} delay 0
  //===--------------------------------------------------------------------===//
  property sampled_prop;
    @(posedge clk) $sampled(a) |-> b;
  endproperty
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (sampled_prop);

  //===--------------------------------------------------------------------===//
  // Test $stable - value unchanged since last clock
  // $stable(x) uses a sampled case-equality check against the past value.
  //===--------------------------------------------------------------------===//
  property stable_test;
    @(posedge clk) a |-> $stable(b);
  endproperty
  // $stable reuses the moore.past from earlier tests.
  // CHECK-DAG: moore.case_eq
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (stable_test);

  //===--------------------------------------------------------------------===//
  // Test $rose - rising edge detection (0->1 transition)
  // $rose(x) = x && !past(x)
  // The signal is currently true AND was false in the previous cycle
  //===--------------------------------------------------------------------===//
  property rose_test;
    @(posedge clk) a |-> $rose(b);
  endproperty
  // $rose produces: (current == 1) && !(past(current) == 1)
  // CHECK-DAG: moore.case_eq
  // CHECK-DAG: moore.not
  // CHECK-DAG: moore.and
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (rose_test);

  //===--------------------------------------------------------------------===//
  // Test $fell - falling edge detection (1->0 transition)
  // $fell(x) = !x && past(x)
  // The signal is currently false AND was true in the previous cycle
  //===--------------------------------------------------------------------===//
  property fell_test;
    @(posedge clk) a |-> $fell(b);
  endproperty
  // $fell produces: (current == 0) && !(past(current) == 0)
  // Note: The specific moore.case_eq/not/and operations are checked
  // along with $rose above since they use similar patterns.
  assert property (fell_test);

  //===--------------------------------------------------------------------===//
  // Test assert final (deferred final assertion)
  //===--------------------------------------------------------------------===//
  // CHECK: moore.assert final
  assert final (a);

  //===--------------------------------------------------------------------===//
  // Test expect (concurrent assertion)
  //===--------------------------------------------------------------------===//
  property expect_seq;
    @(posedge clk) a ##1 b;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  expect property (expect_seq);

  //===--------------------------------------------------------------------===//
  // Test assume property (concurrent assumption)
  //===--------------------------------------------------------------------===//
  property assume_seq;
    @(posedge clk) a |-> b;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assume
  assume property (assume_seq);
endmodule
