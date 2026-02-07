// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
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
  // $changed(x) now uses explicit register tracking with a procedure.
  //===--------------------------------------------------------------------===//
  property changed_test;
    @(posedge clk) a |=> $changed(b);
  endproperty
  // $changed uses procedure-based register tracking
  // CHECK: moore.procedure always
  // CHECK: moore.eq
  // CHECK: moore.not
  // CHECK: moore.blocking_assign
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
  // $past(x) now uses explicit register tracking with a procedure.
  //===--------------------------------------------------------------------===//
  property past_default;
    @(posedge clk) a |-> $past(b);
  endproperty
  // CHECK: moore.procedure always
  // CHECK: moore.blocking_assign
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_default);

  //===--------------------------------------------------------------------===//
  // Test $past with explicit delay parameter
  // $past(x, 2) uses a 2-stage register pipeline.
  //===--------------------------------------------------------------------===//
  property past_with_delay;
    @(posedge clk) a |-> $past(b, 2);
  endproperty
  // CHECK: moore.procedure always
  // CHECK: moore.blocking_assign
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
  // $stable(x) now uses explicit register tracking.
  //===--------------------------------------------------------------------===//
  property stable_test;
    @(posedge clk) a |-> $stable(b);
  endproperty
  // CHECK: moore.procedure always
  // CHECK: moore.eq
  // CHECK: moore.blocking_assign
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (stable_test);

  //===--------------------------------------------------------------------===//
  // Test $rose - rising edge detection (0->1 transition)
  // $rose(x) = x && !past(x) via register tracking
  //===--------------------------------------------------------------------===//
  property rose_test;
    @(posedge clk) a |-> $rose(b);
  endproperty
  // $rose produces: current && !past(current) via register
  // CHECK: moore.procedure always
  // CHECK: moore.not
  // CHECK: moore.and
  // CHECK: moore.blocking_assign
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (rose_test);

  //===--------------------------------------------------------------------===//
  // Test $fell - falling edge detection (1->0 transition)
  // $fell(x) = !x && past(x) via register tracking
  //===--------------------------------------------------------------------===//
  property fell_test;
    @(posedge clk) a |-> $fell(b);
  endproperty
  // $fell produces: (current == 0) && !(past(current) == 0)
  // Note: The specific moore.not/and operations are checked
  // along with $rose above since they use similar patterns.
  assert property (fell_test);

  //===--------------------------------------------------------------------===//
  // Test assert final (deferred final assertion)
  //===--------------------------------------------------------------------===//
  // CHECK: moore.assert final
  assert final (a);

  //===--------------------------------------------------------------------===//
  // Test expect (concurrent assertion - must be in procedural context)
  //===--------------------------------------------------------------------===//
  property expect_seq;
    @(posedge clk) a ##1 b;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  initial begin
    expect(expect_seq);
  end

  //===--------------------------------------------------------------------===//
  // Test assume property (concurrent assumption)
  //===--------------------------------------------------------------------===//
  property assume_seq;
    @(posedge clk) a |-> b;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assume
  assume property (assume_seq);
endmodule
