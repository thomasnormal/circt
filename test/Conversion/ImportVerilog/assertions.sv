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
  // CHECK: verif.assert
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
  // CHECK: verif.assert
  assert property (strong_until_with);

  //===--------------------------------------------------------------------===//
  // Test $changed - value changed from previous clock cycle
  // $changed(x) = (x && !past(x)) || (!x && past(x))
  //===--------------------------------------------------------------------===//
  property changed_test;
    @(posedge clk) a |=> $changed(b);
  endproperty
  // CHECK-DAG: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: verif.assert
  assert property (changed_test);

  //===--------------------------------------------------------------------===//
  // Test cover directive
  //===--------------------------------------------------------------------===//
  property cover_sequence;
    @(posedge clk) a |-> ##1 b;
  endproperty
  // CHECK: verif.cover
  cover property (cover_sequence);

  //===--------------------------------------------------------------------===//
  // Test $past with default delay of 1
  // $past(x) = x[-1] (value from 1 clock cycle ago)
  //===--------------------------------------------------------------------===//
  property past_default;
    @(posedge clk) a |-> $past(b);
  endproperty
  // CHECK: [[PAST1:%[a-z0-9]+]] = ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[PAST1]]
  // CHECK: verif.assert
  assert property (past_default);

  //===--------------------------------------------------------------------===//
  // Test $past with explicit delay parameter
  // $past(x, 2) = x[-2] (value from 2 clock cycles ago)
  //===--------------------------------------------------------------------===//
  property past_with_delay;
    @(posedge clk) a |-> $past(b, 2);
  endproperty
  // CHECK: [[PAST2:%[a-z0-9]+]] = ltl.past {{%[a-z0-9]+}}, 2 : i1
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[PAST2]]
  // CHECK: verif.assert
  assert property (past_with_delay);

  //===--------------------------------------------------------------------===//
  // Test $sampled - returns the sampled value
  // In assertion context, this is effectively a pass-through
  //===--------------------------------------------------------------------===//
  property sampled_prop;
    @(posedge clk) $sampled(a) |-> b;
  endproperty
  // CHECK: ltl.implication
  // CHECK: verif.assert
  assert property (sampled_prop);

  //===--------------------------------------------------------------------===//
  // Test $stable - value unchanged since last clock
  // $stable(x) = (x && past(x)) || (!x && !past(x))
  //===--------------------------------------------------------------------===//
  property stable_test;
    @(posedge clk) a |-> $stable(b);
  endproperty
  // CHECK-DAG: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK-DAG: ltl.not
  // CHECK-DAG: ltl.and
  // CHECK-DAG: ltl.or
  // CHECK: verif.assert
  assert property (stable_test);

  //===--------------------------------------------------------------------===//
  // Test $rose - rising edge detection (0->1 transition)
  // $rose(x) = x && !past(x)
  // The signal is currently true AND was false in the previous cycle
  //===--------------------------------------------------------------------===//
  property rose_test;
    @(posedge clk) a |-> $rose(b);
  endproperty
  // $rose produces: current && !past(current)
  // CHECK: [[ROSE_PAST:%[a-z0-9]+]] = ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: [[ROSE_NOT_PAST:%[a-z0-9]+]] = ltl.not [[ROSE_PAST]]
  // CHECK: [[ROSE_AND:%[a-z0-9]+]] = ltl.and {{%[a-z0-9]+}}, [[ROSE_NOT_PAST]]
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[ROSE_AND]]
  // CHECK: verif.assert
  assert property (rose_test);

  //===--------------------------------------------------------------------===//
  // Test $fell - falling edge detection (1->0 transition)
  // $fell(x) = !x && past(x)
  // The signal is currently false AND was true in the previous cycle
  //===--------------------------------------------------------------------===//
  property fell_test;
    @(posedge clk) a |-> $fell(b);
  endproperty
  // $fell produces: !current && past(current)
  // CHECK: [[FELL_PAST:%[a-z0-9]+]] = ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: [[FELL_NOT_CURR:%[a-z0-9]+]] = ltl.not {{%[a-z0-9]+}} : i1
  // CHECK: [[FELL_AND:%[a-z0-9]+]] = ltl.and [[FELL_NOT_CURR]], [[FELL_PAST]]
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[FELL_AND]]
  // CHECK: verif.assert
  assert property (fell_test);
endmodule
