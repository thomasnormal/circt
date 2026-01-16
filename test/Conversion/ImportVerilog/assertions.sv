// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module Assertions(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @Assertions

  property disable_implication;
    @(posedge clk) disable iff (!rst) a |-> ##1 b;
  endproperty
  // CHECK: ltl.or
  // CHECK: ltl.delay
  // CHECK: ltl.implication
  assert property (disable_implication);

  property strong_until_with;
    @(posedge clk) a s_until_with b;
  endproperty
  // CHECK: ltl.until
  // CHECK: ltl.eventually
  assert property (strong_until_with);

  // Note: $changed and other value change functions produce ltl.property types
  // due to ltl.not usage. When used as implication consequent (with |=>), they work.
  property changed_test;
    @(posedge clk) a |=> $changed(b);
  endproperty
  // CHECK: ltl.past
  // CHECK: ltl.not
  // CHECK: ltl.or
  assert property (changed_test);

  property cover_sequence;
    @(posedge clk) a |-> ##1 b;
  endproperty
  // CHECK: verif.cover
  cover property (cover_sequence);

  // Test $past with default delay of 1
  property past_default;
    @(posedge clk) a |-> $past(b);
  endproperty
  // CHECK: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.implication
  assert property (past_default);

  // Test $past with explicit delay parameter
  property past_with_delay;
    @(posedge clk) a |-> $past(b, 2);
  endproperty
  // CHECK: ltl.past {{%[a-z0-9]+}}, 2 : i1
  // CHECK: ltl.implication
  assert property (past_with_delay);

  // Test $sampled - returns the sampled value (effectively the input in assertion context)
  property sampled_prop;
    @(posedge clk) $sampled(a) |-> b;
  endproperty
  // CHECK: moore.to_builtin_bool
  // CHECK: ltl.implication
  assert property (sampled_prop);

  // Test $stable - value unchanged since last clock
  property stable_test;
    @(posedge clk) a |-> $stable(b);
  endproperty
  // CHECK: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.not
  // CHECK: ltl.and
  // CHECK: ltl.or
  // CHECK: ltl.implication
  assert property (stable_test);

  // Test $rose - value is true and was false
  property rose_test;
    @(posedge clk) a |-> $rose(b);
  endproperty
  // CHECK: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.not
  // CHECK: ltl.and
  // CHECK: ltl.implication
  assert property (rose_test);

  // Test $fell - value is false and was true
  property fell_test;
    @(posedge clk) a |-> $fell(b);
  endproperty
  // CHECK: ltl.past {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.not
  // CHECK: ltl.and
  // CHECK: ltl.implication
  assert property (fell_test);
endmodule
