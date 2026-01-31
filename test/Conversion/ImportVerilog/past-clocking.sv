// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// Test $past with explicit clocking argument in various positions.
// This tests the fix for $past(expr, cycles, @(posedge clk)) where
// the clocking event is in position 3 (the gating_expr position in SVA).

module PastClocking(input logic clk, input logic clk2,
                    input logic [7:0] data, input logic a, input logic b,
                    input logic enable, input logic reset);
  // CHECK-LABEL: moore.module @PastClocking

  //===--------------------------------------------------------------------===//
  // Basic $past with explicit clocking in position 3
  //===--------------------------------------------------------------------===//

  // $past(expr, cycles, @(posedge clk)) - clocking at position 3
  property past_with_clocking_pos3;
    @(posedge clk) a |-> $past(b, 2, @(posedge clk));
  endproperty
  // CHECK: moore.variable
  // CHECK: moore.procedure
  // CHECK: moore.blocking_assign
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_with_clocking_pos3);

  //===--------------------------------------------------------------------===//
  // $past with enable and explicit clocking in position 4
  //===--------------------------------------------------------------------===//

  // $past(expr, cycles, enable, @(posedge clk)) - clocking at position 4
  property past_with_clocking_enable_pos4;
    @(posedge clk) a |-> $past(b, 1, enable, @(posedge clk));
  endproperty
  // CHECK: moore.conditional
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_with_clocking_enable_pos4);

  //===--------------------------------------------------------------------===//
  // $past with different delay values and position 3 clocking
  //===--------------------------------------------------------------------===//

  // $past with delay 1 and explicit clocking
  property past_delay1_clocking;
    @(posedge clk) a |-> $past(data, 1, @(posedge clk)) == 8'h00;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_delay1_clocking);

  // $past with delay 3 and explicit clocking
  property past_delay3_clocking;
    @(posedge clk) a |-> $past(data, 3, @(posedge clk)) == 8'hFF;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_delay3_clocking);

  //===--------------------------------------------------------------------===//
  // $past with negedge clocking
  //===--------------------------------------------------------------------===//

  // $past with negedge clocking at position 3
  property past_negedge_clocking;
    @(negedge clk) a |-> $past(b, 2, @(negedge clk));
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_negedge_clocking);

  //===--------------------------------------------------------------------===//
  // $past with different clock for property vs $past
  //===--------------------------------------------------------------------===//

  // Property uses clk, $past uses clk2
  property past_different_clock;
    @(posedge clk) a |-> $past(b, 1, @(posedge clk2));
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_different_clock);

  //===--------------------------------------------------------------------===//
  // $past with multi-bit data and comparison
  //===--------------------------------------------------------------------===//

  // $past with 8-bit data, explicit clocking, and comparison
  property past_data_compare;
    @(posedge clk) a |-> ($past(data, 2, @(posedge clk)) != data);
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_data_compare);

  //===--------------------------------------------------------------------===//
  // $past in cover statements
  //===--------------------------------------------------------------------===//

  // Cover statement with $past and explicit clocking
  property past_cover_clocking;
    @(posedge clk) $past(a, 1, @(posedge clk)) && !a;
  endproperty
  // CHECK: verif.{{(clocked_)?}}cover
  cover property (past_cover_clocking);

  //===--------------------------------------------------------------------===//
  // $past in assume statements
  //===--------------------------------------------------------------------===//

  // Assume statement with $past and explicit clocking
  property past_assume_clocking;
    @(posedge clk) reset |-> $past(data, 1, @(posedge clk)) == 8'h00;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assume
  assume property (past_assume_clocking);

  //===--------------------------------------------------------------------===//
  // $past with enable and different clock
  //===--------------------------------------------------------------------===//

  // $past with enable, delay, and explicit clocking using different clock
  property past_enable_different_clock;
    @(posedge clk) a |-> $past(data, 2, enable, @(posedge clk2)) == 8'h00;
  endproperty
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_enable_different_clock);

endmodule
