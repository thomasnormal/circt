// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Ensure $past in comparisons lowers without conversion casts in --ir-hw.
module PastCompare(input logic clk, a, b);
  property past_eq;
    @(posedge clk) a |=> b == $past(a);
  endproperty
  assert property (past_eq);

  // CHECK-LABEL: hw.module @PastCompare
  // CHECK: seq.compreg {{.*}} initial
  // CHECK: ltl.delay
  // CHECK: ltl.implication
  // CHECK: verif.clocked_assert
endmodule
