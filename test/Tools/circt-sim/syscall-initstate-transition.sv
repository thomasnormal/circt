// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $initstate transitions from 1 to 0 exactly once.
// Bug: $initstate is stuck at one value (always 1 or always 0).
// IEEE 1800-2017 Section 20.14: Returns 1 during init phase, 0 after.
//
// This test captures the transition by sampling at multiple time points.
module top;
  integer t0_val, t1_val, t100_val;

  initial begin
    // Time 0: initialization phase
    t0_val = $initstate;
    // CHECK: t0=1
    $display("t0=%0d", t0_val);

    // Advance 1 time unit
    #1;
    t1_val = $initstate;
    // CHECK: t1=0
    $display("t1=%0d", t1_val);

    // Advance more
    #99;
    t100_val = $initstate;
    // CHECK: t100=0
    $display("t100=%0d", t100_val);

    // The transition should have happened: 1 → 0
    // CHECK: transition=1
    $display("transition=%0d", (t0_val == 1) && (t1_val == 0));

    $finish;
  end
endmodule
