// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $initstate returns 0 after simulation time advances.
// Bug: $initstate is stubbed to always return 1 (or always 0).
// IEEE 1800-2017 Section 20.14: $initstate returns 1 during
// initialization (time 0, before any time advance), 0 afterwards.
//
// This test checks via a function call and via direct reference,
// both before and after a time advance.
module top;
  function int check_initstate();
    return $initstate;
  endfunction

  initial begin
    // At time 0, before any delay: $initstate should be 1
    // CHECK: direct_t0=1
    $display("direct_t0=%0d", $initstate);

    // Via function call at time 0: should also be 1
    // CHECK: func_t0=1
    $display("func_t0=%0d", check_initstate());

    // Advance time
    #10;

    // After time advance: $initstate should be 0
    // CHECK: direct_t10=0
    $display("direct_t10=%0d", $initstate);

    // Via function call after time advance: should also be 0
    // CHECK: func_t10=0
    $display("func_t10=%0d", check_initstate());

    $finish;
  end
endmodule
