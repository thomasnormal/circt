// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test $value$plusargs and $test$plusargs (IEEE 1800-2017 Section 21.6)
// These are stubbed to return 0 (not found) since command line argument
// handling is not implemented.

module plusargs_test;
  integer i;
  string s;

  initial begin
    // $value$plusargs returns 0 when arg not found, 1 otherwise
    if ($value$plusargs("TEST=%d", i)) begin
      $display("TEST found: %d", i);
    end else begin
      $display("TEST not found");
    end

    // $test$plusargs just checks if arg is present (returns 0 when not found)
    if ($test$plusargs("DEBUG"))
      $display("DEBUG mode enabled");
  end
endmodule

// CHECK: moore.module @plusargs_test
// CHECK: moore.constant 0 : i32
