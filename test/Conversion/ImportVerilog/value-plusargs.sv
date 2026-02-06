// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test $value$plusargs and $test$plusargs (IEEE 1800-2017 Section 21.6)

module plusargs_test;
  integer i;
  string s;

  initial begin
    // $value$plusargs returns 0 when arg not found, 1 otherwise
    // Currently stubbed to always return 0.
    if ($value$plusargs("TEST=%d", i)) begin
      $display("TEST found: %d", i);
    end else begin
      $display("TEST not found");
    end

    // $test$plusargs emits a runtime call to __moore_test_plusargs
    if ($test$plusargs("DEBUG"))
      $display("DEBUG mode enabled");
  end
endmodule

// CHECK: moore.module @plusargs_test
// $value$plusargs still stubbed to constant 0
// CHECK: moore.constant 0 : i32
// $test$plusargs emits runtime call
// CHECK: llvm.call @__moore_test_plusargs
