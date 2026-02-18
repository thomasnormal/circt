// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $value$plusargs and $test$plusargs
module top;
  string s;
  integer found;
  integer val;

  initial begin
    // $test$plusargs returns 1 if the plusarg prefix is found
    // Since we don't pass any plusargs, this should return 0
    found = $test$plusargs("TESTARG");
    // CHECK: test_plusargs=0
    $display("test_plusargs=%0d", found);

    // $value$plusargs returns 1 if found and extracts value
    found = $value$plusargs("MYVAL=%d", val);
    // CHECK: value_plusargs=0
    $display("value_plusargs=%0d", found);

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
