// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $test$plusargs in more detail
module top;
  initial begin
    // With no plusargs passed, everything should return 0
    if ($test$plusargs("VERBOSE"))
      $display("verbose_found");
    else
      // CHECK: verbose_not_found
      $display("verbose_not_found");

    if ($test$plusargs("DEBUG"))
      $display("debug_found");
    else
      // CHECK: debug_not_found
      $display("debug_not_found");

    $finish;
  end
endmodule
