// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && CIRCT_UVM_ARGS="+VERBOSE +DEBUG=3" circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $test$plusargs — must detect plusargs from CIRCT_UVM_ARGS
module top;
  initial begin
    // +VERBOSE is in CIRCT_UVM_ARGS, so $test$plusargs must return 1
    if ($test$plusargs("VERBOSE"))
      // CHECK: verbose_found
      $display("verbose_found");
    else
      $display("verbose_not_found");

    // +DEBUG=3 is in CIRCT_UVM_ARGS, so $test$plusargs("DEBUG") must return 1
    if ($test$plusargs("DEBUG"))
      // CHECK: debug_found
      $display("debug_found");
    else
      $display("debug_not_found");

    // MISSING is NOT in CIRCT_UVM_ARGS — should return 0
    if ($test$plusargs("MISSING"))
      $display("missing_found");
    else
      // CHECK: missing_not_found
      $display("missing_not_found");

    $finish;
  end
endmodule
