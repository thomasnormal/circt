// RUN: circt-verilog %s --ir-hw | CIRCT_UVM_ARGS="+MY_TEST +VERBOSE" circt-sim --top test - | FileCheck %s
// REQUIRES: slang

module test();
  initial begin
    // CHECK: MY_TEST=1
    if ($test$plusargs("MY_TEST"))
      $display("MY_TEST=1");
    else
      $display("MY_TEST=0");

    // CHECK: VERBOSE=1
    if ($test$plusargs("VERBOSE"))
      $display("VERBOSE=1");
    else
      $display("VERBOSE=0");

    // CHECK: MISSING=0
    if ($test$plusargs("MISSING"))
      $display("MISSING=1");
    else
      $display("MISSING=0");
  end
endmodule
