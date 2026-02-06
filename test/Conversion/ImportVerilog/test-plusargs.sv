// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @testPlusargs
module testPlusargs();

  // CHECK: procedure initial
  initial begin
    // CHECK: llvm.call @__moore_test_plusargs
    if ($test$plusargs("MY_TEST"))
      $display("found");
  end
endmodule
