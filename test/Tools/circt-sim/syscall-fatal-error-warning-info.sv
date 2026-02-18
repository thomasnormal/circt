// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $info, $warning, $error severity messaging
module top;
  initial begin
    // $info should print an informational message
    // CHECK: info_msg
    $info("info_msg");

    // $warning should print a warning
    // CHECK: warn_msg
    $warning("warn_msg");

    // $error should print an error
    // CHECK: error_msg
    $error("error_msg");

    // Note: $fatal would terminate simulation, so we don't test it here
    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
