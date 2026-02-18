// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  initial begin
    // $reset_value returns the value passed to the last $reset call
    // Initially should be 0
    // CHECK: reset_value=0
    $display("reset_value=%0d", $reset_value);
    $finish;
  end
endmodule
