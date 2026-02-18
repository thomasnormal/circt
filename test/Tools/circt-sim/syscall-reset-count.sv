// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  initial begin
    // $reset_count returns the number of times $reset has been called
    // Initially should be 0
    // CHECK: reset_count=0
    $display("reset_count=%0d", $reset_count);
    $finish;
  end
endmodule
