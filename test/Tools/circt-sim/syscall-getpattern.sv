// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $getpattern — legacy function, returns 0 (not implemented)
module top;
  reg [31:0] pattern;

  initial begin
    // $getpattern is a legacy function that returns 0
    pattern = $getpattern("pattern_test.dat");
    // CHECK: pattern=0
    $display("pattern=%0d", pattern);
    $finish;
  end
endmodule
