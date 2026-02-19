// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $showvars — display information about variables
module top;
  integer x = 42;
  reg [7:0] y = 8'hAB;

  initial begin
    // $showvars should print variable names and values
    // CHECK-DAG: x
    // CHECK-DAG: 42
    // CHECK-DAG: y
    $showvars(x, y);
    $finish;
  end
endmodule
