// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $cast system function
module top;
  typedef enum {A=0, B=1, C=2} abc_t;
  abc_t e;
  int val;
  int result;

  initial begin
    // $cast returns 1 on success, 0 on failure
    val = 1;
    result = $cast(e, val);
    // CHECK: cast_result=1
    $display("cast_result=%0d", result);
    // CHECK: e_val=1
    $display("e_val=%0d", e);

    // Cast with invalid value
    val = 99;
    result = $cast(e, val);
    // CHECK: cast_invalid=0
    $display("cast_invalid=%0d", result);

    $finish;
  end
endmodule
