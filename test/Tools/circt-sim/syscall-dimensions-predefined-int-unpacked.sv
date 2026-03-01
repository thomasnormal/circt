// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression for issue #39: $dimensions must not add a fake packed dimension
// for predefined integer scalar element types.
module top;
  int iarr[8];
  shortint sarr[3][4];
  byte barr[2];
  logic [7:0] larr[4];

  initial begin
    // CHECK: dim_iarr=1
    $display("dim_iarr=%0d", $dimensions(iarr));
    // CHECK: dim_sarr=2
    $display("dim_sarr=%0d", $dimensions(sarr));
    // CHECK: dim_barr=1
    $display("dim_barr=%0d", $dimensions(barr));
    // Control: packed+unpacked remains unchanged for logic vectors.
    // CHECK: dim_larr=2
    $display("dim_larr=%0d", $dimensions(larr));
    $finish;
  end
endmodule
