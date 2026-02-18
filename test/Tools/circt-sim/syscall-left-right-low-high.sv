// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $left, $right, $low, $high array query functions
module top;
  logic [7:0] ascending [0:3];
  logic [7:0] descending [3:0];
  logic [15:8] packed_range;

  initial begin
    // $left returns leftmost index of the dimension
    // CHECK: left_asc=0
    $display("left_asc=%0d", $left(ascending));
    // CHECK: left_desc=3
    $display("left_desc=%0d", $left(descending));

    // $right returns rightmost index
    // CHECK: right_asc=3
    $display("right_asc=%0d", $right(ascending));
    // CHECK: right_desc=0
    $display("right_desc=%0d", $right(descending));

    // $low returns minimum index
    // CHECK: low_asc=0
    $display("low_asc=%0d", $low(ascending));
    // CHECK: low_desc=0
    $display("low_desc=%0d", $low(descending));

    // $high returns maximum index
    // CHECK: high_asc=3
    $display("high_asc=%0d", $high(ascending));
    // CHECK: high_desc=3
    $display("high_desc=%0d", $high(descending));

    $finish;
  end
endmodule
