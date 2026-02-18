// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $left, $right, $low, $high, $size, $increment on packed/unpacked arrays
module top;
  logic [7:0] packed_arr;           // [7:0]
  logic arr_asc [0:4];             // ascending [0:4]
  logic arr_desc [4:0];            // descending [4:0]
  logic [3:0][7:0] packed_2d;     // 2D packed

  initial begin
    // Packed [7:0]
    // CHECK: p_left=7
    $display("p_left=%0d", $left(packed_arr));
    // CHECK: p_right=0
    $display("p_right=%0d", $right(packed_arr));
    // CHECK: p_low=0
    $display("p_low=%0d", $low(packed_arr));
    // CHECK: p_high=7
    $display("p_high=%0d", $high(packed_arr));
    // CHECK: p_size=8
    $display("p_size=%0d", $size(packed_arr));

    // Ascending unpacked [0:4]
    // CHECK: a_left=0
    $display("a_left=%0d", $left(arr_asc));
    // CHECK: a_right=4
    $display("a_right=%0d", $right(arr_asc));
    // CHECK: a_size=5
    $display("a_size=%0d", $size(arr_asc));

    // Descending unpacked [4:0]
    // CHECK: d_left=4
    $display("d_left=%0d", $left(arr_desc));
    // CHECK: d_right=0
    $display("d_right=%0d", $right(arr_desc));
    // CHECK: d_size=5
    $display("d_size=%0d", $size(arr_desc));

    $finish;
  end
endmodule
