// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  logic [7:0] simple;
  logic [3:0][7:0] packed_2d;
  logic [7:0] unpacked_1d [0:3];
  logic [7:0] mixed [0:1][0:2];

  initial begin
    // $dimensions returns total number of dimensions (packed + unpacked)
    // CHECK: dim_simple=1
    $display("dim_simple=%0d", $dimensions(simple));

    // CHECK: dim_packed_2d=2
    $display("dim_packed_2d=%0d", $dimensions(packed_2d));

    // CHECK: dim_unpacked_1d=2
    $display("dim_unpacked_1d=%0d", $dimensions(unpacked_1d));

    // CHECK: dim_mixed=3
    $display("dim_mixed=%0d", $dimensions(mixed));

    $finish;
  end
endmodule
