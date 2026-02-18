// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  logic [7:0] no_unpacked;
  logic [7:0] one_unpacked [0:3];
  logic [7:0] two_unpacked [0:1][0:2];

  initial begin
    // $unpacked_dimensions returns number of unpacked dimensions
    // CHECK: udim_none=0
    $display("udim_none=%0d", $unpacked_dimensions(no_unpacked));

    // CHECK: udim_one=1
    $display("udim_one=%0d", $unpacked_dimensions(one_unpacked));

    // CHECK: udim_two=2
    $display("udim_two=%0d", $unpacked_dimensions(two_unpacked));

    $finish;
  end
endmodule
