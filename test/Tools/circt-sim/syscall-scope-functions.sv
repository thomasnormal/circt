// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test scope hierarchy functions: $root, hierarchical references
module top;
  reg [7:0] data = 42;

  sub_mod u_sub();

  initial begin
    #1;
    // CHECK: top_data=42
    $display("top_data=%0d", data);
    // CHECK: sub_data=99
    $display("sub_data=%0d", u_sub.sub_data);
    $finish;
  end
endmodule

module sub_mod;
  reg [7:0] sub_data = 99;
endmodule
