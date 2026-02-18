// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test $isunknown — returns 1 if any bit is x or z
module top;
  logic [7:0] a;

  initial begin
    // Known value — should return 0
    a = 8'hFF;
    // CHECK: isunknown_ff=0
    $display("isunknown_ff=%0d", $isunknown(a));

    // Known zero — should return 0
    a = 8'h00;
    // CHECK: isunknown_00=0
    $display("isunknown_00=%0d", $isunknown(a));

    // Value with x bit — should return 1
    a = 8'b1010x010;
    // CHECK: isunknown_x=1
    $display("isunknown_x=%0d", $isunknown(a));

    // Value with z bit — should return 1
    a = 8'b1010z010;
    // CHECK: isunknown_z=1
    $display("isunknown_z=%0d", $isunknown(a));

    $finish;
  end
endmodule
