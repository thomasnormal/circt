// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $countdrivers — returns 0 (single-driver assumption)
module top;
  wire net_a;
  reg d1;
  integer result;

  assign net_a = d1;

  initial begin
    d1 = 1;
    // $countdrivers returns 0 for single-driver nets
    result = $countdrivers(net_a);
    // CHECK: countdrivers=0
    $display("countdrivers=%0d", result);
    $finish;
  end
endmodule
