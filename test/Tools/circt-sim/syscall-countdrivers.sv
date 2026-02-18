// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  wire net_a;
  reg d1, d2;
  integer result;

  assign net_a = d1;
  assign net_a = d2;

  initial begin
    d1 = 1;
    d2 = 0;
    // $countdrivers returns 1 if more than one driver is active
    result = $countdrivers(net_a);
    // CHECK: countdrivers=1
    $display("countdrivers=%0d", result);
    $finish;
  end
endmodule
