// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $monitor and $monitoroff / $monitoron
module top;
  reg [7:0] val;

  initial begin
    $monitor("mon: val=%0d", val);
    val = 1;
    #1;
    // CHECK: mon: val=1
    val = 2;
    #1;
    // CHECK: mon: val=2

    // Turn off monitoring
    $monitoroff;
    // CHECK: monitoroff_called
    $display("monitoroff_called");
    val = 3;
    #1;
    // Should NOT see val=3 from monitor
    // CHECK-NOT: mon: val=3
    // CHECK: no_mon_3
    $display("no_mon_3");

    // Turn monitoring back on
    $monitoron;
    val = 4;
    #1;
    // CHECK: mon: val=4

    $finish;
  end
endmodule
