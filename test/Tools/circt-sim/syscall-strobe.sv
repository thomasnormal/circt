// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: $strobe shows val=10 (immediate) instead of val=20 (end-of-timestep).
// Test $strobe — like $display but prints at end of time step
module top;
  reg [7:0] val;

  initial begin
    val = 10;
    $strobe("strobe: val=%0d", val);
    val = 20;
    // $strobe should show the value at end of time step (20, not 10)
    // CHECK: strobe: val=20
    #1;
    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
