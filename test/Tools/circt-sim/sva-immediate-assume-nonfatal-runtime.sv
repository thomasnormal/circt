// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=20000000 2>&1 | FileCheck %s
// CHECK: SVA assumption failed at time
// CHECK: Simulation completed

module top;
  reg a;

  initial begin
    a = 1'b0;
    assume (a);
    assume #0 (a);
    assume final (a);
    #1;
    $finish;
  end
endmodule
