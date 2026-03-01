// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #29: clocking blocks inside interfaces can reference
// interface ports in their event expression.

interface my_if (input logic clk);
  logic [7:0] data;
  logic valid;
  clocking cb @(posedge clk);
    input data, valid;
  endclocking
endinterface

module tb;
  logic clk = 0;
  my_if ifc(.clk(clk));

  always #5 clk = ~clk;

  initial begin
    ifc.data = 8'hAB;
    ifc.valid = 1'b1;
    @(ifc.cb);
    $display("PASS");
    $finish;
  end

  // CHECK: PASS
endmodule
