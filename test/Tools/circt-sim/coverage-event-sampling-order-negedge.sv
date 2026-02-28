// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: implicit covergroup event sampling must observe signal values
// after Active-region procedural updates at the same clock edge.

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  logic [1:0] addr = 0;

  covergroup cg @(negedge clk);
    cp: coverpoint addr {
      bins z = {0};
      bins o = {1};
    }
  endgroup

  cg cov = new;

  initial begin
    repeat (2) begin
      @(negedge clk);
      addr = 1;
    end
    @(posedge clk);
    #1;
    $display("COV=%0.2f", cov.cp.get_inst_coverage());
    $finish;
  end
endmodule

// CHECK: COV=50.00
