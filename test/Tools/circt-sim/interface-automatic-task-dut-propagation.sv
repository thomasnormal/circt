// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=100000000 2>&1 | FileCheck %s

// Regression for issue #20:
// Writes to interface fields from automatic tasks must drive DUT ports, and
// DUT output updates must be visible when reading interface fields in tasks.

interface mem_if (input logic clk);
  logic we;
  logic [3:0] addr;
  logic [7:0] wdata, rdata;
endinterface

module sram(
    input logic clk,
    input logic we,
    input logic [3:0] addr,
    input logic [7:0] wdata,
    output logic [7:0] rdata);
  logic [7:0] mem [0:15];
  always_ff @(posedge clk) begin
    if (we)
      mem[addr] <= wdata;
    rdata <= mem[addr];
  end
endmodule

module tb;
  logic clk = 0;
  always #5 clk = ~clk;

  mem_if vif(.clk(clk));
  sram dut(
      .clk(vif.clk), .we(vif.we), .addr(vif.addr),
      .wdata(vif.wdata), .rdata(vif.rdata));

  task automatic write_word(input logic [3:0] a, input logic [7:0] data);
    vif.we = 1;
    vif.addr = a;
    vif.wdata = data;
    @(posedge vif.clk);
    #1;
    vif.we = 0;
  endtask

  task automatic read_word(input logic [3:0] a, output logic [7:0] data);
    vif.addr = a;
    @(posedge vif.clk);
    #1;
    data = vif.rdata;
  endtask

  initial begin
    logic [7:0] result;
    write_word(4'd5, 8'd42);
    read_word(4'd5, result);
    if (result === 8'd42)
      $display("PASS");
    else
      $display("FAIL: got %0d", result);
    $finish;
  end

  // CHECK: PASS
endmodule
