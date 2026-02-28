// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: array-element blocking drives populate pendingEpsilonDrives for
// same-process visibility, but those pending entries must be cleared once the
// scheduled commit fires. Otherwise later NBA array writes are masked by stale
// pending state and reads observe old array values.

`timescale 1ns/1ns
module top;
  logic clk = 0;
  logic we = 0;
  logic [3:0] addr = 0;
  logic [7:0] wdata = 0;
  logic [7:0] rdata;
  logic [7:0] mem [0:15];
  integer i;

  always #5 clk = ~clk;

  initial begin
    for (i = 0; i < 16; i = i + 1)
      mem[i] = i[7:0];
  end

  always @(posedge clk) begin
    if (we)
      mem[addr] <= wdata;
    rdata <= mem[addr];
  end

  initial begin
    @(negedge clk);
    we = 1'b1;
    addr = 4'd7;
    wdata = 8'h3a;

    @(posedge clk);
    @(negedge clk);
    we = 1'b0;
    addr = 4'd7;
    wdata = 8'haa;

    @(posedge clk);
    @(posedge clk);

    $display("RESULT rdata=%02x mem7=%02x", rdata, mem[7]);
    // CHECK: RESULT rdata=3a mem7=3a
    #1 $finish;
  end
endmodule
