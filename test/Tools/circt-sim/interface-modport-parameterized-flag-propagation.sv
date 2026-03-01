// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=100000000 2>&1 | FileCheck %s

// Regression for issue #32: writes through modports on parameterized interface
// ports must propagate correctly for single-bit members.

interface plain_if(input logic clk);
  logic [15:0] data;
  logic flag;
  modport src(output data, flag, input clk);
endinterface

interface pif #(parameter int W = 8)(input logic clk);
  logic [W-1:0] data;
  logic flag;
  modport src(output data, flag, input clk);
endinterface

module writer_plain(plain_if.src ifc);
  initial begin
    #1;
    ifc.data = 16'hABCD;
    ifc.flag = 1'b1;
  end
endmodule

module writer_param(pif.src ifc);
  initial begin
    #1;
    ifc.data = 'hDEAD;
    ifc.flag = 1'b1;
  end
endmodule

module tb;
  logic clk = 0;
  plain_if p(.clk(clk));
  pif #(.W(16)) q(.clk(clk));

  writer_plain wp(.ifc(p.src));
  writer_param wq(.ifc(q.src));

  initial begin
    #5;
    $display("plain: data=%h flag=%b", p.data, p.flag);
    $display("param: data=%h flag=%b", q.data, q.flag);
    $finish;
  end

  // CHECK: plain: data=abcd flag=1
  // CHECK: param: data=dead flag=1
endmodule
